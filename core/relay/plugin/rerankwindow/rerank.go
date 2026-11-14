// Package rerankwindow expands text documents before provider adaptation and
// aggregates normalized scores afterwards, so batching is independent of providers.
package rerankwindow

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"math"
	"net/http"
	"net/url"
	"slices"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/labring/aiproxy/core/common"
	"github.com/labring/aiproxy/core/model"
	"github.com/labring/aiproxy/core/relay/adaptor"
	"github.com/labring/aiproxy/core/relay/meta"
	"github.com/labring/aiproxy/core/relay/mode"
	relaymodel "github.com/labring/aiproxy/core/relay/model"
	"github.com/labring/aiproxy/core/relay/plugin"
	"github.com/labring/aiproxy/core/relay/plugin/noop"
	"github.com/labring/aiproxy/core/relay/utils"
)

const stateKey = "rerank_window_state"

type RerankWindow struct{ noop.Noop }

var _ plugin.Plugin = (*RerankWindow)(nil)

// NewPlugin adds opt-in long-document scoring without changing the public API.
func NewPlugin() plugin.Plugin { return &RerankWindow{} }

type requestState struct {
	Config      Config
	Fields      map[string]json.RawMessage
	Request     relaymodel.RerankRequest
	Windows     []window
	QueryTokens int
	Deadline    time.Time
	Response    []byte
	Result      adaptor.DoResponseResult
	TEI         bool
}

func getState(m *meta.Meta) *requestState {
	v, _ := m.Get(stateKey)
	s, _ := v.(*requestState)
	return s
}

// ConvertRequest tokenizes the full input before any scoring. Query text is never
// truncated; invalid configuration or document limits fail the entire request.
func (p *RerankWindow) ConvertRequest(m *meta.Meta, store adaptor.Store, req *http.Request,
	do adaptor.ConvertRequest) (adaptor.ConvertResult, error) {
	if m.Mode != mode.Rerank {
		return do.ConvertRequest(m, store, req)
	}
	cfg := defaultConfig()
	if err := m.ModelConfig.LoadPluginConfig("rerank_window", &cfg); err != nil {
		return adaptor.ConvertResult{}, err
	}
	if !cfg.Enable {
		return do.ConvertRequest(m, store, req)
	}
	if err := cfg.validate(); err != nil {
		return adaptor.ConvertResult{}, err
	}

	s := &requestState{Config: cfg, Deadline: time.Now().Add(time.Duration(cfg.TimeoutSeconds) * time.Second), TEI: m.Channel.Type == model.ChannelTypeTextEmbeddingsInference}
	body, err := common.GetRequestBodyReusable(req)
	if err != nil {
		return adaptor.ConvertResult{}, err
	}
	if err := json.Unmarshal(body, &s.Request); err != nil {
		return adaptor.ConvertResult{}, fmt.Errorf("rerank_window supports string query and documents: %w", err)
	}
	if err := json.Unmarshal(body, &s.Fields); err != nil {
		return adaptor.ConvertResult{}, err
	}
	if s.Request.Query == "" || len(s.Request.Documents) == 0 || len(s.Request.Documents) > cfg.MaxWindows {
		return adaptor.ConvertResult{}, fmt.Errorf("query and nonempty documents are required (at most %d documents)", cfg.MaxWindows)
	}
	if s.Request.TopN != nil && *s.Request.TopN < 1 {
		return adaptor.ConvertResult{}, fmt.Errorf("top_n must be positive")
	}
	if s.Request.MaxChunksPerDoc != nil && *s.Request.MaxChunksPerDoc < 1 {
		return adaptor.ConvertResult{}, fmt.Errorf("max_chunks_per_doc must be positive")
	}
	if s.Request.OverlapTokens != nil {
		cfg.OverlapTokens = *s.Request.OverlapTokens
		if err := cfg.validate(); err != nil {
			return adaptor.ConvertResult{}, err
		}
		s.Config = cfg
	}

	ctx, cancel := context.WithDeadline(req.Context(), s.Deadline)
	defer cancel()
	tokenize := newTokenizer(m, cfg)
	query, err := tokenize(ctx, []string{s.Request.Query})
	if err != nil {
		return adaptor.ConvertResult{}, err
	}
	s.QueryTokens = len(query[0])
	budget := cfg.MaxTokens - cfg.SpecialTokens - s.QueryTokens
	if budget-cfg.SafetyTokens < 8 {
		return adaptor.ConvertResult{}, fmt.Errorf("query exceeds rerank_window token budget")
	}

	// 分批分词，避免长文的 token/offset 响应同时占用过多内存。
	for start := 0; start < len(s.Request.Documents); start += 8 {
		end := min(start+8, len(s.Request.Documents))
		docTokens, err := tokenize(ctx, s.Request.Documents[start:end])
		if err != nil {
			return adaptor.ConvertResult{}, err
		}
		for i, tokens := range docTokens {
			windows, err := splitWindows(s.Request.Documents[start+i], splitOptions{Tokens: tokens, Config: cfg, Document: start + i, Budget: budget})
			if err != nil {
				return adaptor.ConvertResult{}, err
			}
			if s.Request.MaxChunksPerDoc != nil && len(windows) > *s.Request.MaxChunksPerDoc {
				return adaptor.ConvertResult{}, fmt.Errorf("document %d requires %d windows, exceeding max_chunks_per_doc", start+i, len(windows))
			}
			s.Windows = append(s.Windows, windows...)
			if len(s.Windows) > cfg.MaxWindows {
				return adaptor.ConvertResult{}, fmt.Errorf("request exceeds max_windows=%d", cfg.MaxWindows)
			}
		}
	}

	// 切片边界可能改变 SentencePiece 分词；再次校验实际窗口，禁止上游悄悄截断。
	for start := 0; start < len(s.Windows); start += cfg.BatchSize {
		end := min(start+cfg.BatchSize, len(s.Windows))
		texts := make([]string, end-start)
		for i := range texts {
			texts[i] = s.Windows[start+i].Text
		}
		verified, err := tokenize(ctx, texts)
		if err != nil {
			return adaptor.ConvertResult{}, err
		}
		for i, tokens := range verified {
			if len(tokens) > budget {
				return adaptor.ConvertResult{}, fmt.Errorf("window token budget exceeded; increase safety_tokens")
			}
			s.Windows[start+i].Tokens = len(tokens) + s.QueryTokens + cfg.SpecialTokens
		}
	}
	m.Set(stateKey, s)
	batch, err := s.batchRequest(req, [2]int{0, min(cfg.BatchSize, len(s.Windows))})
	if err != nil {
		return adaptor.ConvertResult{}, err
	}
	defer batch.Body.Close()
	return do.ConvertRequest(m, store, batch)
}

// newTokenizer uses exact TEI byte offsets for any provider backed by a compatible
// tokenizer endpoint. Channel credentials are sent only to the same origin.
func newTokenizer(m *meta.Meta, cfg Config) func(context.Context, []string) ([][]token, error) {
	return func(ctx context.Context, texts []string) ([][]token, error) {
		endpoint := cfg.TokenizerURL
		if endpoint == "" {
			if m.Channel.Type != model.ChannelTypeTextEmbeddingsInference {
				return nil, fmt.Errorf("rerank_window requires tokenizer_url for this channel")
			}
			var err error
			endpoint, err = url.JoinPath(m.Channel.BaseURL, "tokenize")
			if err != nil {
				return nil, err
			}
		}
		parsed, err := url.Parse(endpoint)
		if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" || parsed.User != nil {
			return nil, fmt.Errorf("invalid tokenizer_url")
		}
		payload, err := json.Marshal(map[string]any{"inputs": texts, "add_special_tokens": false})
		if err != nil {
			return nil, err
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "application/json")
		base, _ := url.Parse(m.Channel.BaseURL)
		if base != nil && base.Scheme == parsed.Scheme && base.Host == parsed.Host && m.Channel.Key != "" {
			req.Header.Set("Authorization", "Bearer "+m.Channel.Key)
		}
		client, err := utils.LoadHTTPClientWithTLSConfigE(30*time.Second, m.Channel.ProxyURL, m.Channel.SkipTLSVerify)
		if err != nil {
			return nil, err
		}
		// 禁止分词端点重定向，避免跨域转发渠道凭证。
		tokenClient := *client
		tokenClient.CheckRedirect = func(_ *http.Request, _ []*http.Request) error { return http.ErrUseLastResponse }
		resp, err := tokenClient.Do(req)
		if err != nil {
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				return nil, err
			}
			return nil, relaymodel.WrapperErrorWithMessage(m.Mode, http.StatusBadGateway, "tokenizer request failed: "+err.Error())
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return nil, relaymodel.WrapperErrorWithMessage(m.Mode, http.StatusBadGateway, fmt.Sprintf("tokenizer returned HTTP %d", resp.StatusCode))
		}
		body, err := common.GetResponseBodyLimit(resp, 64<<20)
		if err != nil {
			return nil, err
		}
		var tokens [][]token
		if err := json.Unmarshal(body, &tokens); err != nil {
			return nil, fmt.Errorf("invalid tokenizer response: %w", err)
		}
		if len(tokens) != len(texts) {
			return nil, fmt.Errorf("tokenizer result count mismatch")
		}
		return tokens, nil
	}
}

// batchRequest preserves provider options but applies top_n only after document
// aggregation. Every window must receive a score before a response is committed.
func (s *requestState) batchRequest(original *http.Request, bounds [2]int) (*http.Request, error) {
	fields := maps.Clone(s.Fields)
	for _, key := range []string{"top_n", "max_chunks_per_doc", "overlap_tokens"} {
		delete(fields, key)
	}
	fields["return_documents"] = json.RawMessage("false")
	texts := make([]string, bounds[1]-bounds[0])
	for i := range texts {
		texts[i] = s.Windows[bounds[0]+i].Text
	}
	documents, err := json.Marshal(texts)
	if err != nil {
		return nil, err
	}
	fields["documents"] = documents
	// TEI 用 truncate=false 防止配置或 tokenizer 不匹配时返回不完整评分。
	if s.TEI {
		fields["truncate"] = json.RawMessage("false")
	}
	body, err := json.Marshal(fields)
	if err != nil {
		return nil, err
	}
	req := original.Clone(original.Context())
	req.Body = io.NopCloser(bytes.NewReader(body))
	req.GetBody = func() (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(body)), nil }
	req.ContentLength = int64(len(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Del("Content-Length")
	req.Header.Del("Content-Encoding")
	return req, nil
}

// DoRequest runs bounded sequential batches through the existing provider adapter.
// Partial failures never expose partial rankings; billing and outer cache see one result.
func (p *RerankWindow) DoRequest(m *meta.Meta, store adaptor.Store, c *gin.Context,
	req *http.Request, do adaptor.DoRequest) (*http.Response, error) {
	s := getState(m)
	if s == nil {
		return do.DoRequest(m, store, c, req)
	}
	inner, ok := do.(adaptor.Adaptor)
	if !ok {
		return nil, fmt.Errorf("rerank_window requires a complete adapter")
	}
	ctx, cancel := context.WithDeadline(c.Request.Context(), s.Deadline)
	defer cancel()
	scores := make([]float64, len(s.Request.Documents))
	for i := range scores {
		scores[i] = math.Inf(-1)
	}
	for start := 0; start < len(s.Windows); start += s.Config.BatchSize {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		end := min(start+s.Config.BatchSize, len(s.Windows))
		batchReq, err := s.batchRequest(c.Request.WithContext(ctx), [2]int{start, end})
		if err != nil {
			return nil, err
		}
		batchMeta := m.Fork()
		batchMeta.RequestUsage = model.Usage{}
		for _, w := range s.Windows[start:end] {
			batchMeta.RequestUsage.InputTokens += model.ZeroNullInt64(w.Tokens)
		}
		batchMeta.RequestUsage.TotalTokens = batchMeta.RequestUsage.InputTokens
		batchContext := c.Copy()
		batchContext.Request = batchReq
		writer := &captureWriter{ResponseWriter: c.Writer, headers: make(http.Header), status: http.StatusOK}
		batchContext.Writer = writer
		result, err := scoreBatch(inner, batchCall{Meta: batchMeta, Store: store, Context: batchContext, Request: req.WithContext(ctx)})
		_ = batchReq.Body.Close()
		if err != nil {
			return nil, err
		}
		var response relaymodel.RerankResponse
		if err := json.Unmarshal(writer.body.Bytes(), &response); err != nil {
			return nil, fmt.Errorf("invalid normalized rerank response: %w", err)
		}
		if len(response.Results) != end-start {
			return nil, fmt.Errorf("upstream omitted window scores")
		}
		seen := make([]bool, end-start)
		for _, item := range response.Results {
			if item == nil || item.Index < 0 || item.Index >= len(seen) || seen[item.Index] || math.IsNaN(item.RelevanceScore) || math.IsInf(item.RelevanceScore, 0) {
				return nil, fmt.Errorf("invalid or duplicate window score index")
			}
			seen[item.Index] = true
			id := s.Windows[start+item.Index].Document
			scores[id] = max(scores[id], item.RelevanceScore)
		}
		s.Result.Usage.Add(result.Usage)
	}

	results := make([]*relaymodel.RerankResult, len(scores))
	for id, score := range scores {
		item := &relaymodel.RerankResult{Index: id, RelevanceScore: score}
		if s.Request.ReturnDocuments != nil && *s.Request.ReturnDocuments {
			item.Document = &relaymodel.Document{Text: s.Request.Documents[id]}
		}
		results[id] = item
	}
	slices.SortStableFunc(results, func(a, b *relaymodel.RerankResult) int {
		if a.RelevanceScore > b.RelevanceScore {
			return -1
		}
		if a.RelevanceScore < b.RelevanceScore {
			return 1
		}
		return 0
	})
	if s.Request.TopN != nil && *s.Request.TopN < len(results) {
		results = results[:*s.Request.TopN]
	}
	response, err := json.Marshal(relaymodel.RerankResponse{ID: m.RequestID, Results: results,
		Meta: relaymodel.RerankMeta{Model: m.OriginModel, Tokens: &relaymodel.RerankMetaTokens{InputTokens: int64(s.Result.Usage.InputTokens), OutputTokens: int64(s.Result.Usage.OutputTokens)}}})
	if err != nil {
		return nil, err
	}
	s.Response = response
	return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(bytes.NewReader(nil))}, nil
}

type batchCall struct {
	Meta    *meta.Meta
	Store   adaptor.Store
	Context *gin.Context
	Request *http.Request
}

// scoreBatch keeps provider conversion, authentication and usage accounting intact.
func scoreBatch(inner adaptor.Adaptor, call batchCall) (adaptor.DoResponseResult, error) {
	converted, err := inner.ConvertRequest(call.Meta, call.Store, call.Context.Request)
	if err != nil {
		return adaptor.DoResponseResult{}, err
	}
	body, err := io.ReadAll(converted.Body)
	if closer, ok := converted.Body.(io.Closer); ok {
		_ = closer.Close()
	}
	if err != nil {
		return adaptor.DoResponseResult{}, err
	}
	req := call.Request.Clone(call.Request.Context())
	req.Body = io.NopCloser(bytes.NewReader(body))
	req.GetBody = func() (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(body)), nil }
	req.ContentLength = int64(len(body))
	req.Header.Del("Content-Length")
	maps.Copy(req.Header, converted.Header)
	defer req.Body.Close()
	if err := inner.SetupRequestHeader(call.Meta, call.Store, call.Context, req); err != nil {
		return adaptor.DoResponseResult{}, err
	}
	resp, err := inner.DoRequest(call.Meta, call.Store, call.Context, req)
	if err != nil {
		return adaptor.DoResponseResult{}, err
	}
	if resp == nil {
		return adaptor.DoResponseResult{}, fmt.Errorf("upstream response is nil")
	}
	if resp.Body != nil {
		defer resp.Body.Close()
	}
	result, relayErr := inner.DoResponse(call.Meta, call.Store, call.Context, resp)
	if relayErr != nil {
		return result, relayErr
	}
	return result, nil
}

// DoResponse commits only the fully aggregated response, leaving outer cache and
// tenant monitoring hooks to account for the combined request once.
func (p *RerankWindow) DoResponse(m *meta.Meta, store adaptor.Store, c *gin.Context,
	resp *http.Response, do adaptor.DoResponse) (adaptor.DoResponseResult, adaptor.Error) {
	s := getState(m)
	if s == nil || s.Response == nil {
		return do.DoResponse(m, store, c, resp)
	}
	c.Header("Content-Type", "application/json")
	c.Header("Content-Length", strconv.Itoa(len(s.Response)))
	c.Header("X-Aiproxy-Rerank-Windows", strconv.Itoa(len(s.Windows)))
	if _, err := c.Writer.Write(s.Response); err != nil {
		common.GetLogger(c).Warnf("write windowed rerank response failed: %v", err)
	}
	return s.Result, nil
}

// captureWriter buffers a normalized adapter response without writing any headers
// or bytes to the client before all upstream batches have succeeded.
type captureWriter struct {
	gin.ResponseWriter
	headers http.Header
	body    bytes.Buffer
	status  int
}

func (w *captureWriter) Header() http.Header { return w.headers }
func (w *captureWriter) Write(b []byte) (int, error) {
	if w.body.Len()+len(b) > 4<<20 {
		return 0, fmt.Errorf("batch response too large")
	}
	return w.body.Write(b)
}
func (w *captureWriter) WriteString(s string) (int, error) { return w.Write([]byte(s)) }
func (w *captureWriter) WriteHeader(code int)              { w.status = code }
func (w *captureWriter) WriteHeaderNow()                   {}
func (w *captureWriter) Status() int                       { return w.status }
func (w *captureWriter) Size() int                         { return w.body.Len() }
func (w *captureWriter) Written() bool                     { return w.body.Len() > 0 }
func (w *captureWriter) Flush()                            {}
