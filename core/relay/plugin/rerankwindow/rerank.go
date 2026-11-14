// Package rerankwindow expands text documents before provider adaptation and
// aggregates normalized scores afterwards, so batching is independent of providers.
package rerankwindow

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
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
	"github.com/labring/aiproxy/core/relay/controller"
	"github.com/labring/aiproxy/core/relay/meta"
	"github.com/labring/aiproxy/core/relay/mode"
	relaymodel "github.com/labring/aiproxy/core/relay/model"
	"github.com/labring/aiproxy/core/relay/plugin"
	"github.com/labring/aiproxy/core/relay/plugin/noop"
	"github.com/labring/aiproxy/core/relay/utils"
)

const (
	stateKey         = "rerank_window_state"
	maxScoringTokens = 512 * 2048
)

type RerankWindow struct {
	noop.Noop
	checkBudget func(*gin.Context, *meta.Meta, model.Usage) adaptor.Error
}

var _ plugin.Plugin = (*RerankWindow)(nil)

// NewPlugin adds opt-in long-document scoring without changing the public API.
func NewPlugin(checkBudget func(*gin.Context, *meta.Meta, model.Usage) adaptor.Error) plugin.Plugin {
	return &RerankWindow{checkBudget: checkBudget}
}

type requestState struct {
	BatchSize   int
	Fields      map[string]json.RawMessage
	Request     relaymodel.RerankRequest
	Windows     []window
	InputTokens int64
	Deadline    time.Time
	Response    []byte
	Result      adaptor.DoResponseResult
	Failure     adaptor.Error
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

	if m.Channel.BaseURL == "" {
		if inner, ok := do.(adaptor.Adaptor); ok {
			m.Channel.BaseURL = inner.DefaultBaseURL()
		}
	}

	s := &requestState{BatchSize: cfg.BatchSize, Deadline: time.Now().Add(time.Duration(cfg.TimeoutSeconds) * time.Second), TEI: m.Channel.Type == model.ChannelTypeTextEmbeddingsInference}
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
	}

	ctx, cancel := context.WithDeadline(req.Context(), s.Deadline)
	defer cancel()
	tokenize, err := newTokenizer(m, cfg)
	if err != nil {
		return adaptor.ConvertResult{}, err
	}

	query, err := tokenize(ctx, []string{s.Request.Query})
	if err != nil {
		return adaptor.ConvertResult{}, err
	}
	queryTokens := len(query[0])
	budget := cfg.MaxTokens - cfg.SpecialTokens - queryTokens
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
			s.Windows[start+i].Tokens = len(tokens) + queryTokens + cfg.SpecialTokens
			s.InputTokens += int64(s.Windows[start+i].Tokens)
			if s.InputTokens > maxScoringTokens {
				return adaptor.ConvertResult{}, fmt.Errorf("request exceeds the windowed rerank scoring budget of %d tokens", maxScoringTokens)
			}
		}
	}
	for _, key := range []string{"top_n", "max_chunks_per_doc", "overlap_tokens"} {
		delete(s.Fields, key)
	}
	s.Fields["return_documents"] = json.RawMessage("false")
	if s.TEI {
		s.Fields["truncate"] = json.RawMessage("false")
	}

	m.Set(stateKey, s)

	// Scoring invokes the normal adapter pipeline for each batch. The outer request is never sent upstream.
	return adaptor.ConvertResult{}, nil
}

func (p *RerankWindow) SetupRequestHeader(m *meta.Meta, store adaptor.Store, c *gin.Context,
	req *http.Request, do adaptor.SetupRequestHeader) error {
	if getState(m) != nil {
		return nil
	}

	return do.SetupRequestHeader(m, store, c, req)
}

// newTokenizer uses exact TEI byte offsets for any provider backed by a compatible
// tokenizer endpoint. Channel credentials are sent only to the same origin.
func newTokenizer(m *meta.Meta, cfg Config) (func(context.Context, []string) ([][]token, error), error) {
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
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Hostname() == "" || parsed.User != nil {
		return nil, fmt.Errorf("invalid tokenizer_url")
	}

	client, err := utils.LoadHTTPClientWithTLSConfigE(30*time.Second, m.Channel.ProxyURL, m.Channel.SkipTLSVerify)
	if err != nil {
		return nil, err
	}

	tokenClient := *client
	tokenClient.CheckRedirect = func(_ *http.Request, _ []*http.Request) error { return http.ErrUseLastResponse }

	var authorization string
	base, _ := url.Parse(m.Channel.BaseURL)
	if base != nil && base.Scheme == parsed.Scheme && base.Host == parsed.Host && m.Channel.Key != "" {
		authorization = "Bearer " + m.Channel.Key
	}

	return func(ctx context.Context, texts []string) ([][]token, error) {
		payload, err := json.Marshal(map[string]any{"inputs": texts, "add_special_tokens": false})
		if err != nil {
			return nil, err
		}

		req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "application/json")
		if authorization != "" {
			req.Header.Set("Authorization", authorization)
		}

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
	}, nil
}

// batchRequest preserves provider options but applies top_n only after document
// aggregation. Every window must receive a score before a response is committed.
func (s *requestState) batchRequest(original *http.Request, start, end int) (*http.Request, *batchInput, error) {
	texts := make([]string, end-start)
	for i := range texts {
		texts[i] = s.Windows[start+i].Text
	}
	documents, err := json.Marshal(texts)
	if err != nil {
		return nil, nil, err
	}
	s.Fields["documents"] = documents
	body, err := json.Marshal(s.Fields)
	if err != nil {
		return nil, nil, err
	}
	req := original.Clone(original.Context())
	common.SetRequestBody(req, body)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Del("Content-Length")
	req.Header.Del("Content-Encoding")
	return req, &batchInput{model: s.Request.Model, query: s.Request.Query, documents: texts, disableTruncation: s.TEI}, nil
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

	if err := ctx.Err(); err != nil {
		return nil, err
	}

	usage := model.Usage{
		InputTokens: model.ZeroNullInt64(s.InputTokens),
		TotalTokens: model.ZeroNullInt64(s.InputTokens),
	}
	if p.checkBudget != nil {
		if err := p.checkBudget(c, m, usage); err != nil {
			return nil, err
		}
	}

	status := http.StatusOK
	response, err := s.scoreWindows(ctx, inner, m, store, c)
	if err != nil {
		s.Failure = controller.MapRequestError(m, err, http.StatusBadGateway, "windowed rerank failed")
		s.Result.PartialUsage = s.Result.Usage.TotalTokens > 0
		status = s.Failure.StatusCode()
	}
	s.Response = response

	// Unwind the outer response hooks so partial usage reaches the group monitor.
	return &http.Response{StatusCode: status, Body: http.NoBody}, nil
}

func (s *requestState) scoreWindows(ctx context.Context, inner adaptor.Adaptor, m *meta.Meta,
	store adaptor.Store, c *gin.Context) ([]byte, error) {
	scores := make([]float64, len(s.Request.Documents))
	for i := range scores {
		scores[i] = math.Inf(-1)
	}
	for start := 0; start < len(s.Windows); start += s.BatchSize {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		end := min(start+s.BatchSize, len(s.Windows))
		batchReq, expected, err := s.batchRequest(c.Request.WithContext(ctx), start, end)
		if err != nil {
			return nil, err
		}
		batchMeta := m.Fork()
		batchMeta.RequestUsage = model.Usage{}
		for _, w := range s.Windows[start:end] {
			batchMeta.RequestUsage.InputTokens += model.ZeroNullInt64(w.Tokens)
		}
		batchMeta.RequestUsage.TotalTokens = batchMeta.RequestUsage.InputTokens
		batchMeta.Set(batchKey, expected)
		batchContext := c.Copy()
		batchContext.Request = batchReq
		writer := &captureWriter{ResponseWriter: c.Writer, headers: make(http.Header), status: http.StatusOK}
		batchContext.Writer = writer
		result, _, relayErr := controller.DoHelperWithContext(ctx, inner, batchContext, batchMeta, store)
		if relayErr != nil {
			return nil, relayErr
		}
		s.Result.Usage.Add(result.Usage)

		var response struct {
			Results []struct {
				Index *int     `json:"index"`
				Score *float64 `json:"relevance_score"`
			} `json:"results"`
		}
		if err := json.Unmarshal(writer.body.Bytes(), &response); err != nil {
			return nil, fmt.Errorf("invalid normalized rerank response: %w", err)
		}
		if len(response.Results) != end-start {
			return nil, fmt.Errorf("upstream omitted window scores")
		}

		seen := make([]bool, end-start)
		for _, item := range response.Results {
			if item.Index == nil || item.Score == nil {
				return nil, fmt.Errorf("upstream omitted a window score or index")
			}
			index, score := *item.Index, *item.Score
			if index < 0 || index >= len(seen) || seen[index] || math.IsNaN(score) || math.IsInf(score, 0) {
				return nil, fmt.Errorf("invalid or duplicate window score index")
			}
			seen[index] = true
			id := s.Windows[start+index].Document
			scores[id] = max(scores[id], score)
		}
	}

	if err := ctx.Err(); err != nil {
		return nil, err
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
	return json.Marshal(relaymodel.RerankResponse{
		ID: m.RequestID, Results: results,
		Meta: relaymodel.RerankMeta{Model: m.OriginModel, Tokens: &relaymodel.RerankMetaTokens{
			InputTokens: int64(s.Result.Usage.InputTokens), OutputTokens: int64(s.Result.Usage.OutputTokens),
		}},
	})
}

// DoResponse commits only the fully aggregated response, leaving outer cache and
// tenant monitoring hooks to account for the combined request once.
func (p *RerankWindow) DoResponse(m *meta.Meta, store adaptor.Store, c *gin.Context,
	resp *http.Response, do adaptor.DoResponse) (adaptor.DoResponseResult, adaptor.Error) {
	s := getState(m)
	if s == nil {
		return do.DoResponse(m, store, c, resp)
	}
	if s.Failure != nil {
		return s.Result, s.Failure
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
