package rerankwindow

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/gin-gonic/gin"
	"github.com/labring/aiproxy/core/model"
	"github.com/labring/aiproxy/core/relay/adaptor"
	"github.com/labring/aiproxy/core/relay/adaptor/openai"
	tei "github.com/labring/aiproxy/core/relay/adaptor/text-embeddings-inference"
	"github.com/labring/aiproxy/core/relay/controller"
	"github.com/labring/aiproxy/core/relay/meta"
	"github.com/labring/aiproxy/core/relay/mode"
	relaymodel "github.com/labring/aiproxy/core/relay/model"
	"github.com/labring/aiproxy/core/relay/plugin"
	"github.com/labring/aiproxy/core/relay/plugin/cache"
	"github.com/stretchr/testify/require"
)

// runeTokens models UTF-8 byte offsets, including multibyte text and emoji.
func runeTokens(text string) []token {
	var tokens []token
	for offset, r := range text {
		tokens = append(tokens, token{Start: offset, Stop: offset + utf8.RuneLen(r)})
	}
	return tokens
}

type mockBackend struct {
	Calls   int
	Failure string
	TEI     bool
}

func (b *mockBackend) handler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if r.URL.Path == "/tokenize" {
		var body struct {
			Inputs []string `json:"inputs"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		tokens := make([][]token, len(body.Inputs))
		for i, text := range body.Inputs {
			tokens[i] = runeTokens(text)
		}
		_ = json.NewEncoder(w).Encode(tokens)
		return
	}
	b.Calls++
	if b.Failure == "http" && b.Calls == 2 {
		w.WriteHeader(503)
		_, _ = w.Write([]byte(`{"error":"unavailable","error_type":"Backend"}`))
		return
	}
	var body struct {
		Documents []string `json:"documents"`
		Texts     []string `json:"texts"`
		TopN      *int     `json:"top_n"`
	}
	_ = json.NewDecoder(r.Body).Decode(&body)
	if body.TopN != nil {
		panic("top_n reached upstream before aggregation")
	}
	texts := body.Documents
	if b.TEI {
		texts = body.Texts
	}
	items := make([]*relaymodel.RerankResult, len(texts))
	for i, text := range texts {
		score := 0.1
		if strings.Contains(text, "MATCH") {
			score = 0.9
		}
		items[i] = &relaymodel.RerankResult{Index: i, RelevanceScore: score}
	}
	if b.Failure == "missing" && b.Calls == 2 {
		items = items[:0]
	}
	if b.Failure == "duplicate" && b.Calls == 2 && len(items) > 1 {
		items[1].Index = items[0].Index
	}
	if b.TEI {
		result := make([]map[string]any, len(items))
		for i, item := range items {
			result[i] = map[string]any{"index": item.Index, "score": item.RelevanceScore}
		}
		_ = json.NewEncoder(w).Encode(result)
	} else {
		_ = json.NewEncoder(w).Encode(relaymodel.RerankResponse{Results: items, Meta: relaymodel.RerankMeta{Tokens: &relaymodel.RerankMetaTokens{InputTokens: 11}}})
	}
}

type runOptions struct {
	URL     string
	Body    any
	Config  map[string]any
	TEI     bool
	Context context.Context
	Cache   bool
}

// runRelay exercises the real conversion, HTTP, response and usage pipeline.
func runRelay(t *testing.T, options runOptions) (*httptest.ResponseRecorder, adaptor.DoResponseResult, adaptor.Error) {
	t.Helper()
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(recorder)
	body, err := json.Marshal(options.Body)
	require.NoError(t, err)
	ctx := options.Context
	if ctx == nil {
		ctx = context.Background()
	}
	c.Request = httptest.NewRequestWithContext(ctx, http.MethodPost, "/v1/rerank", strings.NewReader(string(body)))
	c.Request.Header.Set("Content-Type", "application/json")
	channel := &model.Channel{BaseURL: options.URL, Type: model.ChannelTypeOpenAI}
	var inner adaptor.Adaptor = &openai.Adaptor{}
	if options.TEI {
		channel.Type = model.ChannelTypeTextEmbeddingsInference
		inner = &tei.Adaptor{}
	}
	cfg := model.ModelConfig{Model: "bce", Type: mode.Rerank, Plugin: map[string]map[string]any{"rerank_window": options.Config}}
	m := meta.NewMeta(channel, mode.Rerank, "bce", cfg, meta.WithRequestID("window-test"))
	plugins := []plugin.Plugin{NewPlugin()}
	if options.Cache {
		cfg.Plugin["cache"] = map[string]any{"enable": true, "ttl": 60, "item_max_size": 100000, "add_cache_hit_header": true}
		plugins = append([]plugin.Plugin{cache.NewCachePlugin(nil)}, plugins...)
	}
	wrapped := plugin.WrapperAdaptor(inner, plugins...)
	result, _, relayErr := controller.DoHelper(wrapped, c, m, nil)
	return recorder, result, relayErr
}

func TestWindowedRerankAcrossProviders(t *testing.T) {
	for _, isTEI := range []bool{true, false} {
		t.Run(fmt.Sprint(isTEI), func(t *testing.T) {
			backend := &mockBackend{TEI: isTEI}
			server := httptest.NewServer(http.HandlerFunc(backend.handler))
			defer server.Close()
			long := strings.Repeat("中文😀无关段落", 24) + "MATCH"
			cfg := map[string]any{"enable": true, "max_tokens": 64, "special_tokens": 4, "safety_tokens": 2, "overlap_tokens": 8, "batch_size": 2}
			if !isTEI {
				cfg["tokenizer_url"] = server.URL + "/tokenize"
			}
			recorder, result, err := runRelay(t, runOptions{URL: server.URL, TEI: isTEI, Config: cfg, Body: map[string]any{"model": "bce", "query": "q", "documents": []string{"unrelated", long}, "top_n": 1, "return_documents": true}})
			require.Nil(t, err)
			require.Greater(t, backend.Calls, 1)
			var response relaymodel.RerankResponse
			require.NoError(t, json.Unmarshal(recorder.Body.Bytes(), &response))
			require.Len(t, response.Results, 1)
			require.Equal(t, 1, response.Results[0].Index)
			require.Equal(t, long, response.Results[0].Document.Text)
			require.InDelta(t, 0.9, response.Results[0].RelevanceScore, 0.00001)
			require.Greater(t, int64(result.Usage.InputTokens), int64(0))
			require.Equal(t, int64(result.Usage.InputTokens), response.Meta.Tokens.InputTokens)
			require.Equal(t, "window-test", response.ID)
		})
	}
}

func TestWindowedRerankNoPartialResponses(t *testing.T) {
	for _, failure := range []string{"http", "missing", "duplicate"} {
		t.Run(failure, func(t *testing.T) {
			backend := &mockBackend{TEI: true, Failure: failure}
			server := httptest.NewServer(http.HandlerFunc(backend.handler))
			defer server.Close()
			recorder, _, err := runRelay(t, runOptions{URL: server.URL, TEI: true, Config: map[string]any{"enable": true, "max_tokens": 32, "overlap_tokens": 2, "safety_tokens": 2, "batch_size": 2}, Body: map[string]any{"model": "bce", "query": "q", "documents": []string{strings.Repeat("a", 200)}}})
			require.NotNil(t, err)
			require.Empty(t, recorder.Body.String())
		})
	}
}

func TestWindowCacheStoresFinalRankingAndInvalidatesSettings(t *testing.T) {
	backend := &mockBackend{TEI: true}
	server := httptest.NewServer(http.HandlerFunc(backend.handler))
	defer server.Close()
	opts := runOptions{URL: server.URL, TEI: true, Cache: true, Config: map[string]any{"enable": true, "max_tokens": 64, "overlap_tokens": 4, "batch_size": 2}, Body: map[string]any{"model": "bce", "query": "cache-scope", "documents": []string{strings.Repeat("b", 120) + "MATCH"}, "return_documents": true}}
	first, _, err := runRelay(t, opts)
	require.Nil(t, err)
	calls := backend.Calls
	second, _, err := runRelay(t, opts)
	require.Nil(t, err)
	require.Equal(t, "hit", second.Header().Get("X-Aiproxy-Cache"))
	require.Equal(t, calls, backend.Calls)
	require.Equal(t, first.Body.String(), second.Body.String())
	opts.Config["max_tokens"] = 48
	_, _, err = runRelay(t, opts)
	require.Nil(t, err)
	require.Greater(t, backend.Calls, calls)
}

func TestDisabledWindowingPreservesNativeRequest(t *testing.T) {
	backend := &mockBackend{TEI: true}
	server := httptest.NewServer(http.HandlerFunc(backend.handler))
	defer server.Close()
	recorder, _, err := runRelay(t, runOptions{URL: server.URL, TEI: true, Config: map[string]any{"enable": false}, Body: map[string]any{"model": "bce", "query": "q", "documents": []string{"short MATCH"}}})
	require.Nil(t, err)
	require.Equal(t, 1, backend.Calls)
	require.Empty(t, recorder.Header().Get("X-Aiproxy-Rerank-Windows"))
}

func TestWindowLimitsAndCancellation(t *testing.T) {
	for _, test := range []struct {
		Name   string
		Query  string
		Config map[string]any
		Cancel bool
	}{
		{Name: "query", Query: strings.Repeat("q", 100)},
		{Name: "document", Query: "q", Config: map[string]any{"max_document_tokens": 20}},
		{Name: "windows", Query: "q", Config: map[string]any{"max_windows": 2, "batch_size": 2}},
		{Name: "cancel", Query: "q", Cancel: true},
	} {
		t.Run(test.Name, func(t *testing.T) {
			backend := &mockBackend{TEI: true}
			server := httptest.NewServer(http.HandlerFunc(backend.handler))
			defer server.Close()
			cfg := map[string]any{"enable": true, "max_tokens": 32, "overlap_tokens": 2, "safety_tokens": 2}
			for key, value := range test.Config {
				cfg[key] = value
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if test.Cancel {
				cancel()
			}
			recorder, _, err := runRelay(t, runOptions{URL: server.URL, TEI: true, Context: ctx, Config: cfg, Body: map[string]any{"model": "bce", "query": test.Query, "documents": []string{strings.Repeat("a", 200)}}})
			require.NotNil(t, err)
			require.Zero(t, backend.Calls)
			require.Empty(t, recorder.Body.String())
		})
	}
}

func TestWindowOffsetsAndTail(t *testing.T) {
	text := strings.Repeat("中文😀", 100) + "尾部"
	tokens := runeTokens(text)
	// SentencePiece can emit a prefix and a word sharing the same byte range.
	tokens = append([]token{tokens[0]}, tokens...)
	windows, err := splitWindows(text, splitOptions{Tokens: tokens, Config: defaultConfig(), Budget: 32})
	require.NoError(t, err)
	require.Greater(t, len(windows), 1)
	require.True(t, strings.HasSuffix(windows[len(windows)-1].Text, "尾部"))
	for _, w := range windows {
		require.True(t, utf8.ValidString(w.Text))
	}
	_, err = splitWindows(text, splitOptions{Tokens: []token{{Start: 1, Stop: 3}}, Config: defaultConfig(), Budget: 32})
	require.Error(t, err)
}

// TestRealTEILongDocument is opt-in and scores a tail-only answer on an existing BCE.
func TestRealTEILongDocument(t *testing.T) {
	endpoint := os.Getenv("RERANK_WINDOW_TEI_URL")
	if endpoint == "" {
		t.Skip("set RERANK_WINDOW_TEI_URL to run the live BCE check")
	}
	long := strings.Repeat("这段介绍普通办公用品的采购与库存管理，不涉及任何项目暗号。", 100) + "项目代号苍穹的紧急联络暗号是蓝鲸七号。"
	recorder, _, err := runRelay(t, runOptions{URL: endpoint, TEI: true, Config: map[string]any{"enable": true, "batch_size": 8}, Body: map[string]any{"model": "bce", "query": "项目代号苍穹的紧急联络暗号是什么？", "documents": []string{"项目代号苍穹的办公用品由采购部门负责。", long}, "return_documents": true}})
	require.Nil(t, err)
	var response relaymodel.RerankResponse
	require.NoError(t, json.Unmarshal(recorder.Body.Bytes(), &response))
	require.Equal(t, 1, response.Results[0].Index)
	require.Equal(t, long, response.Results[0].Document.Text)
	t.Logf("windows=%s scores=%+v", recorder.Header().Get("X-Aiproxy-Rerank-Windows"), response.Results)
}
