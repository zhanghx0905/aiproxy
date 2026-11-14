package rerankwindow

import "fmt"

// Config opts a text reranker into exact-token windowing. TokenizerURL must expose
// TEI's /tokenize protocol using the same tokenizer as the scoring model.
type Config struct {
	Enable            bool   `json:"enable"`
	TokenizerURL      string `json:"tokenizer_url"`
	MaxTokens         int    `json:"max_tokens"`
	SpecialTokens     int    `json:"special_tokens"`
	SafetyTokens      int    `json:"safety_tokens"`
	OverlapTokens     int    `json:"overlap_tokens"`
	BatchSize         int    `json:"batch_size"`
	MaxWindows        int    `json:"max_windows"`
	MaxDocumentTokens int    `json:"max_document_tokens"`
	TimeoutSeconds    int    `json:"timeout_seconds"`
}

func defaultConfig() Config {
	return Config{MaxTokens: 512, SpecialTokens: 4, SafetyTokens: 8,
		OverlapTokens: 80, BatchSize: 32, MaxWindows: 2048,
		MaxDocumentTokens: 32768, TimeoutSeconds: 180}
}

// validate rejects unsafe limits instead of silently clipping documents or queries.
func (c Config) validate() error {
	if c.MaxTokens < 16 || c.MaxTokens > 131072 || c.SpecialTokens < 0 || c.SpecialTokens >= c.MaxTokens ||
		c.SafetyTokens < 0 || c.SafetyTokens >= c.MaxTokens || c.SpecialTokens+c.SafetyTokens >= c.MaxTokens ||
		c.OverlapTokens < 0 || c.OverlapTokens >= c.MaxTokens ||
		c.BatchSize < 1 || c.BatchSize > 128 || c.MaxWindows < c.BatchSize ||
		c.MaxWindows > 8192 || c.MaxDocumentTokens < 1 ||
		c.MaxDocumentTokens > 131072 || c.TimeoutSeconds < 1 || c.TimeoutSeconds > 600 {
		return fmt.Errorf("invalid rerank_window limits")
	}
	return nil
}
