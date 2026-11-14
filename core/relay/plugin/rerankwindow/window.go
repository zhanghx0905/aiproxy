package rerankwindow

import (
	"fmt"
	"strings"
	"unicode/utf8"
)

type token struct {
	Start   int  `json:"start"`
	Stop    int  `json:"stop"`
	Special bool `json:"special"`
}

type window struct {
	Text     string
	Document int
	Tokens   int
}

type splitOptions struct {
	Tokens   []token
	Config   Config
	Document int
	Budget   int
}

// splitWindows uses UTF-8 byte offsets from the model tokenizer, retaining the
// document tail and overlap. Duplicate SentencePiece offsets are valid.
func splitWindows(text string, options splitOptions) ([]window, error) {
	tokens, cfg, document, budget := options.Tokens, options.Config, options.Document, options.Budget
	if strings.TrimSpace(text) == "" || len(tokens) == 0 {
		return nil, fmt.Errorf("document %d is empty", document)
	}
	if len(tokens) > cfg.MaxDocumentTokens {
		return nil, fmt.Errorf("document %d exceeds max_document_tokens=%d", document, cfg.MaxDocumentTokens)
	}
	previousStart, previousStop := 0, 0
	for _, t := range tokens {
		if t.Special || t.Start < previousStart || t.Stop < previousStop ||
			t.Start < 0 || t.Stop <= t.Start || t.Stop > len(text) ||
			(t.Start < len(text) && !utf8.RuneStart(text[t.Start])) ||
			(t.Stop < len(text) && !utf8.RuneStart(text[t.Stop])) {
			return nil, fmt.Errorf("invalid tokenizer byte offsets for document %d", document)
		}
		previousStart, previousStop = t.Start, t.Stop
	}

	width := budget - cfg.SafetyTokens
	if width < 8 {
		return nil, fmt.Errorf("query leaves insufficient document token budget")
	}
	overlap := min(cfg.OverlapTokens, width/4)
	var result []window
	for start := 0; start < len(tokens); {
		end := min(start+width, len(tokens))
		from, to := tokens[start].Start, tokens[end-1].Stop
		if start == 0 {
			from = 0
		}
		if end == len(tokens) {
			to = len(text)
		}
		if from >= to {
			return nil, fmt.Errorf("empty tokenizer window for document %d", document)
		}
		result = append(result, window{Text: text[from:to], Document: document})
		if len(result) > cfg.MaxWindows {
			return nil, fmt.Errorf("document %d exceeds max_windows=%d", document, cfg.MaxWindows)
		}
		if end == len(tokens) {
			break
		}
		start = end - overlap
	}
	return result, nil
}
