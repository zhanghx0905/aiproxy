# Long-document rerank scoring

Enable **Long-document window scoring** in the model editor for a text Rerank
model. Existing models keep their current behavior until enabled. This works
before provider conversion and after response normalization, including TEI and
OpenAI-compatible rerank channels. Multimodal requests cannot use this feature.

For a BCE model served by TEI, the default settings are sufficient:

```json
{
  "plugin": {
    "rerank_window": {
      "enable": true,
      "max_tokens": 512,
      "special_tokens": 4,
      "safety_tokens": 8,
      "overlap_tokens": 80,
      "batch_size": 32,
      "max_document_tokens": 32768,
      "max_windows": 2048,
      "timeout_seconds": 180
    }
  }
}
```

The channel base URL must point to the TEI server root. For other channel types,
also configure `tokenizer_url` to a TEI-compatible `/tokenize` endpoint using the
**scoring model's tokenizer**. A generic GPT tokenizer is not an exact substitute.
The endpoint receives `{"inputs":["text"],"add_special_tokens":false}` and returns
an array per input, containing tokens with UTF-8 byte offsets `start` and `stop`.
Channel credentials are forwarded to the tokenizer only when its origin matches
the scoring channel. Redirects are rejected. The channel proxy and TLS settings
also apply to tokenizer requests.

`max_tokens` is the model's per-pair context limit. `special_tokens` is the number
of special tokens added to a query/document pair (four for BCE's XLM-RoBERTa).
Both must match the actual model. Query tokens, special tokens and the safety
reserve are deducted before splitting document tokens. Overlap is capped at one
quarter of the available window to ensure progress for longer queries. Each
window is tokenized again to check the actual boundary; TEI scoring uses
`truncate=false`. The entire original document, including its tail, is covered.

Scoring batches run sequentially with at most `batch_size` windows. Each original
document receives the **maximum** of its window scores, following BCE's long-text
aggregation method. This detects a relevant passage anywhere in the document;
it does not measure the fraction of the document that is relevant. Longer
documents have more opportunities to obtain a high score, so relevance thresholds
should be checked against the actual corpus.

The public `/v1/rerank` request/response shape is unchanged:

- Result indices refer to original documents, not windows.
- `return_documents=true` returns the complete original document.
- `top_n` is applied after aggregation, with stable input order for score ties.
- Usage sums all scoring batches, including repeated queries and overlaps. TEI
  usage is estimated from the verified query/document windows and configured special
  tokens; providers reporting their own usage retain those values. The gateway
  checks the expanded usage against the group balance and remaining model TPM
  before the first scoring batch. Cache hits are checked against their stored usage.
- The gateway caps scoring at 1,048,576 input tokens per attempt, even when the
  configured window and token limits would allow more. Request patches may set
  provider options. The model, query, documents, window controls, and
  TEI's `truncate=false` are fixed for generated scoring batches.
- `X-Aiproxy-Rerank-Windows` reports the number of scored windows.
- Overlong queries, empty documents, invalid tokenizer offsets or exceeded limits
  fail explicitly. `max_chunks_per_doc`, when supplied, is an additional hard
  limit; documents are not silently shortened to meet it.
- Missing or duplicate window scores and failed batches fail the entire request
  without sending a partial ranking. Completed scoring work from failed attempts
  still counts toward usage and rate metrics. Failed requests are not charged.
  The per-attempt deadline covers tokenization and scoring, and client cancellation
  stops remaining work.

Windowing increases inference work. Keep the native context limit at its actual
value and use `max_document_tokens` / `max_windows` to bound expansion. Enabling
this feature does not extend the model's native attention window.

Run the focused checks:

```sh
cd core
go test ./relay/plugin/rerankwindow
# Optional: exercise a real BCE server; use a reachable URL without an HTTP proxy.
RERANK_WINDOW_TEI_URL=http://tei:80 go test ./relay/plugin/rerankwindow -run TestRealTEILongDocument -v

cd ../web
pnpm exec vitest run src/validation/model.test.ts src/feature/model/components/RerankWindowFields.test.tsx --maxWorkers=1
pnpm run build
```
