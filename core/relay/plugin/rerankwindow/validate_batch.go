package rerankwindow

import (
	"encoding/json"
	"net/http"
	"slices"

	"github.com/labring/aiproxy/core/common"
	"github.com/labring/aiproxy/core/relay/adaptor"
	"github.com/labring/aiproxy/core/relay/meta"
	relaymodel "github.com/labring/aiproxy/core/relay/model"
	"github.com/labring/aiproxy/core/relay/plugin"
	"github.com/labring/aiproxy/core/relay/plugin/noop"
)

const batchKey = "rerank_window_batch"

type batchInput struct {
	model             string
	query             string
	documents         []string
	disableTruncation bool
}

type batchValidator struct{ noop.Noop }

// NewBatchValidator checks the scoring input after request patches and before provider conversion.
func NewBatchValidator() plugin.Plugin { return &batchValidator{} }

func (v *batchValidator) ConvertRequest(m *meta.Meta, store adaptor.Store, req *http.Request,
	do adaptor.ConvertRequest) (adaptor.ConvertResult, error) {
	value, _ := m.Get(batchKey)
	expected, _ := value.(*batchInput)
	if expected == nil {
		return do.ConvertRequest(m, store, req)
	}

	var input struct {
		relaymodel.RerankRequest
		Truncate *bool `json:"truncate"`
	}
	body, err := common.GetRequestBodyReusable(req)
	if err != nil {
		return adaptor.ConvertResult{}, err
	}

	if err := json.Unmarshal(body, &input); err != nil ||
		input.Model != expected.model || input.Query != expected.query || !slices.Equal(input.Documents, expected.documents) ||
		input.TopN != nil || input.MaxChunksPerDoc != nil || input.OverlapTokens != nil ||
		input.ReturnDocuments == nil || *input.ReturnDocuments ||
		expected.disableTruncation && (input.Truncate == nil || *input.Truncate) {
		return adaptor.ConvertResult{}, relaymodel.WrapperErrorWithMessage(
			m.Mode, http.StatusBadRequest, "request patch changed fields managed by windowed rerank",
		)
	}

	return do.ConvertRequest(m, store, req)
}
