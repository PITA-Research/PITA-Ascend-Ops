#include "dot_fp32_left_order_no_neumaier_tiling.h"
#include "register/op_def_registry.h"

namespace {
constexpr uint32_t kMaxBlockDim = 32;

uint32_t ClampBlockDim(uint32_t batch)
{
    if (batch == 0) {
        return 1;
    }
    return batch < kMaxBlockDim ? batch : kMaxBlockDim;
}

uint32_t LeadingSize(const gert::Shape& shape)
{
    const size_t rank = shape.GetDimNum();
    uint32_t size = 1;
    for (size_t i = 0; i + 1 < rank; ++i) {
        size *= static_cast<uint32_t>(shape.GetDim(i));
    }
    return size;
}

bool LeadingDimsEqual(const gert::Shape& lhs_shape, const gert::Shape& rhs_shape)
{
    const size_t lhs_rank = lhs_shape.GetDimNum();
    const size_t rhs_rank = rhs_shape.GetDimNum();
    if (lhs_rank != rhs_rank) {
        return false;
    }
    for (size_t i = 0; i + 1 < lhs_rank; ++i) {
        if (lhs_shape.GetDim(i) != rhs_shape.GetDim(i)) {
            return false;
        }
    }
    return true;
}

const gert::Shape* SelectOutputBatchShape(const gert::Shape* lhs_shape, const gert::Shape* rhs_shape)
{
    const size_t lhs_rank = lhs_shape->GetDimNum();
    const size_t rhs_rank = rhs_shape->GetDimNum();
    if (lhs_rank == 1 && rhs_rank > 1) {
        return rhs_shape;
    }
    return lhs_shape;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    DotFp32LeftOrderNoNeumaierTilingData tiling;
    const gert::StorageShape* lhs_shape = context->GetInputShape(0);
    const gert::StorageShape* rhs_shape = context->GetInputShape(1);
    const gert::Shape& lhs = lhs_shape->GetStorageShape();
    const gert::Shape& rhs = rhs_shape->GetStorageShape();
    const size_t lhs_rank = lhs.GetDimNum();
    const size_t rhs_rank = rhs.GetDimNum();
    if (lhs_rank == 0 || rhs_rank == 0) {
        return ge::GRAPH_PARAM_INVALID;
    }

    const uint32_t dot_len = static_cast<uint32_t>(lhs.GetDim(lhs_rank - 1));
    if (dot_len != static_cast<uint32_t>(rhs.GetDim(rhs_rank - 1))) {
        return ge::GRAPH_PARAM_INVALID;
    }

    const bool lhs_vector = lhs_rank == 1;
    const bool rhs_vector = rhs_rank == 1;
    const uint32_t lhs_batch = LeadingSize(lhs);
    const uint32_t rhs_batch = LeadingSize(rhs);
    uint32_t batch = 1;
    uint32_t lhs_row_stride = 0;
    uint32_t rhs_row_stride = 0;
    if (lhs_vector && rhs_vector) {
        batch = 1;
    } else if (lhs_vector) {
        batch = rhs_batch;
        rhs_row_stride = dot_len;
    } else if (rhs_vector) {
        batch = lhs_batch;
        lhs_row_stride = dot_len;
    } else {
        if (!LeadingDimsEqual(lhs, rhs)) {
            return ge::GRAPH_PARAM_INVALID;
        }
        batch = lhs_batch;
        lhs_row_stride = dot_len;
        rhs_row_stride = dot_len;
    }

    tiling.set_batch(batch);
    tiling.set_dotLen(dot_len);
    tiling.set_lhsRowStride(lhs_row_stride);
    tiling.set_rhsRowStride(rhs_row_stride);
    context->SetBlockDim(ClampBlockDim(batch));
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* lhs_shape = context->GetInputShape(0);
    const gert::Shape* rhs_shape = context->GetInputShape(1);
    gert::Shape* out_shape = context->GetOutputShape(0);
    out_shape->SetDimNum(0);
    const size_t lhs_rank = lhs_shape->GetDimNum();
    const size_t rhs_rank = rhs_shape->GetDimNum();
    if (lhs_rank == 0 || rhs_rank == 0) {
        return GRAPH_PARAM_INVALID;
    }
    if (lhs_shape->GetDim(lhs_rank - 1) != rhs_shape->GetDim(rhs_rank - 1)) {
        return GRAPH_PARAM_INVALID;
    }
    if (lhs_rank > 1 && rhs_rank > 1 && !LeadingDimsEqual(*lhs_shape, *rhs_shape)) {
        return GRAPH_PARAM_INVALID;
    }

    const gert::Shape* batch_shape = SelectOutputBatchShape(lhs_shape, rhs_shape);
    const size_t batch_rank = batch_shape->GetDimNum();
    for (size_t i = 0; i + 1 < batch_rank; ++i) {
        out_shape->AppendDim(batch_shape->GetDim(i));
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class DotFp32LeftOrderNoNeumaier : public OpDef {
public:
    explicit DotFp32LeftOrderNoNeumaier(const char* name) : OpDef(name)
    {
        this->Input("lhs")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("rhs")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910");
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(DotFp32LeftOrderNoNeumaier);
}  // namespace ops
