#include "fma_addcmul_tiling.h"
#include "register/op_def_registry.h"

#include <cstdint>
#include <limits>

namespace {
constexpr uint32_t kMaxBlockDim = 32;

bool SameShape(const gert::Shape& lhs, const gert::Shape& rhs)
{
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < lhs.GetDimNum(); ++i) {
        if (lhs.GetDim(i) != rhs.GetDim(i)) {
            return false;
        }
    }
    return true;
}

bool NumElements(const gert::Shape& shape, uint32_t* result)
{
    uint64_t elements = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        const int64_t dim = shape.GetDim(i);
        if (dim < 0) {
            return false;
        }
        elements *= static_cast<uint64_t>(dim);
        if (elements > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
    }
    *result = static_cast<uint32_t>(elements);
    return true;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* self_storage = context->GetInputShape(0);
    const gert::StorageShape* tensor1_storage = context->GetInputShape(1);
    const gert::StorageShape* tensor2_storage = context->GetInputShape(2);
    const gert::StorageShape* out_storage = context->GetOutputShape(0);
    if (self_storage == nullptr || tensor1_storage == nullptr ||
        tensor2_storage == nullptr || out_storage == nullptr) {
        return ge::GRAPH_PARAM_INVALID;
    }

    const gert::Shape& self = self_storage->GetStorageShape();
    if (!SameShape(self, tensor1_storage->GetStorageShape()) ||
        !SameShape(self, tensor2_storage->GetStorageShape()) ||
        !SameShape(self, out_storage->GetStorageShape())) {
        return ge::GRAPH_PARAM_INVALID;
    }

    uint32_t num_elements = 0;
    if (!NumElements(self, &num_elements)) {
        return ge::GRAPH_PARAM_INVALID;
    }

    FmaAddcmulTilingData tiling;
    tiling.set_numElements(num_elements);
    // The ACLNN scalar is passed as the fourth kernel argument.  Its value is
    // read once by the kernel; keeping no scalar in tiling avoids a host-side
    // value-dependence and preserves the standard aclScalar ABI.
    tiling.set_value(0.0f);
    context->SetBlockDim(num_elements == 0 ? 1 :
        (num_elements < kMaxBlockDim ? num_elements : kMaxBlockDim));
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* self = context->GetInputShape(0);
    const gert::Shape* tensor1 = context->GetInputShape(1);
    const gert::Shape* tensor2 = context->GetInputShape(2);
    gert::Shape* out = context->GetOutputShape(0);
    if (self == nullptr || tensor1 == nullptr || tensor2 == nullptr ||
        !SameShape(*self, *tensor1) || !SameShape(*self, *tensor2)) {
        return GRAPH_PARAM_INVALID;
    }
    out->SetDimNum(0);
    for (size_t i = 0; i < self->GetDimNum(); ++i) {
        out->AppendDim(self->GetDim(i));
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
class FmaAddcmul : public OpDef {
public:
    explicit FmaAddcmul(const char* name) : OpDef(name)
    {
        this->Input("self").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("tensor1").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("tensor2").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("value").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Scalar();
        this->Output("out").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910");
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(FmaAddcmul);
}  // namespace ops
