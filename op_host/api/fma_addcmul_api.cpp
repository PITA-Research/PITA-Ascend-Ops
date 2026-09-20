#include <aclnn/acl_meta.h>
#include <cstdint>
#include <limits>
#include <vector>

#include "graph/types.h"

namespace {
constexpr aclnnStatus kParamNull = 161001;
constexpr aclnnStatus kParamInvalid = 161002;

struct TensorDesc {
  ge::DataType dtype;
  ge::Format format;
};
struct SupportInfo {
  TensorDesc* inputs_desc;
  size_t inputs_num;
  TensorDesc* outputs_desc;
  size_t outputs_num;
};
struct OpSocSupportInfo {
  SupportInfo* support_info;
  size_t num;
};
struct OpSupportList {
  OpSocSupportInfo* soc_support_info;
  size_t num;
};
struct NnopbaseDfxId {
  uint32_t id;
  const char* func_name;
  bool has_reg;
};

enum SocType { kAscend910A = 1, kAscend910B };

TensorDesc kInputs[4] = {{ge::DT_FLOAT, ge::FORMAT_ND},
                          {ge::DT_FLOAT, ge::FORMAT_ND},
                          {ge::DT_FLOAT, ge::FORMAT_ND},
                          {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc kOutputs[1] = {{ge::DT_FLOAT, ge::FORMAT_ND}};
SupportInfo kSupportInfo[1] = {{kInputs, 4, kOutputs, 1}};
OpSocSupportInfo kSocSupport[2] = {{kSupportInfo, 1}, {kSupportInfo, 1}};
OpSupportList kSupportList = {kSocSupport, 2};
uint32_t kSocList[2] = {kAscend910A, kAscend910B};

struct TensorRange {
  std::vector<int64_t> shape;
  uintptr_t begin = 0;
  uintptr_t end = 0;
};

bool GetTensorRange(const aclTensor* tensor, TensorRange* result) {
  aclDataType dtype;
  aclFormat format;
  int64_t *view = nullptr, *storage = nullptr, *strides = nullptr;
  uint64_t rank = 0, storage_rank = 0, stride_rank = 0;
  int64_t offset = 0;
  void* address = nullptr;
  if (aclGetDataType(tensor, &dtype) != 0 || dtype != ACL_FLOAT ||
      aclGetFormat(tensor, &format) != 0 || format != ACL_FORMAT_ND ||
      aclGetViewShape(tensor, &view, &rank) != 0 ||
      aclGetStorageShape(tensor, &storage, &storage_rank) != 0 ||
      aclGetViewStrides(tensor, &strides, &stride_rank) != 0 ||
      aclGetViewOffset(tensor, &offset) != 0 || offset != 0 ||
      aclGetRawTensorAddr(tensor, &address) != 0 || rank != storage_rank ||
      rank != stride_rank || rank > 25) {
    return false;
  }
  uint64_t elements = 1;
  result->shape.assign(view, view + rank);
  for (uint64_t i = 0; i < rank; ++i) {
    if (view[i] < 0 || storage[i] != view[i] || strides[i] < 0) {
      return false;
    }
    if (view[i] != 0 && elements >
        std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(view[i])) {
      return false;
    }
    elements *= static_cast<uint64_t>(view[i]);
  }
  uint64_t expected_stride = 1;
  for (uint64_t i = rank; i > 0; --i) {
    if (static_cast<uint64_t>(strides[i - 1]) != expected_stride) {
      return false;
    }
    if (view[i - 1] != 0 && expected_stride >
        std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(view[i - 1])) {
      return false;
    }
    expected_stride *= static_cast<uint64_t>(view[i - 1]);
  }
  if (elements == 0) {
    result->begin = result->end = 0;
    return true;
  }
  if (address == nullptr || elements >
      std::numeric_limits<uintptr_t>::max() / sizeof(float)) {
    return false;
  }
  result->begin = reinterpret_cast<uintptr_t>(address);
  const uintptr_t bytes = static_cast<uintptr_t>(elements * sizeof(float));
  if (result->begin > std::numeric_limits<uintptr_t>::max() - bytes) {
    return false;
  }
  result->end = result->begin + bytes;
  return true;
}

bool Overlaps(const TensorRange& a, const TensorRange& b) {
  return a.begin != a.end && b.begin != b.end && a.begin < b.end && b.begin < a.end;
}

bool ValidateArguments(const aclTensor* self, const aclTensor* tensor1,
                       const aclTensor* tensor2, const aclTensor* out) {
  TensorRange self_range, tensor1_range, tensor2_range, out_range;
  if (!GetTensorRange(self, &self_range) || !GetTensorRange(tensor1, &tensor1_range) ||
      !GetTensorRange(tensor2, &tensor2_range) || !GetTensorRange(out, &out_range) ||
      self_range.shape != tensor1_range.shape || self_range.shape != tensor2_range.shape ||
      self_range.shape != out_range.shape) {
    return false;
  }
  return !Overlaps(out_range, self_range) && !Overlaps(out_range, tensor1_range) &&
         !Overlaps(out_range, tensor2_range);
}
}  // namespace

extern "C" {
extern aclnnStatus NnopbaseCreateExecutorSpace(void** space);
extern void* NnopbaseGetExecutor(void* space, const char* op_type, char* inputs_desc,
                                 uint32_t input_num, char* outputs_desc, uint32_t output_num,
                                 char* attrs_desc, uint32_t attrs_num);
extern aclnnStatus NnopbaseAddInput(void* executor, const aclTensor* tensor, uint32_t index);
extern aclnnStatus NnopbaseAddOutput(void* executor, const aclTensor* tensor, uint32_t index);
extern aclnnStatus NnopbaseAddScalarInput(void* executor, const aclScalar* scalar, uint32_t index,
                                          int32_t src_index, ge::DataType dtype);
extern aclnnStatus NnopbaseAddTilingId(void* executor, NnopbaseDfxId* id);
extern aclnnStatus NnopbaseAddSupportList(void* executor, OpSupportList* list,
                                          uint32_t* soc_list, size_t soc_list_len);
extern aclnnStatus NnopbaseRunForWorkspace(void* executor, uint64_t* workspace_size);
extern aclnnStatus NnopbaseRunWithWorkspace(void* executor, aclrtStream stream,
                                            void* workspace, uint64_t workspace_size);
extern uint64_t NnopbaseMsprofSysTime();
extern void NnopbaseReportApiInfo(uint64_t begin_time, NnopbaseDfxId& id);
extern void NnopbaseSetMatchArgsFlag(void* executor) __attribute__((weak));
extern bool NnopbaseMatchArgs(void* executor, uint64_t* workspace_size) __attribute__((weak));
extern aclnnStatus NnopbaseAddParamName(void* executor, uint32_t index, const char* name,
                                        bool is_input) __attribute__((weak));
}

#define ACLNN_API __attribute__((visibility("default")))

extern "C" ACLNN_API aclnnStatus aclnnFmaAddcmulGetWorkspaceSize(
    const aclTensor* self, const aclTensor* tensor1, const aclTensor* tensor2,
    const aclScalar* value, const aclTensor* out, uint64_t* workspace_size,
    aclOpExecutor** executor) {
  if (self == nullptr || tensor1 == nullptr || tensor2 == nullptr || value == nullptr ||
      out == nullptr || workspace_size == nullptr || executor == nullptr) {
    return kParamNull;
  }
  if (!ValidateArguments(self, tensor1, tensor2, out)) {
    return kParamInvalid;
  }
  const uint64_t begin_time = NnopbaseMsprofSysTime();
  static NnopbaseDfxId dfx_id = {0x60000, __func__, false};
  static NnopbaseDfxId tiling_id = {0x60000, "aclnnFmaAddcmulTiling", false};
  static void* executor_space = nullptr;
  if (executor_space == nullptr) {
    const aclnnStatus status = NnopbaseCreateExecutorSpace(&executor_space);
    if (status != 0) return status;
  }
  char inputs_desc[] = {1, 1, 1, 1};
  char outputs_desc[] = {1};
  char attrs_desc[] = {};
  void* raw_executor = NnopbaseGetExecutor(executor_space, "FmaAddcmul", inputs_desc, 4,
                                           outputs_desc, 1, attrs_desc, 0);
  if (raw_executor == nullptr) return kParamInvalid;
  *executor = reinterpret_cast<aclOpExecutor*>(raw_executor);
  aclnnStatus status = NnopbaseAddTilingId(*executor, &tiling_id);
  if (status != 0) return status;
  if (NnopbaseSetMatchArgsFlag != nullptr) NnopbaseSetMatchArgsFlag(*executor);
  if ((status = NnopbaseAddInput(*executor, self, 0)) != 0 ||
      (status = NnopbaseAddInput(*executor, tensor1, 1)) != 0 ||
      (status = NnopbaseAddInput(*executor, tensor2, 2)) != 0 ||
      (status = NnopbaseAddScalarInput(*executor, value, 3, -1, ge::DT_UNDEFINED)) != 0 ||
      (status = NnopbaseAddOutput(*executor, out, 0)) != 0) {
    return status;
  }
  if (NnopbaseMatchArgs != nullptr && NnopbaseMatchArgs(*executor, workspace_size)) {
    NnopbaseReportApiInfo(begin_time, dfx_id);
    return 0;
  }
  if (NnopbaseAddParamName != nullptr) {
    if ((status = NnopbaseAddParamName(*executor, 0, "self", true)) != 0 ||
        (status = NnopbaseAddParamName(*executor, 1, "tensor1", true)) != 0 ||
        (status = NnopbaseAddParamName(*executor, 2, "tensor2", true)) != 0 ||
        (status = NnopbaseAddParamName(*executor, 3, "value", true)) != 0 ||
        (status = NnopbaseAddParamName(*executor, 0, "out", false)) != 0) return status;
  }
  if ((status = NnopbaseAddSupportList(*executor, &kSupportList, kSocList, 2)) != 0 ||
      (status = NnopbaseRunForWorkspace(*executor, workspace_size)) != 0) return status;
  NnopbaseReportApiInfo(begin_time, dfx_id);
  return 0;
}

extern "C" ACLNN_API aclnnStatus aclnnFmaAddcmul(void* workspace, uint64_t workspace_size,
                                                  aclOpExecutor* executor, aclrtStream stream) {
  if (executor == nullptr) return kParamNull;
  const uint64_t begin_time = NnopbaseMsprofSysTime();
  static NnopbaseDfxId dfx_id = {0x60000, __func__, false};
  const aclnnStatus status = NnopbaseRunWithWorkspace(executor, stream, workspace, workspace_size);
  NnopbaseReportApiInfo(begin_time, dfx_id);
  return status;
}
