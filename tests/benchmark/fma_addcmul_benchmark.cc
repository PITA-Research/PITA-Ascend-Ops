#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_addcmul.h>

#include "aclnn_fma_addcmul.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

#define CHECK(expr) do { const aclError status = (expr); if (status != ACL_SUCCESS) { \
  std::printf("FAIL %s: %d\n", #expr, static_cast<int>(status)); std::exit(1); } } while (0)

namespace {
std::vector<int64_t> Strides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

aclTensor* Tensor(const std::vector<int64_t>& shape, void* data) {
  const std::vector<int64_t> strides = Strides(shape);
  return aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0,
                         ACL_FORMAT_ND, shape.data(), shape.size(), data);
}

struct Workspace {
  ~Workspace() { if (data) aclrtFree(data); }
  void* Get(uint64_t size) {
    if (size > capacity) {
      if (data) CHECK(aclrtFree(data));
      CHECK(aclrtMalloc(&data, size, ACL_MEM_MALLOC_NORMAL_ONLY));
      capacity = size;
    }
    return data;
  }
  void* data = nullptr;
  uint64_t capacity = 0;
};

struct Result { float device_us; double host_us; };

Result Measure(const std::function<void()>& fn, int warmup, int iters, aclrtStream stream) {
  for (int i = 0; i < warmup; ++i) fn();
  CHECK(aclrtSynchronizeStream(stream));
  aclrtEvent start = nullptr, end = nullptr;
  CHECK(aclrtCreateEvent(&start));
  CHECK(aclrtCreateEvent(&end));
  const auto host_start = std::chrono::steady_clock::now();
  CHECK(aclrtRecordEvent(start, stream));
  for (int i = 0; i < iters; ++i) fn();
  CHECK(aclrtRecordEvent(end, stream));
  CHECK(aclrtSynchronizeEvent(end));
  const auto host_end = std::chrono::steady_clock::now();
  float device_ms = 0.0f;
  CHECK(aclrtEventElapsedTime(&device_ms, start, end));
  CHECK(aclrtDestroyEvent(start));
  CHECK(aclrtDestroyEvent(end));
  return {device_ms * 1000.0f / static_cast<float>(iters),
          std::chrono::duration<double, std::micro>(host_end - host_start).count() / iters};
}

void RunFmaAddcmul(aclTensor* self, aclTensor* a, aclTensor* b, aclScalar* value,
                   aclTensor* out, Workspace* workspace, aclrtStream stream) {
  uint64_t size = 0;
  aclOpExecutor* executor = nullptr;
  CHECK(aclnnFmaAddcmulGetWorkspaceSize(self, a, b, value, out, &size, &executor));
  if (size != 0) {
    std::printf("FAIL custom workspace: %llu\n", static_cast<unsigned long long>(size));
    std::exit(1);
  }
  CHECK(aclnnFmaAddcmul(workspace->Get(size), size, executor, stream));
}

void RunAddcmul(aclTensor* self, aclTensor* a, aclTensor* b, aclScalar* value,
                aclTensor* out, Workspace* workspace, aclrtStream stream) {
  uint64_t size = 0;
  aclOpExecutor* executor = nullptr;
  CHECK(aclnnAddcmulGetWorkspaceSize(self, a, b, value, out, &size, &executor));
  CHECK(aclnnAddcmul(workspace->Get(size), size, executor, stream));
}
}  // namespace

int main() {
  constexpr int kElements = 4096;
  constexpr int kWarmup = 100;
  constexpr int kIterations = 1000;
  CHECK(aclInit(nullptr));
  CHECK(aclrtSetDevice(0));
  aclrtStream stream = nullptr;
  CHECK(aclrtCreateStream(&stream));

  std::vector<float> self(kElements), a(kElements), b(kElements), out(kElements);
  for (int i = 0; i < kElements; ++i) {
    self[i] = static_cast<float>((i * 17) % 101 - 50) * 0.03125f;
    a[i] = static_cast<float>((i * 13) % 97 - 48) * 0.015625f;
    b[i] = static_cast<float>((i * 29) % 89 - 44) * 0.0078125f;
  }
  void *dself = nullptr, *da = nullptr, *db = nullptr, *dout = nullptr;
  CHECK(aclrtMalloc(&dself, self.size() * sizeof(float), ACL_MEM_MALLOC_NORMAL_ONLY));
  CHECK(aclrtMalloc(&da, a.size() * sizeof(float), ACL_MEM_MALLOC_NORMAL_ONLY));
  CHECK(aclrtMalloc(&db, b.size() * sizeof(float), ACL_MEM_MALLOC_NORMAL_ONLY));
  CHECK(aclrtMalloc(&dout, out.size() * sizeof(float), ACL_MEM_MALLOC_NORMAL_ONLY));
  CHECK(aclrtMemcpy(dself, self.size() * sizeof(float), self.data(), self.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK(aclrtMemcpy(da, a.size() * sizeof(float), a.data(), a.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK(aclrtMemcpy(db, b.size() * sizeof(float), b.data(), b.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
  aclTensor *ts = Tensor({kElements}, dself), *ta = Tensor({kElements}, da);
  aclTensor *tb = Tensor({kElements}, db), *to = Tensor({kElements}, dout);
  float value = 1.25f;
  aclScalar* scalar = aclCreateScalar(&value, ACL_FLOAT);
  if (!ts || !ta || !tb || !to || !scalar) return 1;
  Workspace custom_workspace, native_workspace;
  const Result custom = Measure([&] { RunFmaAddcmul(ts, ta, tb, scalar, to, &custom_workspace, stream); },
                                kWarmup, kIterations, stream);
  const Result native = Measure([&] { RunAddcmul(ts, ta, tb, scalar, to, &native_workspace, stream); },
                                kWarmup, kIterations, stream);
  std::printf("elements,variant,iters,device_avg_us,host_avg_us,workspace_bytes,note\n");
  std::printf("%d,fma_addcmul_custom,%d,%.6f,%.6f,0,single_kernel_exact_fma\n",
              kElements, kIterations, custom.device_us, custom.host_us);
  std::printf("%d,aclnn_addcmul,%d,%.6f,%.6f,%llu,ordinary_nonfused_reference\n",
              kElements, kIterations, native.device_us, native.host_us,
              static_cast<unsigned long long>(native_workspace.capacity));
  std::printf("custom_to_native_device_ratio,%.3f\n", custom.device_us / native.device_us);
  aclDestroyScalar(scalar);
  aclDestroyTensor(ts); aclDestroyTensor(ta); aclDestroyTensor(tb); aclDestroyTensor(to);
  CHECK(aclrtFree(dself)); CHECK(aclrtFree(da)); CHECK(aclrtFree(db)); CHECK(aclrtFree(dout));
  CHECK(aclrtDestroyStream(stream));
  CHECK(aclrtResetDevice(0));
  CHECK(aclFinalize());
  return 0;
}
