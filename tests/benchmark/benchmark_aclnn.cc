#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_dot.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_reduce_sum.h>
#include "aclnn_dot_fp32_left_order_neumaier.h"
#include "aclnn_dot_fp32_left_order_no_neumaier.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#define CHECK_ACL(expr, msg)                                                  \
  do {                                                                        \
    aclError _ret = (expr);                                                   \
    if (_ret != ACL_SUCCESS) {                                                \
      std::printf("FAIL %s: ret=%d\n", (msg), static_cast<int>(_ret));        \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

#define CHECK_ACLNN(expr, msg)                                                \
  do {                                                                        \
    aclnnStatus _ret = (expr);                                                \
    if (_ret != 0) {                                                          \
      std::printf("FAIL %s: ret=%d\n", (msg), static_cast<int>(_ret));        \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

namespace {

struct ShapeCase {
  int64_t batch;
  int64_t dot_len;
  int warmup;
  int iters;
  int dot_loop_warmup;
  int dot_loop_iters;
};

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t bytes) { Reset(bytes); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() { Free(); }

  void Reset(size_t bytes) {
    Free();
    bytes_ = bytes;
    if (bytes_ > 0) {
      CHECK_ACL(aclrtMalloc(&ptr_, bytes_, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc");
    }
  }

  void* Get() const { return ptr_; }
  size_t Size() const { return bytes_; }

 private:
  void Free() {
    if (ptr_ != nullptr) {
      aclrtFree(ptr_);
      ptr_ = nullptr;
      bytes_ = 0;
    }
  }

  void* ptr_ = nullptr;
  size_t bytes_ = 0;
};

class WorkspaceCache {
 public:
  WorkspaceCache() = default;
  WorkspaceCache(const WorkspaceCache&) = delete;
  WorkspaceCache& operator=(const WorkspaceCache&) = delete;
  ~WorkspaceCache() {
    if (ptr_ != nullptr) {
      aclrtFree(ptr_);
    }
  }

  void* Get(uint64_t bytes) {
    if (bytes == 0) {
      return nullptr;
    }
    if (bytes > capacity_) {
      if (ptr_ != nullptr) {
        aclrtFree(ptr_);
        ptr_ = nullptr;
        capacity_ = 0;
      }
      CHECK_ACL(aclrtMalloc(&ptr_, bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc workspace");
      capacity_ = bytes;
    }
    return ptr_;
  }

 private:
  void* ptr_ = nullptr;
  uint64_t capacity_ = 0;
};

std::vector<int64_t> COrderStrides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  return strides;
}

aclTensor* CreateTensor(const std::vector<int64_t>& shape, void* addr) {
  std::vector<int64_t> strides = COrderStrides(shape);
  const int64_t* shape_ptr = shape.empty() ? nullptr : shape.data();
  const int64_t* stride_ptr = strides.empty() ? nullptr : strides.data();
  aclTensor* tensor = aclCreateTensor(shape_ptr, shape.size(), ACL_FLOAT, stride_ptr,
                                      0, ACL_FORMAT_ND, shape_ptr, shape.size(), addr);
  if (tensor == nullptr) {
    std::printf("FAIL aclCreateTensor rank=%zu\n", shape.size());
    std::exit(1);
  }
  return tensor;
}

void FillInputs(std::vector<float>& lhs, std::vector<float>& rhs) {
  for (size_t i = 0; i < lhs.size(); ++i) {
    const int a = static_cast<int>(i % 251);
    const int b = static_cast<int>((i * 17 + 13) % 257);
    lhs[i] = (static_cast<float>(a) - 125.0f) * 0.0078125f;
    rhs[i] = (static_cast<float>(b) - 128.0f) * 0.0068359375f;
  }
}

void RunDotFp32LeftOrderNoNeumaier(aclTensor* lhs, aclTensor* rhs, aclTensor* out,
                                   aclrtStream stream, WorkspaceCache& workspace) {
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  CHECK_ACLNN(aclnnDotFp32LeftOrderNoNeumaierGetWorkspaceSize(
                  lhs, rhs, out, &workspace_size, &executor),
              "aclnnDotFp32LeftOrderNoNeumaierGetWorkspaceSize");
  CHECK_ACLNN(aclnnDotFp32LeftOrderNoNeumaier(
                  workspace.Get(workspace_size), workspace_size, executor, stream),
              "aclnnDotFp32LeftOrderNoNeumaier");
}

void RunDotFp32LeftOrderNeumaier(aclTensor* lhs, aclTensor* rhs, aclTensor* out,
                                 aclrtStream stream, WorkspaceCache& workspace) {
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  CHECK_ACLNN(aclnnDotFp32LeftOrderNeumaierGetWorkspaceSize(
                  lhs, rhs, out, &workspace_size, &executor),
              "aclnnDotFp32LeftOrderNeumaierGetWorkspaceSize");
  CHECK_ACLNN(aclnnDotFp32LeftOrderNeumaier(
                  workspace.Get(workspace_size), workspace_size, executor, stream),
              "aclnnDotFp32LeftOrderNeumaier");
}

void RunNativeDot(aclTensor* lhs, aclTensor* rhs, aclTensor* out, aclrtStream stream,
                  WorkspaceCache& workspace) {
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  CHECK_ACLNN(aclnnDotGetWorkspaceSize(lhs, rhs, out, &workspace_size, &executor),
              "aclnnDotGetWorkspaceSize");
  CHECK_ACLNN(aclnnDot(workspace.Get(workspace_size), workspace_size, executor, stream),
              "aclnnDot");
}

void RunNativeMulReduceSum(aclTensor* lhs, aclTensor* rhs, aclTensor* tmp, aclTensor* out,
                           const aclIntArray* reduce_dims, aclrtStream stream,
                           WorkspaceCache& workspace) {
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  CHECK_ACLNN(aclnnMulGetWorkspaceSize(lhs, rhs, tmp, &workspace_size, &executor),
              "aclnnMulGetWorkspaceSize");
  CHECK_ACLNN(aclnnMul(workspace.Get(workspace_size), workspace_size, executor, stream),
              "aclnnMul");

  workspace_size = 0;
  executor = nullptr;
  CHECK_ACLNN(aclnnReduceSumGetWorkspaceSize(tmp, reduce_dims, false, ACL_FLOAT, out,
                                             &workspace_size, &executor),
              "aclnnReduceSumGetWorkspaceSize");
  CHECK_ACLNN(aclnnReduceSum(workspace.Get(workspace_size), workspace_size, executor, stream),
              "aclnnReduceSum");
}

struct TimingResult {
  float device_ms = 0.0f;
  double host_ms = 0.0;
};

TimingResult TimeLoop(const std::function<void()>& fn, int warmup, int iters,
                      aclrtStream stream) {
  for (int i = 0; i < warmup; ++i) {
    fn();
  }
  CHECK_ACL(aclrtSynchronizeStream(stream), "warmup sync");

  aclrtEvent start = nullptr;
  aclrtEvent end = nullptr;
  CHECK_ACL(aclrtCreateEvent(&start), "aclrtCreateEvent start");
  CHECK_ACL(aclrtCreateEvent(&end), "aclrtCreateEvent end");

  const auto host_start = std::chrono::steady_clock::now();
  CHECK_ACL(aclrtRecordEvent(start, stream), "record start");
  for (int i = 0; i < iters; ++i) {
    fn();
  }
  CHECK_ACL(aclrtRecordEvent(end, stream), "record end");
  CHECK_ACL(aclrtSynchronizeEvent(end), "sync end event");
  const auto host_end = std::chrono::steady_clock::now();

  float device_ms = 0.0f;
  CHECK_ACL(aclrtEventElapsedTime(&device_ms, start, end), "event elapsed");
  CHECK_ACL(aclrtDestroyEvent(start), "destroy start");
  CHECK_ACL(aclrtDestroyEvent(end), "destroy end");

  const double host_ms =
      std::chrono::duration<double, std::milli>(host_end - host_start).count();
  return {device_ms / static_cast<float>(iters), host_ms / static_cast<double>(iters)};
}

void PrintResult(const ShapeCase& shape, const char* variant, int iters,
                 const TimingResult& result, const char* note) {
  const double ops = 2.0 * static_cast<double>(shape.batch) *
                     static_cast<double>(shape.dot_len);
  const double gflops =
      result.device_ms > 0.0f ? ops / (static_cast<double>(result.device_ms) * 1.0e6) : 0.0;
  std::printf("%ld,%ld,%ld,%s,%d,%.6f,%.6f,%.3f,%s\n",
              static_cast<long>(shape.batch), static_cast<long>(shape.dot_len),
              static_cast<long>(shape.batch * shape.dot_len), variant, iters,
              static_cast<double>(result.device_ms) * 1000.0, result.host_ms * 1000.0,
              gflops, note);
}

void BenchmarkShape(const ShapeCase& shape, aclrtStream stream) {
  const int64_t elements = shape.batch * shape.dot_len;
  std::vector<float> lhs_host(elements);
  std::vector<float> rhs_host(elements);
  std::vector<float> out_init(shape.batch, 0.0f);
  FillInputs(lhs_host, rhs_host);

  DeviceBuffer lhs_dev(lhs_host.size() * sizeof(float));
  DeviceBuffer rhs_dev(rhs_host.size() * sizeof(float));
  DeviceBuffer out_no_neumaier(shape.batch * sizeof(float));
  DeviceBuffer out_neumaier(shape.batch * sizeof(float));
  DeviceBuffer out_native_dot(shape.batch * sizeof(float));
  DeviceBuffer out_mul_reduce(shape.batch * sizeof(float));
  DeviceBuffer tmp_mul(lhs_host.size() * sizeof(float));

  CHECK_ACL(aclrtMemcpy(lhs_dev.Get(), lhs_dev.Size(), lhs_host.data(), lhs_dev.Size(),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            "copy lhs");
  CHECK_ACL(aclrtMemcpy(rhs_dev.Get(), rhs_dev.Size(), rhs_host.data(), rhs_dev.Size(),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            "copy rhs");
  CHECK_ACL(aclrtMemcpy(out_no_neumaier.Get(), out_no_neumaier.Size(), out_init.data(),
                        out_no_neumaier.Size(), ACL_MEMCPY_HOST_TO_DEVICE),
            "copy out_no_neumaier");
  CHECK_ACL(aclrtMemcpy(out_neumaier.Get(), out_neumaier.Size(), out_init.data(),
                        out_neumaier.Size(), ACL_MEMCPY_HOST_TO_DEVICE),
            "copy out_neumaier");
  CHECK_ACL(aclrtMemcpy(out_native_dot.Get(), out_native_dot.Size(), out_init.data(),
                        out_native_dot.Size(), ACL_MEMCPY_HOST_TO_DEVICE),
            "copy out_native_dot");
  CHECK_ACL(aclrtMemcpy(out_mul_reduce.Get(), out_mul_reduce.Size(), out_init.data(),
                        out_mul_reduce.Size(), ACL_MEMCPY_HOST_TO_DEVICE),
            "copy out_mul_reduce");

  aclTensor* lhs_2d = CreateTensor({shape.batch, shape.dot_len}, lhs_dev.Get());
  aclTensor* rhs_2d = CreateTensor({shape.batch, shape.dot_len}, rhs_dev.Get());
  aclTensor* out_no_neumaier_1d = CreateTensor({shape.batch}, out_no_neumaier.Get());
  aclTensor* out_neumaier_1d = CreateTensor({shape.batch}, out_neumaier.Get());
  aclTensor* out_mul_reduce_1d = CreateTensor({shape.batch}, out_mul_reduce.Get());
  aclTensor* tmp_mul_2d = CreateTensor({shape.batch, shape.dot_len}, tmp_mul.Get());
  const int64_t reduce_axis = 1;
  aclIntArray* reduce_dims = aclCreateIntArray(&reduce_axis, 1);
  if (reduce_dims == nullptr) {
    std::printf("FAIL aclCreateIntArray reduce_dims\n");
    std::exit(1);
  }

  std::vector<aclTensor*> lhs_rows;
  std::vector<aclTensor*> rhs_rows;
  std::vector<aclTensor*> out_scalars;
  lhs_rows.reserve(shape.batch);
  rhs_rows.reserve(shape.batch);
  out_scalars.reserve(shape.batch);
  for (int64_t row = 0; row < shape.batch; ++row) {
    const int64_t offset = row * shape.dot_len;
    lhs_rows.push_back(CreateTensor({shape.dot_len},
                                    static_cast<float*>(lhs_dev.Get()) + offset));
    rhs_rows.push_back(CreateTensor({shape.dot_len},
                                    static_cast<float*>(rhs_dev.Get()) + offset));
    out_scalars.push_back(CreateTensor({},
                                       static_cast<float*>(out_native_dot.Get()) + row));
  }

  WorkspaceCache no_neumaier_workspace;
  WorkspaceCache neumaier_workspace;
  WorkspaceCache native_dot_workspace;
  WorkspaceCache mul_reduce_workspace;

  const TimingResult no_neumaier_time = TimeLoop(
      [&]() {
        RunDotFp32LeftOrderNoNeumaier(lhs_2d, rhs_2d, out_no_neumaier_1d, stream,
                                      no_neumaier_workspace);
      },
      shape.warmup, shape.iters, stream);
  PrintResult(shape, "custom_no_neumaier", shape.iters, no_neumaier_time,
              "one_aclnn_call");

  const TimingResult neumaier_time = TimeLoop(
      [&]() {
        RunDotFp32LeftOrderNeumaier(lhs_2d, rhs_2d, out_neumaier_1d, stream,
                                    neumaier_workspace);
      },
      shape.warmup, shape.iters, stream);
  PrintResult(shape, "custom_neumaier", shape.iters, neumaier_time, "one_aclnn_call");

  const TimingResult mul_reduce_time = TimeLoop(
      [&]() {
        RunNativeMulReduceSum(lhs_2d, rhs_2d, tmp_mul_2d, out_mul_reduce_1d,
                              reduce_dims, stream, mul_reduce_workspace);
      },
      shape.warmup, shape.iters, stream);
  PrintResult(shape, "native_mul_reduce_sum", shape.iters, mul_reduce_time,
              "aclnnMul_plus_aclnnReduceSum");

  const TimingResult native_dot_time = TimeLoop(
      [&]() {
        for (int64_t row = 0; row < shape.batch; ++row) {
          RunNativeDot(lhs_rows[row], rhs_rows[row], out_scalars[row], stream,
                       native_dot_workspace);
        }
      },
      shape.dot_loop_warmup, shape.dot_loop_iters, stream);
  PrintResult(shape, "native_dot_loop", shape.dot_loop_iters, native_dot_time,
              "batch_aclnnDot_calls_1d_only");

  for (aclTensor* tensor : lhs_rows) {
    aclDestroyTensor(tensor);
  }
  for (aclTensor* tensor : rhs_rows) {
    aclDestroyTensor(tensor);
  }
  for (aclTensor* tensor : out_scalars) {
    aclDestroyTensor(tensor);
  }
  aclDestroyIntArray(reduce_dims);
  aclDestroyTensor(lhs_2d);
  aclDestroyTensor(rhs_2d);
  aclDestroyTensor(out_no_neumaier_1d);
  aclDestroyTensor(out_neumaier_1d);
  aclDestroyTensor(out_mul_reduce_1d);
  aclDestroyTensor(tmp_mul_2d);
}

}  // namespace

int main() {
  const std::vector<ShapeCase> shapes = {
      {1, 18, 20, 200, 20, 200},
      {16, 18, 20, 200, 10, 100},
      {1024, 18, 10, 100, 2, 5},
      {4096, 18, 10, 50, 1, 2},
      {1024, 64, 10, 100, 2, 5},
      {1024, 256, 5, 50, 1, 3},
      {256, 1024, 5, 50, 1, 5},
      {64, 4096, 5, 30, 1, 10},
  };

  CHECK_ACL(aclInit(nullptr), "aclInit");
  CHECK_ACL(aclrtSetDevice(0), "aclrtSetDevice");
  aclrtStream stream = nullptr;
  CHECK_ACL(aclrtCreateStream(&stream), "aclrtCreateStream");

  std::printf("batch,dot_len,elements,variant,iters,device_avg_us,host_avg_us,device_gflops,note\n");
  for (const ShapeCase& shape : shapes) {
    BenchmarkShape(shape, stream);
  }

  CHECK_ACL(aclrtDestroyStream(stream), "aclrtDestroyStream");
  CHECK_ACL(aclrtResetDevice(0), "aclrtResetDevice");
  CHECK_ACL(aclFinalize(), "aclFinalize");
  return 0;
}
