#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include "aclnn_dot_fp32_left_order_neumaier.h"
#include "aclnn_dot_fp32_left_order_no_neumaier.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

struct CaseDef {
  const char* name;
  std::vector<int64_t> lhs_shape;
  std::vector<int64_t> rhs_shape;
  std::vector<int64_t> out_shape;
};

struct PrecisionStats {
  double max_no_ref_error = 0.0;
  double max_neu_ref_error = 0.0;
  double max_no_f64_error = 0.0;
  double max_neu_f64_error = 0.0;
};

static int64_t Numel(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t dim : shape) {
    n *= dim;
  }
  return n;
}

static int64_t DotLen(const std::vector<int64_t>& shape) {
  return shape.back();
}

static int64_t Batch(const std::vector<int64_t>& shape) {
  if (shape.size() <= 1) {
    return 1;
  }
  int64_t n = 1;
  for (size_t i = 0; i + 1 < shape.size(); ++i) {
    n *= shape[i];
  }
  return n;
}

static uint32_t FloatBits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static std::vector<int64_t> COrderStrides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  return strides;
}

static aclTensor* CreateTensor(const std::vector<int64_t>& shape, void* addr) {
  std::vector<int64_t> strides = COrderStrides(shape);
  const int64_t* shape_ptr = shape.empty() ? nullptr : shape.data();
  const int64_t* strides_ptr = strides.empty() ? nullptr : strides.data();
  aclTensor* tensor = aclCreateTensor(
      shape_ptr, shape.size(), ACL_FLOAT, strides_ptr, 0, ACL_FORMAT_ND,
      shape_ptr, shape.size(), addr);
  if (tensor == nullptr) {
    std::printf("FAIL aclCreateTensor rank=%zu\n", shape.size());
    std::exit(1);
  }
  return tensor;
}

static void* DeviceMalloc(size_t bytes) {
  void* p = nullptr;
  CHECK_ACL(aclrtMalloc(&p, bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc");
  return p;
}

static void Fill(std::vector<float>& values, int seed) {
  for (size_t i = 0; i < values.size(); ++i) {
    uint32_t state = static_cast<uint32_t>(i + 1) * 1664525u +
                     static_cast<uint32_t>(seed) * 1013904223u;
    state ^= state >> 13;
    state *= 747796405u;
    const int coarse = static_cast<int>(state % 4001u) - 2000;
    const int fine = static_cast<int>((state >> 16) % 97u) - 48;
    values[i] = static_cast<float>(coarse) * 0.00137f +
                static_cast<float>(fine) * 0.000019f;
  }
}

static float AddFloat(float lhs, float rhs) {
  volatile float result = lhs + rhs;
  return result;
}

static float MulFloat(float lhs, float rhs) {
  volatile float result = lhs * rhs;
  return result;
}

static float FmaFloat(float lhs, float rhs, float addend) {
  return std::fma(lhs, rhs, addend);
}

static float CpuDotJaxAarch64Row(const std::vector<float>& lhs, const std::vector<int64_t>& lhs_shape,
                                 const std::vector<float>& rhs, const std::vector<int64_t>& rhs_shape,
                                 int64_t row) {
  const int64_t dot_len = DotLen(lhs_shape);
  const int64_t lhs_base = lhs_shape.size() == 1 ? 0 : row * dot_len;
  const int64_t rhs_base = rhs_shape.size() == 1 ? 0 : row * dot_len;
  float sum = 0.0f;
  if (dot_len > 0) {
    sum = MulFloat(lhs[lhs_base], rhs[rhs_base]);
  }
  for (int64_t k = 1; k < dot_len && k < 8; ++k) {
    const float product = MulFloat(lhs[lhs_base + k], rhs[rhs_base + k]);
    sum = AddFloat(product, sum);
  }
  for (int64_t k = 8; k < dot_len; ++k) {
    sum = FmaFloat(lhs[lhs_base + k], rhs[rhs_base + k], sum);
  }
  return sum;
}

static float CpuDotNeumaierRow(const std::vector<float>& lhs, const std::vector<int64_t>& lhs_shape,
                               const std::vector<float>& rhs, const std::vector<int64_t>& rhs_shape,
                               int64_t row) {
  const int64_t dot_len = DotLen(lhs_shape);
  const int64_t lhs_base = lhs_shape.size() == 1 ? 0 : row * dot_len;
  const int64_t rhs_base = rhs_shape.size() == 1 ? 0 : row * dot_len;
  float sum = 0.0f;
  float compensation = 0.0f;
  for (int64_t k = 0; k < dot_len; ++k) {
    const float product = lhs[lhs_base + k] * rhs[rhs_base + k];
    const float next_sum = sum + product;
    float correction = 0.0f;
    if (std::fabs(sum) >= std::fabs(product)) {
      correction = (sum - next_sum) + product;
    } else {
      correction = (product - next_sum) + sum;
    }
    compensation = compensation + correction;
    sum = next_sum;
  }
  return sum + compensation;
}

static double CpuDotDoubleRow(const std::vector<float>& lhs, const std::vector<int64_t>& lhs_shape,
                              const std::vector<float>& rhs, const std::vector<int64_t>& rhs_shape,
                              int64_t row) {
  const int64_t dot_len = DotLen(lhs_shape);
  const int64_t lhs_base = lhs_shape.size() == 1 ? 0 : row * dot_len;
  const int64_t rhs_base = rhs_shape.size() == 1 ? 0 : row * dot_len;
  double sum = 0.0;
  for (int64_t k = 0; k < dot_len; ++k) {
    sum += static_cast<double>(lhs[lhs_base + k]) *
           static_cast<double>(rhs[rhs_base + k]);
  }
  return sum;
}

static void RunNoNeumaier(aclTensor* lhs, aclTensor* rhs, aclTensor* out, aclrtStream stream) {
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  CHECK_ACLNN(aclnnDotFp32LeftOrderNoNeumaierGetWorkspaceSize(
                  lhs, rhs, out, &workspace_size, &executor),
              "aclnnDotFp32LeftOrderNoNeumaierGetWorkspaceSize");
  void* workspace = workspace_size == 0 ? nullptr : DeviceMalloc(workspace_size);
  CHECK_ACLNN(aclnnDotFp32LeftOrderNoNeumaier(workspace, workspace_size, executor, stream),
              "aclnnDotFp32LeftOrderNoNeumaier");
  CHECK_ACL(aclrtSynchronizeStream(stream), "sync no neumaier");
  if (workspace != nullptr) {
    aclrtFree(workspace);
  }
}

static void RunNeumaier(aclTensor* lhs, aclTensor* rhs, aclTensor* out, aclrtStream stream) {
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  CHECK_ACLNN(aclnnDotFp32LeftOrderNeumaierGetWorkspaceSize(
                  lhs, rhs, out, &workspace_size, &executor),
              "aclnnDotFp32LeftOrderNeumaierGetWorkspaceSize");
  void* workspace = workspace_size == 0 ? nullptr : DeviceMalloc(workspace_size);
  CHECK_ACLNN(aclnnDotFp32LeftOrderNeumaier(workspace, workspace_size, executor, stream),
              "aclnnDotFp32LeftOrderNeumaier");
  CHECK_ACL(aclrtSynchronizeStream(stream), "sync neumaier");
  if (workspace != nullptr) {
    aclrtFree(workspace);
  }
}

static bool CheckCase(const CaseDef& c, aclrtStream stream) {
  std::vector<float> lhs(Numel(c.lhs_shape));
  std::vector<float> rhs(Numel(c.rhs_shape));
  std::vector<float> out_no(Numel(c.out_shape), std::numeric_limits<float>::quiet_NaN());
  std::vector<float> out_neu(Numel(c.out_shape), std::numeric_limits<float>::quiet_NaN());
  std::vector<float> out_init(Numel(c.out_shape), 12345.0f);
  Fill(lhs, 3);
  Fill(rhs, 11);

  void* d_lhs = DeviceMalloc(lhs.size() * sizeof(float));
  void* d_rhs = DeviceMalloc(rhs.size() * sizeof(float));
  void* d_out_no = DeviceMalloc(out_no.size() * sizeof(float));
  void* d_out_neu = DeviceMalloc(out_neu.size() * sizeof(float));

  CHECK_ACL(aclrtMemcpy(d_lhs, lhs.size() * sizeof(float), lhs.data(),
                        lhs.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE),
            "H2D lhs");
  CHECK_ACL(aclrtMemcpy(d_rhs, rhs.size() * sizeof(float), rhs.data(),
                        rhs.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE),
            "H2D rhs");
  CHECK_ACL(aclrtMemcpy(d_out_no, out_init.size() * sizeof(float), out_init.data(),
                        out_init.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE),
            "H2D out_no");
  CHECK_ACL(aclrtMemcpy(d_out_neu, out_init.size() * sizeof(float), out_init.data(),
                        out_init.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE),
            "H2D out_neu");

  aclTensor* lhs_tensor = CreateTensor(c.lhs_shape, d_lhs);
  aclTensor* rhs_tensor = CreateTensor(c.rhs_shape, d_rhs);
  aclTensor* out_no_tensor = CreateTensor(c.out_shape, d_out_no);
  aclTensor* out_neu_tensor = CreateTensor(c.out_shape, d_out_neu);

  RunNoNeumaier(lhs_tensor, rhs_tensor, out_no_tensor, stream);
  RunNeumaier(lhs_tensor, rhs_tensor, out_neu_tensor, stream);

  CHECK_ACL(aclrtMemcpy(out_no.data(), out_no.size() * sizeof(float), d_out_no,
                        out_no.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST),
            "D2H out_no");
  CHECK_ACL(aclrtMemcpy(out_neu.data(), out_neu.size() * sizeof(float), d_out_neu,
                        out_neu.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST),
            "D2H out_neu");

  bool ok = true;
  PrecisionStats stats;
  const int64_t rows = Numel(c.out_shape);
  for (int64_t row = 0; row < rows; ++row) {
    const float ref_no = CpuDotJaxAarch64Row(lhs, c.lhs_shape, rhs, c.rhs_shape, row);
    const float ref_neu = CpuDotNeumaierRow(lhs, c.lhs_shape, rhs, c.rhs_shape, row);
    const double ref_f64 = CpuDotDoubleRow(lhs, c.lhs_shape, rhs, c.rhs_shape, row);
    const double no_ref_error =
        std::fabs(static_cast<double>(out_no[row]) - static_cast<double>(ref_no));
    const double neu_ref_error =
        std::fabs(static_cast<double>(out_neu[row]) - static_cast<double>(ref_neu));
    const double no_f64_error = std::fabs(static_cast<double>(out_no[row]) - ref_f64);
    const double neu_f64_error = std::fabs(static_cast<double>(out_neu[row]) - ref_f64);
    if (no_ref_error > stats.max_no_ref_error) {
      stats.max_no_ref_error = no_ref_error;
    }
    if (neu_ref_error > stats.max_neu_ref_error) {
      stats.max_neu_ref_error = neu_ref_error;
    }
    if (no_f64_error > stats.max_no_f64_error) {
      stats.max_no_f64_error = no_f64_error;
    }
    if (neu_f64_error > stats.max_neu_f64_error) {
      stats.max_neu_f64_error = neu_f64_error;
    }
    if (out_no[row] != ref_no || out_neu[row] != ref_neu) {
      std::printf(
          "FAIL %s row=%ld no got %.12f 0x%08x ref %.12f 0x%08x "
          "neu got %.12f 0x%08x ref %.12f 0x%08x\n",
          c.name, static_cast<long>(row), out_no[row], FloatBits(out_no[row]),
          ref_no, FloatBits(ref_no), out_neu[row], FloatBits(out_neu[row]),
          ref_neu, FloatBits(ref_neu));
      ok = false;
      break;
    }
  }
  std::printf(
      "%s %s rows=%ld no_ref_max=%.12g neu_ref_max=%.12g "
      "no_f64_max=%.12g neu_f64_max=%.12g\n",
      ok ? "PASS" : "FAIL", c.name, static_cast<long>(rows),
      stats.max_no_ref_error, stats.max_neu_ref_error, stats.max_no_f64_error,
      stats.max_neu_f64_error);

  aclDestroyTensor(lhs_tensor);
  aclDestroyTensor(rhs_tensor);
  aclDestroyTensor(out_no_tensor);
  aclDestroyTensor(out_neu_tensor);
  aclrtFree(d_lhs);
  aclrtFree(d_rhs);
  aclrtFree(d_out_no);
  aclrtFree(d_out_neu);
  return ok;
}

int main() {
  const std::vector<CaseDef> cases = {
      {"vector_vector", {18}, {18}, {}},
      {"batch_batch_2d", {7, 18}, {7, 18}, {7}},
      {"vector_batch_2d", {18}, {7, 18}, {7}},
      {"batch_vector_2d", {7, 18}, {18}, {7}},
      {"batch_batch_3d", {2, 3, 18}, {2, 3, 18}, {2, 3}},
      {"vector_batch_3d", {18}, {2, 3, 18}, {2, 3}},
      {"batch_vector_3d", {2, 3, 18}, {18}, {2, 3}},
      {"batch1024_batch_2d", {1024, 18}, {1024, 18}, {1024}},
      {"batch4096_batch_2d", {4096, 18}, {4096, 18}, {4096}},
  };

  CHECK_ACL(aclInit(nullptr), "aclInit");
  CHECK_ACL(aclrtSetDevice(0), "aclrtSetDevice");
  aclrtStream stream = nullptr;
  CHECK_ACL(aclrtCreateStream(&stream), "aclrtCreateStream");

  bool ok = true;
  for (const CaseDef& c : cases) {
    ok = CheckCase(c, stream) && ok;
  }

  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();
  std::printf("%s DotFp32LeftOrder precision verification\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 2;
}
