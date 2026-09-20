#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include "aclnn_fma_addcmul.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#define CHECK_ACL(expr) do { aclError r = (expr); if (r != ACL_SUCCESS) { \
  std::printf("FAIL %s: %d\n", #expr, static_cast<int>(r)); return false; } } while (0)

static uint32_t Bits(float x) { uint32_t bits; std::memcpy(&bits, &x, 4); return bits; }
static float FromBits(uint32_t bits) { float x; std::memcpy(&x, &bits, 4); return x; }
static float ScaleRN(float value, float x) { volatile float p = value * x; return p; }
static float Reference(float self, float a, float b, float value) {
  return std::fmaf(ScaleRN(value, a), b, self);
}

static int64_t Numel(const std::vector<int64_t>& shape) {
  int64_t n = 1; for (int64_t d : shape) n *= d; return n;
}
static std::vector<int64_t> Strides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> s(shape.size(), 1);
  for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) s[i] = s[i + 1] * shape[i + 1];
  return s;
}
static aclTensor* Tensor(const std::vector<int64_t>& shape, void* p, aclDataType dtype = ACL_FLOAT) {
  const auto strides = Strides(shape);
  return aclCreateTensor(shape.empty() ? nullptr : shape.data(), shape.size(), dtype,
      strides.empty() ? nullptr : strides.data(), 0, ACL_FORMAT_ND,
      shape.empty() ? nullptr : shape.data(), shape.size(), p);
}
static aclTensor* TensorWithStrides(const std::vector<int64_t>& shape,
                                    const std::vector<int64_t>& strides, void* p) {
  return aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0,
      ACL_FORMAT_ND, shape.data(), shape.size(), p);
}
static void* Malloc(size_t bytes) { void* p = nullptr; if (aclrtMalloc(&p, bytes, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS) return nullptr; return p; }

static bool Run(const char* name, const std::vector<int64_t>& shape,
                const std::vector<float>& self, const std::vector<float>& a,
                const std::vector<float>& b, float value, aclrtStream stream,
                int repetitions = 1) {
  const size_t bytes = self.size() * sizeof(float);
  void *ds = Malloc(bytes), *da = Malloc(bytes), *db = Malloc(bytes), *do_ = Malloc(bytes);
  if (!ds || !da || !db || !do_) return false;
  bool ok = true;
  CHECK_ACL(aclrtMemcpy(ds, bytes, self.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK_ACL(aclrtMemcpy(da, bytes, a.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK_ACL(aclrtMemcpy(db, bytes, b.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
  aclTensor *ts = Tensor(shape, ds), *ta = Tensor(shape, da), *tb = Tensor(shape, db), *to = Tensor(shape, do_);
  aclScalar* scalar = aclCreateScalar(&value, ACL_FLOAT);
  if (!ts || !ta || !tb || !to || !scalar) ok = false;
  uint64_t workspace_size = 0; aclOpExecutor* executor = nullptr;
  if (ok && aclnnFmaAddcmulGetWorkspaceSize(ts, ta, tb, scalar, to, &workspace_size, &executor) != ACL_SUCCESS) ok = false;
  if (ok && workspace_size != 0) {
    std::printf("FAIL %s: workspace=%llu, expected 0\n", name,
                static_cast<unsigned long long>(workspace_size));
    ok = false;
  }
  void* workspace = workspace_size == 0 ? nullptr : Malloc(workspace_size);
  if (ok && workspace_size != 0 && workspace == nullptr) ok = false;
  std::vector<float> first;
  for (int run = 0; ok && run < repetitions; ++run) {
    if (aclnnFmaAddcmul(workspace, workspace_size, executor, stream) != ACL_SUCCESS ||
        aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
      ok = false;
      break;
    }
    std::vector<float> got(self.size());
    if (aclrtMemcpy(got.data(), bytes, do_, bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
      ok = false;
      break;
    }
    if (run == 0) first = got;
    for (size_t i = 0; ok && i < got.size(); ++i) {
      const float ref = Reference(self[i], a[i], b[i], value);
      if (std::isnan(ref)) { ok = std::isnan(got[i]); }
      else { ok = Bits(got[i]) == Bits(ref) && Bits(got[i]) == Bits(first[i]); }
      if (!ok) std::printf("FAIL %s run=%d i=%zu got=0x%08x first=0x%08x ref=0x%08x\n",
                           name, run, i, Bits(got[i]), Bits(first[i]), Bits(ref));
    }
  }
  std::printf("%s %s n=%zu runs=%d\n", ok ? "PASS" : "FAIL", name,
              self.size(), repetitions);
  if (workspace) aclrtFree(workspace);
  if (scalar) aclDestroyScalar(scalar);
  if (ts) aclDestroyTensor(ts); if (ta) aclDestroyTensor(ta); if (tb) aclDestroyTensor(tb); if (to) aclDestroyTensor(to);
  aclrtFree(ds); aclrtFree(da); aclrtFree(db); aclrtFree(do_);
  return ok;
}

static bool ExpectRejected(const char* name, aclTensor* self, aclTensor* a,
                           aclTensor* b, aclScalar* value, aclTensor* out,
                           void* dout, size_t out_elements) {
  constexpr uint32_t kSentinel = 0x4f1e2d3cu;
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  const aclnnStatus status = aclnnFmaAddcmulGetWorkspaceSize(
      self, a, b, value, out, &workspace_size, &executor);
  std::vector<float> got(out_elements);
  bool unchanged = aclrtMemcpy(got.data(), got.size() * sizeof(float), dout,
                               got.size() * sizeof(float),
                               ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
  for (size_t i = 0; unchanged && i < got.size(); ++i) {
    unchanged = Bits(got[i]) == kSentinel;
  }
  const bool ok = status != 0 && unchanged;
  std::printf("%s %s status=%d out_unchanged=%s\n", ok ? "PASS" : "FAIL", name,
              static_cast<int>(status), unchanged ? "true" : "false");
  return ok;
}

static bool RunInvalidContractTests(aclrtStream stream) {
  (void)stream;
  constexpr uint32_t kSentinel = 0x4f1e2d3cu;
  const float self_value = 3.0f;
  float value = 1.0f;
  const float sentinel = FromBits(kSentinel);
  void *ds = Malloc(2 * sizeof(float)), *da = Malloc(2 * sizeof(float));
  void *db = Malloc(2 * sizeof(float)), *dout = Malloc(2 * sizeof(float));
  if (!ds || !da || !db || !dout) return false;
  float two[2] = {self_value, self_value};
  bool ok = aclrtMemcpy(ds, sizeof(two), two, sizeof(two), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
            aclrtMemcpy(da, sizeof(two), two, sizeof(two), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
            aclrtMemcpy(db, sizeof(two), two, sizeof(two), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
  aclScalar* scalar = aclCreateScalar(&value, ACL_FLOAT);
  uint16_t half_one = 0x3c00;
  aclScalar* scalar_half = aclCreateScalar(&half_one, ACL_FLOAT16);
  const std::vector<int64_t> one_shape{1}, two_shape{2};
  aclTensor *self1 = Tensor(one_shape, ds), *a1 = Tensor(one_shape, da), *b1 = Tensor(one_shape, db);
  aclTensor *a_half = Tensor(one_shape, da, ACL_FLOAT16), *a2 = Tensor(two_shape, da);
  aclTensor *a_noncontig = TensorWithStrides(two_shape, {2}, da);
  aclTensor *out1 = Tensor(one_shape, dout), *out2 = Tensor(two_shape, dout);
  if (!scalar || !scalar_half || !self1 || !a1 || !b1 || !a_half || !a2 || !a_noncontig || !out1 || !out2) ok = false;
  const auto reset_out = [&]() {
    float values[2] = {sentinel, sentinel};
    return aclrtMemcpy(dout, sizeof(values), values, sizeof(values), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
  };
  if (ok && reset_out()) ok &= ExpectRejected("invalid_dtype", self1, a_half, b1, scalar, out1, dout, 1);
  else ok = false;
  if (ok && reset_out()) ok &= ExpectRejected("invalid_scalar_dtype", self1, a1, b1, scalar_half, out1, dout, 1);
  else ok = false;
  if (ok && reset_out()) ok &= ExpectRejected("invalid_shape", self1, a2, b1, scalar, out1, dout, 1);
  else ok = false;
  if (ok && reset_out()) ok &= ExpectRejected("invalid_broadcast", out2, a1, out2, scalar, out2, dout, 2);
  else ok = false;
  if (ok && reset_out()) ok &= ExpectRejected("invalid_noncontiguous", out2, a_noncontig, out2, scalar, out2, dout, 2);
  else ok = false;
  if (ok) {
    // Alias is invalid even though GetWorkspaceSize never launches the kernel.
    const float alias_sentinel = sentinel;
    if (aclrtMemcpy(ds, sizeof(alias_sentinel), &alias_sentinel, sizeof(alias_sentinel),
                    ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS) {
      ok &= ExpectRejected("invalid_output_alias", self1, a1, b1, scalar, self1, ds, 1);
    } else {
      ok = false;
    }
  }
  if (scalar) aclDestroyScalar(scalar);
  if (scalar_half) aclDestroyScalar(scalar_half);
  if (self1) aclDestroyTensor(self1); if (a1) aclDestroyTensor(a1); if (b1) aclDestroyTensor(b1);
  if (a_half) aclDestroyTensor(a_half); if (a2) aclDestroyTensor(a2); if (a_noncontig) aclDestroyTensor(a_noncontig);
  if (out1) aclDestroyTensor(out1); if (out2) aclDestroyTensor(out2);
  aclrtFree(ds); aclrtFree(da); aclrtFree(db); aclrtFree(dout);
  return ok;
}

static uint32_t Next(uint32_t* state) { *state = *state * 1664525u + 1013904223u; return *state; }
static float Normal(uint32_t* state) { return static_cast<float>(static_cast<int32_t>(Next(state) % 2000001) - 1000000) * 0.00073f; }

int main() {
  if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(0) != ACL_SUCCESS) return 1;
  aclrtStream stream = nullptr;
  if (aclrtCreateStream(&stream) != ACL_SUCCESS) return 1;
  bool ok = true;
  const float alpha = FromBits(0x3f7ffff4), q1 = FromBits(0xc1c82b67), q2 = FromBits(0x41482b70);
  ok &= Run("captured_scalar", {}, {q1}, {alpha}, {q2}, 2.0f, stream);
  ok &= Run("captured_batch", {4096}, std::vector<float>(4096, q1), std::vector<float>(4096, alpha), std::vector<float>(4096, q2), 2.0f, stream);
  uint32_t state = 17; std::vector<float> self(16384), a(16384), b(16384);
  for (size_t i = 0; i < a.size(); ++i) { a[i] = Normal(&state); b[i] = Normal(&state); self[i] = -ScaleRN(a[i], b[i]); }
  ok &= Run("cancellation_random", {16384}, self, a, b, 1.0f, stream);
  for (float value : {2.0f, 1.0f, -1.0f, 0.5f, 1.25f}) ok &= Run("scale_random", {16384}, self, a, b, value, stream);
  for (size_t i = 0; i < a.size(); ++i) self[i] = Normal(&state);
  ok &= Run("ordinary_random", {16384}, self, a, b, 1.0f, stream);
  std::vector<float> zself, za, zb;
  for (float x : {0.0f, -0.0f}) for (float y : {-3.0f, 3.0f}) for (float z : {-2.0f, 2.0f}) { zself.push_back(x); za.push_back(y); zb.push_back(z); }
  ok &= Run("signed_zero", {static_cast<int64_t>(zself.size())}, zself, za, zb, 1.0f, stream);
  const float den = std::numeric_limits<float>::denorm_min(), min = std::numeric_limits<float>::min();
  ok &= Run("subnormal", {4}, {den, -den, min, -min}, {min, min, den, -den}, {1.0f, -1.0f, 1.0f, 1.0f}, 1.0f, stream);
  const float inf = std::numeric_limits<float>::infinity(), max = std::numeric_limits<float>::max(), nan = std::numeric_limits<float>::quiet_NaN();
  ok &= Run("overflow_inf", {4}, {max, -max, inf, -inf}, {max, max, inf, inf}, {2.0f, -2.0f, 0.0f, 0.0f}, 1.0f, stream);
  ok &= Run("finite_product_overflow", {2}, {1.0f, -1.0f}, {max * 0.5f, -max * 0.5f}, {3.0f, 3.0f}, 1.0f, stream);
  ok &= Run("nan", {3}, {nan, 1.0f, 1.0f}, {1.0f, nan, 1.0f}, {1.0f, 1.0f, nan}, 1.0f, stream);
  ok &= Run("shape_1", {1}, {q1}, {alpha}, {q2}, 2.0f, stream);
  ok &= Run("shape_2", {2}, {q1, q1}, {alpha, alpha}, {q2, q2}, 2.0f, stream);
  ok &= Run("shape_2x3x4", {2, 3, 4}, std::vector<float>(24, q1), std::vector<float>(24, alpha), std::vector<float>(24, q2), 2.0f, stream);
  // This shape spans three tiles per core at block_dim=32.  Reusing the same
  // ACLNN executor makes any missing MTE3->MTE2 buffer-reuse dependency a
  // run-to-run bit mismatch rather than a one-off launch failure.
  const std::vector<int64_t> tiled_shape{2048, 18, 3, 3};
  const size_t tiled_elements = static_cast<size_t>(Numel(tiled_shape));
  std::vector<float> tiled_self(tiled_elements), tiled_a(tiled_elements), tiled_b(tiled_elements);
  for (size_t i = 0; i < tiled_elements; ++i) {
    tiled_a[i] = Normal(&state);
    tiled_b[i] = Normal(&state);
    tiled_self[i] = Normal(&state);
  }
  ok &= Run("multitile_repeat", tiled_shape, tiled_self, tiled_a, tiled_b,
            1.0f, stream, 20);
  ok &= RunInvalidContractTests(stream);
  aclrtDestroyStream(stream);
  aclrtResetDevice(0); aclFinalize();
  return ok ? 0 : 2;
}
