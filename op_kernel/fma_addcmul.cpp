#include "kernel_operator.h"

// Computes out = fma(RN_f32(value * tensor1), tensor2, self).  The first
// scale is deliberately a separate vector instruction.  910B vector MAC is
// not final-rounding FMA, so the kernel reconstructs that final rounding from
// product and sum residuals without any intermediate global buffers.
class KernelFmaAddcmul {
public:
    static constexpr uint32_t kTileElements = 4096;

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR tensor1, GM_ADDR tensor2,
                                GM_ADDR value, GM_ADDR out, uint32_t num_elements,
                                AscendC::TPipe* pipe)
    {
        self_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(self));
        tensor1_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tensor1));
        tensor2_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tensor2));
        value_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(value));
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(out));
        numElements_ = num_elements;
        pipe_ = pipe;
        const uint32_t tile_bytes = kTileElements * sizeof(float);
        pipe_->InitBuffer(selfBuf_, tile_bytes);
        pipe_->InitBuffer(tensor1Buf_, tile_bytes);
        pipe_->InitBuffer(tensor2Buf_, tile_bytes);
        pipe_->InitBuffer(productBuf_, tile_bytes);
        pipe_->InitBuffer(aHighBuf_, tile_bytes);
        pipe_->InitBuffer(bHighBuf_, tile_bytes);
        pipe_->InitBuffer(tempBuf_, tile_bytes);
        pipe_->InitBuffer(baseBuf_, tile_bytes);
        pipe_->InitBuffer(specialBuf_, tile_bytes);
    }

    __aicore__ inline void Process()
    {
        const float value = value_.GetValue(0);
        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t core_count = AscendC::GetBlockNum();
        for (uint32_t offset = core * kTileElements; offset < numElements_;
             offset += core_count * kTileElements) {
            const uint32_t count = (numElements_ - offset < kTileElements)
                ? numElements_ - offset : kTileElements;
            CopyIn(offset, count);
            Compute(count, value);
            CopyOut(offset, count);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count)
    {
        const AscendC::DataCopyExtParams params(1, count * sizeof(float), 0, 0, 0);
        const AscendC::DataCopyPadExtParams<float> pad(false, 0, 0, 0.0f);
        AscendC::DataCopyPad(selfBuf_.Get<float>(), self_[offset], params, pad);
        AscendC::DataCopyPad(tensor1Buf_.Get<float>(), tensor1_[offset], params, pad);
        AscendC::DataCopyPad(tensor2Buf_.Get<float>(), tensor2_[offset], params, pad);
        event_t event = static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event);
    }

    __aicore__ inline void Compute(uint32_t count, float value)
    {
        const auto self_local = selfBuf_.Get<float>();
        const auto tensor1_local = tensor1Buf_.Get<float>();
        const auto tensor2_local = tensor2Buf_.Get<float>();
        const auto product = productBuf_.Get<float>();
        const auto a_high = aHighBuf_.Get<float>();
        const auto b_high = bHighBuf_.Get<float>();
        const auto temp = tempBuf_.Get<float>();
        const auto base = baseBuf_.Get<float>();
        const auto special = specialBuf_.Get<float>();
        AscendC::Muls(tensor1_local, tensor1_local, value, count);
        // The 910B vector MAC has non-fused rounding.  Recover the exact
        // product residual with Dekker splitting entirely inside this one
        // kernel, then add the two-product and two-sum residuals.
        AscendC::Mul(product, tensor1_local, tensor2_local, count);
        constexpr float kMinNormal = 1.17549435e-38f;
        constexpr float kMaxFinite = 3.40282347e+38f;
        for (uint32_t i = 0; i < count; ++i) {
            const float x = tensor1_local.GetValue(i);
            const float y = tensor2_local.GetValue(i);
            const float z = self_local.GetValue(i);
            const float r = product.GetValue(i);
            const float ax = x < 0.0f ? -x : x;
            const float ay = y < 0.0f ? -y : y;
            const float az = z < 0.0f ? -z : z;
            const float ar = r < 0.0f ? -r : r;
            special.SetValue(i, (x != x || y != y || z != z || ax >= kMaxFinite ||
                ay >= kMaxFinite || az >= kMaxFinite || r != r || ar >= kMaxFinite ||
                (ax != 0.0f && ax < kMinNormal) ||
                (ay != 0.0f && ay < kMinNormal) ||
                (az != 0.0f && az < kMinNormal)) ? 1.0f : 0.0f);
        }
        AscendC::Muls(a_high, tensor1_local, 4097.0f, count);
        AscendC::Sub(temp, a_high, tensor1_local, count);
        AscendC::Sub(a_high, a_high, temp, count);
        AscendC::Muls(b_high, tensor2_local, 4097.0f, count);
        AscendC::Sub(temp, b_high, tensor2_local, count);
        AscendC::Sub(b_high, b_high, temp, count);
        AscendC::Mul(temp, a_high, b_high, count);
        AscendC::Sub(temp, temp, product, count);
        AscendC::Sub(tensor2_local, tensor2_local, b_high, count);
        AscendC::Mul(base, a_high, tensor2_local, count);
        AscendC::Add(temp, temp, base, count);
        AscendC::Sub(tensor1_local, tensor1_local, a_high, count);
        AscendC::Mul(base, tensor1_local, b_high, count);
        AscendC::Add(temp, temp, base, count);
        AscendC::Mul(base, tensor1_local, tensor2_local, count);
        AscendC::Add(temp, temp, base, count);
        // base = RN(product + self), b_high = sum residual.
        AscendC::Add(base, product, self_local, count);
        AscendC::Sub(a_high, base, product, count);
        AscendC::Sub(b_high, base, a_high, count);
        AscendC::Sub(b_high, product, b_high, count);
        AscendC::Sub(tensor2_local, self_local, a_high, count);
        AscendC::Add(b_high, b_high, tensor2_local, count);
        AscendC::Add(temp, temp, b_high, count);
        AscendC::Add(tensor1_local, base, temp, count);
        // Dekker is valid for finite normal operands.  Preserve IEEE special
        // classifications (and subnormal behavior) with the base operation.
        for (uint32_t i = 0; i < count; ++i) {
            if (special.GetValue(i) != 0.0f) {
                tensor1_local.SetValue(i, base.GetValue(i));
            }
        }
        event_t event = static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::V_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count)
    {
        const AscendC::DataCopyExtParams params(1, count * sizeof(float), 0, 0, 0);
        AscendC::DataCopyPad(out_[offset], tensor1Buf_.Get<float>(), params);
        // All tiles share these local buffers.  DataCopyPad above is queued on
        // MTE3, while the next iteration starts a new MTE2 copy into the same
        // buffers.  V->MTE3 only guarantees that vector computation finished;
        // it does not stop MTE2 from overwriting data that MTE3 is still
        // draining.  Make the buffer reuse explicit before the next CopyIn.
        event_t event = static_cast<event_t>(
            pipe_->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
    }

    AscendC::TPipe* pipe_ = nullptr;
    AscendC::GlobalTensor<float> self_, tensor1_, tensor2_, value_, out_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> selfBuf_, tensor1Buf_, tensor2Buf_, productBuf_,
        aHighBuf_, bHighBuf_, tempBuf_, baseBuf_, specialBuf_;
    uint32_t numElements_ = 0;
};

extern "C" __global__ __aicore__ void fma_addcmul(
    GM_ADDR self, GM_ADDR tensor1, GM_ADDR tensor2, GM_ADDR value, GM_ADDR out,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    AscendC::TPipe pipe;
    KernelFmaAddcmul op;
    op.Init(self, tensor1, tensor2, value, out, tiling_data.numElements, &pipe);
    op.Process();
}
