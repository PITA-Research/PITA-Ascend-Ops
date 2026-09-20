#include "kernel_operator.h"

class KernelDotFp32LeftOrderNeumaier {
public:
    __aicore__ inline void Init(GM_ADDR lhs, GM_ADDR rhs, GM_ADDR out, uint32_t batch,
                                uint32_t dot_len, uint32_t lhs_row_stride,
                                uint32_t rhs_row_stride, AscendC::TPipe* pipe)
    {
        lhs_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(lhs));
        rhs_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rhs));
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(out));
        batch_ = batch;
        dotLen_ = dot_len;
        lhsRowStride_ = lhs_row_stride;
        rhsRowStride_ = rhs_row_stride;
        pipe_ = pipe;
        const uint32_t row_bytes = dotLen_ * sizeof(float);
        pipe_->InitBuffer(lhsBuf_, row_bytes);
        pipe_->InitBuffer(rhsBuf_, row_bytes);
        pipe_->InitBuffer(outBuf_, sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint32_t core_id = AscendC::GetBlockIdx();
        const uint32_t core_num = AscendC::GetBlockNum();
        for (uint32_t row = core_id; row < batch_; row += core_num) {
            CopyIn(row);
            Compute();
            CopyOut(row);
        }
    }

private:
    __aicore__ inline float AbsFloat(float value)
    {
        return value < 0.0f ? -value : value;
    }

    __aicore__ inline float AddFloat(float lhs, float rhs)
    {
        volatile float result = lhs + rhs;
        return result;
    }

    __aicore__ inline float SubFloat(float lhs, float rhs)
    {
        volatile float result = lhs - rhs;
        return result;
    }

    __aicore__ inline float MulFloat(float lhs, float rhs)
    {
        volatile float result = lhs * rhs;
        return result;
    }

    __aicore__ inline void SyncMte2ToScalar()
    {
        event_t event_id = static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::MTE2_S));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(event_id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(event_id);
    }

    __aicore__ inline void SyncScalarToMte3()
    {
        event_t event_id = static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::S_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(event_id);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(event_id);
    }

    __aicore__ inline void SyncMte3ToScalar()
    {
        event_t event_id = static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::MTE3_S));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(event_id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(event_id);
    }

    __aicore__ inline void CopyIn(uint32_t row)
    {
        const uint32_t lhs_base = row * lhsRowStride_;
        const uint32_t rhs_base = row * rhsRowStride_;
        const AscendC::DataCopyExtParams copy_params(
            1, dotLen_ * static_cast<uint32_t>(sizeof(float)), 0, 0, 0);
        const AscendC::DataCopyPadExtParams<float> pad_params(false, 0, 0, 0.0f);
        AscendC::DataCopyPad(lhsBuf_.Get<float>(), lhs_[lhs_base], copy_params, pad_params);
        AscendC::DataCopyPad(rhsBuf_.Get<float>(), rhs_[rhs_base], copy_params, pad_params);
        SyncMte2ToScalar();
    }

    __aicore__ inline void Compute()
    {
        const AscendC::LocalTensor<float> lhs_local = lhsBuf_.Get<float>();
        const AscendC::LocalTensor<float> rhs_local = rhsBuf_.Get<float>();
        const AscendC::LocalTensor<float> out_local = outBuf_.Get<float>();
        float sum = 0.0f;
        float compensation = 0.0f;

        for (uint32_t k = 0; k < dotLen_; ++k) {
            const float product = MulFloat(lhs_local.GetValue(k), rhs_local.GetValue(k));
            const float next_sum = AddFloat(sum, product);
            float correction = 0.0f;
            if (AbsFloat(sum) >= AbsFloat(product)) {
                correction = AddFloat(SubFloat(sum, next_sum), product);
            } else {
                correction = AddFloat(SubFloat(product, next_sum), sum);
            }
            compensation = AddFloat(compensation, correction);
            sum = next_sum;
        }

        out_local.SetValue(0, AddFloat(sum, compensation));
        SyncScalarToMte3();
    }

    __aicore__ inline void CopyOut(uint32_t row)
    {
        const AscendC::DataCopyExtParams copy_params(1, sizeof(float), 0, 0, 0);
        AscendC::DataCopyPad(out_[row], outBuf_.Get<float>(), copy_params);
        SyncMte3ToScalar();
    }

private:
    AscendC::TPipe* pipe_ = nullptr;
    AscendC::GlobalTensor<float> lhs_;
    AscendC::GlobalTensor<float> rhs_;
    AscendC::GlobalTensor<float> out_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> lhsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rhsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outBuf_;
    uint32_t batch_ = 0;
    uint32_t dotLen_ = 0;
    uint32_t lhsRowStride_ = 0;
    uint32_t rhsRowStride_ = 0;
};

extern "C" __global__ __aicore__ void dot_fp32_left_order_neumaier(
    GM_ADDR lhs, GM_ADDR rhs, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    AscendC::TPipe pipe;
    KernelDotFp32LeftOrderNeumaier op;
    op.Init(lhs, rhs, out, tiling_data.batch, tiling_data.dotLen,
            tiling_data.lhsRowStride, tiling_data.rhsRowStride, &pipe);
    op.Process();
}
