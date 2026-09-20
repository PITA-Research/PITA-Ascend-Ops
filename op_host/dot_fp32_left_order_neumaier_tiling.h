#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DotFp32LeftOrderNeumaierTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batch);
  TILING_DATA_FIELD_DEF(uint32_t, dotLen);
  TILING_DATA_FIELD_DEF(uint32_t, lhsRowStride);
  TILING_DATA_FIELD_DEF(uint32_t, rhsRowStride);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DotFp32LeftOrderNeumaier, DotFp32LeftOrderNeumaierTilingData)
}
