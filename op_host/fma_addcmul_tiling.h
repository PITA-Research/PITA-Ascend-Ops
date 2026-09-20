#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(FmaAddcmulTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, numElements);
  TILING_DATA_FIELD_DEF(float, value);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FmaAddcmul, FmaAddcmulTilingData)
}  // namespace optiling
