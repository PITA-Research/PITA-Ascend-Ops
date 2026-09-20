#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CANN_HOME="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-8.5.0}"
INSTALL_DIR="${ROOT_DIR}/custom_opp_install"
TMP_DIR="$(mktemp -d /tmp/optry_fma_addcmul_benchmark.XXXXXX)"
trap 'rm -rf "${TMP_DIR}"' EXIT

set +u
source "${CANN_HOME}/set_env.sh"
source "${INSTALL_DIR}/vendors/optry/bin/set_env.bash"
set -u

g++ -std=c++17 -O2 "${ROOT_DIR}/tests/benchmark/fma_addcmul_benchmark.cc" \
  -I"${CANN_HOME}/include" -I"${INSTALL_DIR}/vendors/optry/op_api/include" \
  -L"${CANN_HOME}/lib64" -L"${INSTALL_DIR}/vendors/optry/op_api/lib" \
  -lcust_opapi -lopapi_math -lascendcl -lnnopbase \
  -Wl,-rpath,"${CANN_HOME}/lib64" -Wl,-rpath,"${INSTALL_DIR}/vendors/optry/op_api/lib" \
  -o "${TMP_DIR}/fma_addcmul_benchmark"
"${TMP_DIR}/fma_addcmul_benchmark"
