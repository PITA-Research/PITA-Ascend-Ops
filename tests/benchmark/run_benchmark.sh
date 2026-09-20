#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_DIR="${ROOT_DIR}/tests/benchmark"
CANN_HOME="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-8.5.0}"
CUSTOM_OPP_ENV="${ROOT_DIR}/custom_opp_install/vendors/optry/bin/set_env.bash"

if [[ -f "${CANN_HOME}/set_env.sh" ]]; then
  # shellcheck disable=SC1091
  set +u
  source "${CANN_HOME}/set_env.sh"
  set -u
fi

if [[ -f "${CUSTOM_OPP_ENV}" ]]; then
  # shellcheck disable=SC1090
  set +u
  source "${CUSTOM_OPP_ENV}"
  set -u
fi

mkdir -p "${TEST_DIR}/bin" "${TEST_DIR}/results"

g++ -std=c++17 -O2 "${TEST_DIR}/benchmark_aclnn.cc" \
  -I"${CANN_HOME}/include" \
  -I"${ROOT_DIR}/build_out/autogen" \
  -L"${CANN_HOME}/lib64" \
  -L"${CANN_HOME}/aarch64-linux/lib64" \
  -L"${ROOT_DIR}/build_out/op_host" \
  -lcust_opapi -lopapi_math -lascendcl -lnnopbase \
  -Wl,-rpath,"${CANN_HOME}/lib64" \
  -Wl,-rpath,"${CANN_HOME}/aarch64-linux/lib64" \
  -Wl,-rpath,"${ROOT_DIR}/build_out/op_host" \
  -o "${TEST_DIR}/bin/benchmark_aclnn"

"${TEST_DIR}/bin/benchmark_aclnn" | tee "${TEST_DIR}/results/dot_perf_results.csv"
