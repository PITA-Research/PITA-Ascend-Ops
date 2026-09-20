#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_DIR="${ROOT_DIR}/tests/ut"
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

mkdir -p "${TEST_DIR}/bin"

g++ -std=c++17 -O2 "${TEST_DIR}/precision_aclnn.cc" \
  -I"${CANN_HOME}/include" \
  -I"${ROOT_DIR}/build_out/autogen" \
  -L"${CANN_HOME}/lib64" \
  -L"${ROOT_DIR}/build_out/op_host" \
  -lcust_opapi -lascendcl -lnnopbase \
  -Wl,-rpath,"${CANN_HOME}/lib64" \
  -Wl,-rpath,"${ROOT_DIR}/build_out/op_host" \
  -o "${TEST_DIR}/bin/precision_aclnn"

"${TEST_DIR}/bin/precision_aclnn"
