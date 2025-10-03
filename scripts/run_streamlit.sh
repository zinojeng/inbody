#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE' >&2
Usage: run_streamlit.sh [streamlit-options]

這支腳本會：
  1. 建立（或啟用）專案根目錄下的 .venv 虛擬環境。
  2. 安裝 requirements.txt 內的套件。
  3. 啟動 Streamlit，預設載入 streamlit_app.py。

範例：
  ./scripts/run_streamlit.sh
  ./scripts/run_streamlit.sh --server.port=8686
  PORT=9000 ./scripts/run_streamlit.sh --server.address=127.0.0.1
USAGE
}

if [[ ${1-} == "--help" ]]; then
  usage
  exit 0
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
venv_dir="${project_root}/.venv"

if [[ ! -d "${venv_dir}" ]]; then
  python3 -m venv "${venv_dir}"
fi

# shellcheck source=/dev/null
source "${venv_dir}/bin/activate"
python -m pip install --upgrade pip >/dev/null
python -m pip install -r "${project_root}/requirements.txt"

streamlit_args=("run" "${project_root}/streamlit_app.py")

if [[ -n "${PORT-}" ]]; then
  streamlit_args+=("--server.port=${PORT}")
else
  streamlit_args+=("--server.port=8501")
fi

has_address_flag=false
for arg in "$@"; do
  if [[ "${arg}" == --server.address=* ]]; then
    has_address_flag=true
    break
  fi
  if [[ "${arg}" == "--server.address" ]]; then
    has_address_flag=true
    break
  fi
done

if [[ "${has_address_flag}" == false ]]; then
  streamlit_args+=("--server.address=0.0.0.0")
fi

streamlit_args+=("$@")

exec streamlit "${streamlit_args[@]}"
