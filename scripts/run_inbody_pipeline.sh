#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE' >&2
Usage: run_inbody_pipeline.sh CSV_PATH [options]

Options:
  --summary-only           只執行資料轉換，不產出最終報告
  --final-only             假設摘要檔已存在，只產出報告
  --no-gpt                 產出報告時停用 GPT 洞察
  --encoding ENC [...]      傳遞給 main.py 的編碼順序（僅轉檔階段）
  --output-dir DIR         指定轉檔輸出目錄（預設為 <csv 同層>/inbody_clean）
  --model MODEL            GPT 模型（預設 gpt-4.1）
  --temperature VALUE      GPT 溫度（預設 0.3；若使用 GPT-5 可設為 -1 以停用）
  --reasoning-effort LEVEL GPT-5 推理強度：minimal | low | medium | high
  --verbosity LEVEL        GPT-5 輸出詳盡度：low | medium | high
  --max-output-tokens N    GPT-5 輸出 token 上限
  --help                   顯示說明

範例：
  # 完整流程：解析 InBody CSV 並產出 GPT 報告
  ./scripts/run_inbody_pipeline.sh data/e122508493_20250923160358.csv

  # 只進行資料轉換
  ./scripts/run_inbody_pipeline.sh data/e122508493_20250923160358.csv --summary-only

  # 只產出報告（假設摘要已存在）
  ./scripts/run_inbody_pipeline.sh data/e122508493_20250923160358.csv --final-only

資料轉換完成後會自動尋找 <output-dir>/inbody_summary.json 作為報告輸入。
USAGE
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

CSV_PATH=""
SUMMARY_ONLY=false
FINAL_ONLY=false
NO_GPT=false
MODEL="gpt-4.1"
TEMPERATURE="0.3"
REASONING_EFFORT=""
VERBOSITY=""
MAX_OUTPUT_TOKENS=""
OUTPUT_DIR=""
ENCODING_ARGS=()

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
venv_dir="${project_root}/.venv"

parse_args() {
  local positional_consumed=false
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --help)
        usage
        exit 0
        ;;
      --summary-only)
        SUMMARY_ONLY=true
        shift
        ;;
      --final-only)
        FINAL_ONLY=true
        shift
        ;;
      --no-gpt)
        NO_GPT=true
        shift
        ;;
      --model)
        MODEL="$2"
        shift 2
        ;;
      --temperature)
        TEMPERATURE="$2"
        shift 2
        ;;
      --reasoning-effort)
        REASONING_EFFORT="$2"
        shift 2
        ;;
      --verbosity)
        VERBOSITY="$2"
        shift 2
        ;;
      --max-output-tokens)
        MAX_OUTPUT_TOKENS="$2"
        shift 2
        ;;
      --output-dir)
        OUTPUT_DIR="$2"
        shift 2
        ;;
      --encoding)
        shift
        while [[ $# -gt 0 && $1 != --* ]]; do
          ENCODING_ARGS+=("$1")
          shift
        done
        ;;
      *)
        if [[ "$positional_consumed" == false ]]; then
          CSV_PATH="$1"
          positional_consumed=true
          shift
        else
          echo "Unrecognized argument: $1" >&2
          usage
          exit 1
        fi
        ;;
    esac
  done
}

parse_args "$@"

if [[ -z "${CSV_PATH}" && ${FINAL_ONLY} == false ]]; then
  echo "請指定 InBody CSV 路徑" >&2
  usage
  exit 1
fi

if [[ ! -d "${venv_dir}" ]]; then
  python3 -m venv "${venv_dir}"
fi

# shellcheck source=/dev/null
source "${venv_dir}/bin/activate"
python -m pip install --upgrade pip >/dev/null
python -m pip install -r "${project_root}/requirements.txt"

if [[ ${FINAL_ONLY} == false ]]; then
  if [[ ! -f "${CSV_PATH}" ]]; then
    echo "找不到 CSV 檔案：${CSV_PATH}" >&2
    exit 1
  fi
  echo "[1/2] 解析 InBody CSV -> 摘要檔"
  python "${project_root}/main.py" "${CSV_PATH}" \
    ${OUTPUT_DIR:+--output-dir "${OUTPUT_DIR}"} \
    ${ENCODING_ARGS:+--encoding "${ENCODING_ARGS[@]}"}
fi

if [[ ${SUMMARY_ONLY} == true ]]; then
  echo "已完成資料轉換 (略過報告產生)"
  exit 0
fi

summary_dir=""
if [[ -n "${OUTPUT_DIR}" ]]; then
  summary_dir="${OUTPUT_DIR}"
else
  if [[ -n "${CSV_PATH}" ]]; then
    summary_dir="$(cd "$(dirname "${CSV_PATH}")" && pwd)/inbody_clean"
  fi
  if [[ -z "${summary_dir}" ]]; then
    summary_dir="${project_root}/data/inbody_clean"
  fi
fi

summary_json="${summary_dir}/inbody_summary.json"
if [[ ! -f "${summary_json}" ]]; then
  echo "找不到摘要檔案：${summary_json}" >&2
  exit 1
fi

final_args=(--input "${summary_json}" --model "${MODEL}" --temperature "${TEMPERATURE}")
if [[ ${NO_GPT} == true ]]; then
  final_args+=(--no-gpt)
fi
if [[ -n "${REASONING_EFFORT}" ]]; then
  final_args+=(--reasoning-effort "${REASONING_EFFORT}")
fi
if [[ -n "${VERBOSITY}" ]]; then
  final_args+=(--verbosity "${VERBOSITY}")
fi
if [[ -n "${MAX_OUTPUT_TOKENS}" ]]; then
  final_args+=(--max-output-tokens "${MAX_OUTPUT_TOKENS}")
fi

status="enabled"
if [[ ${NO_GPT} == true ]]; then
  status="disabled"
fi
echo "[2/2] 產出最終報告 (GPT=${status})"
python "${project_root}/final_analysis.py" "${final_args[@]}"
