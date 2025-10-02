# Agent Overview

The agent now operates as a two-part pipeline：

- **Part 1 – 基本數據轉檔處理**：由 `main.py` 搭配 `inbody_processing.py` 完成，將原始 InBody CSV 轉為標準化摘要（CSV / JSON / Markdown）。
- **Part 2 – LLM 分析**：由 `final_analysis.py` 負責，消化 Part 1 產出的摘要並生成進一步的語意建議或報告。

## Architecture

```
raw CSV ──▶ main.py ──▶ inbody_summary.{csv,json,md} ──▶ final_analysis.py ──▶ LLM insights
```

### Part 1：Basic Transformation (`main.py` + `inbody_processing.py`)
- 自動判斷常見編碼（`utf-8-sig` → `big5` → `cp950`）。
- `inbody_processing.py` 內的 `extract_core_metrics()` 將欄位對應到統一的 metric keys。
- `process_inbody_file()` 會同時輸出三種格式：
  - `inbody_summary.csv`
  - `inbody_summary.json`
  - `inbody_report.md`
- To run: `python main.py data/<input>.csv [--output-dir <dir>] [--encoding utf-8 big5 ...]`

### Part 2：LLM Analysis (`final_analysis.py`)
- 輸入來源為 Part 1 的 CSV 或 JSON 摘要，並使用 `MetricStore` 進行欄位模糊匹配與數值解析。
- 預設會呼叫 OpenAI GPT-4.1 系列（可自行指定 `--model`；預設 `gpt-4.1`，並自動 fallback 至 `gpt-4o-mini`）產生完整 Markdown 報告。若啟用 LLM，內建規則段落會被模型生成的內容取代。
- `final_analysis.py` 會將 `reference/` 目錄下的文檔（如 `InBody報告深度文獻分析.md`）作為 RAG 來源，挑選相關節錄供 GPT 參考，並將個人指標（BMI、內臟脂肪面積、ECW/TBW、相位角等）一併注入提示。
- 可透過 `./scripts/run_inbody_pipeline.sh <csv>` 一次完成轉換與報告產出，或手動執行 `python final_analysis.py inbody_clean/inbody_summary.json --model gpt-5.0`。

## Extensibility Notes
- 想支援新的欄位：在 `inbody_processing.extract_core_metrics()` 或 `final_analysis.KEYS` 加入對應關鍵字。
- 想輸出更多檔案型態：擴充 `write_outputs()` 或在 Part 2 讀取 JSON 後另存。
- 想串接不同 LLM：在 `final_analysis.py` 中替換或包裝現有的推論函式即可。

## 操作流程摘要
1. 放置 InBody 原始 CSV 至專案工作目錄（或指定絕對路徑）。
2. 執行 `python main.py <input.csv>` 產生標準化摘要資料夾（預設 `<input parent>/inbody_clean/`）。
3. 將摘要結果餵給 `python final_analysis.py <summary.json>` 進行進階分析。
4. 檢視 Markdown 報告與 LLM 建議，必要時再行調整或覆寫。
