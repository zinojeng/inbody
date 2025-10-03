# InBody 報表擷取工具

本程式用來從 InBody CSV 報表自動擷取常用指標，並輸出成結構化檔案與文字摘要，方便後續統計或製作報告。

## 互動式網頁介面（Streamlit）

想以更友善的方式產出報告，可啟動內建的 Streamlit 應用：

1. 安裝依賴：`pip install -r requirements.txt`
2. 啟動服務：`streamlit run streamlit_app.py`
3. 瀏覽器開啟預設網址（通常為 `http://localhost:8501`）。
4. 在側邊欄輸入 OpenAI API Key（選填）、選擇模型。
5. 上傳 InBody 原始 CSV 後按下「產出報告」，即可下載 Markdown 並線上預覽。

未提供 API Key 時，系統會自動改用內建的規則式摘要；提供金鑰則會呼叫 GPT 產出完整個人化報告。

## 如何使用（重點說明）

1. 將 InBody 匯出的 CSV 放到 `data/`（或任何你喜歡的位置）。
2. 直接執行 `python main.py data/你的檔案.csv`，或使用 `./scripts/run_inbody_pipeline.sh data/你的檔案.csv`（會自動建立虛擬環境、安裝套件並執行完整 pipeline）。
3. 程式會依序以 `utf-8-sig → big5 → cp950` 嘗試讀取 CSV，確保不同編碼下都能正常解析。
4. 解析時會用關鍵字比對欄位名稱（刻意略過 Normal Range 的上下限欄位），自動抓取下列資訊：
   - 基本資料：姓名、ID、性別、年齡、身高、測試時間
   - 身體組成：體重、SMM、BFM、PBF、TBW、ICW、ECW、蛋白質、礦物質、BMI、BMR、VFA、ECW/TBW、SMI、InBody 分數
   - 體重控制：目標體重、建議減脂、建議增肌
   - 部位分析：左右上肢、左右下肢、軀幹的 Lean/Fat（kg）
5. 成功解析後會輸出三個檔案（預設儲存在與輸入 CSV 相同的資料夾內）：
   - `inbody_summary.csv`：所有擷取到的指標，格式為兩欄（項目/數值）。
   - `inbody_summary.json`：與 `csv` 同內容的 JSON 版，方便與其他系統串接。
   - `inbody_report.md`：簡潔的人類可讀摘要，可轉成 PDF 或貼到 Word。

## 自訂與擴充

- 若要擴充擷取的 InBody 欄位，可在 `extract_core_metrics()` 的關鍵字清單加入新項目。
- 若需要額外輸出格式，可仿照既有流程，在擷取完成後新增寫檔邏輯。
- 程式的關鍵函式集中於 `parse_inbody_csv` 主流程，複製這段程式碼並調整輸入路徑，就能在你的環境重複使用。

## 虛擬環境快速啟動

- 使用 `./scripts/run_inbody_pipeline.sh data/你的檔案.csv` 會自動建立 `.venv/`、安裝 `requirements.txt`，先執行 CSV 轉換，再視需求產出報告。
- 若只想轉出摘要，可加上 `--summary-only`；只想生成報告（假設摘要已存在）則加 `--final-only`。
- 需自訂輸出位置或編碼，可分別使用 `--output-dir` 與 `--encoding` 參數。

## 執行環境

- Python 3.9 以上（建議 3.10+）。
- 必要套件請參考 `requirements.txt`（內含 `pandas`、`streamlit`、`openai` 與 `python-dotenv`）。

## 疑難排解

- **CSV 讀取失敗**：確認檔案確實存在，並檢查是否為支援的編碼。
- **欄位抓取不到資料**：請檢查 CSV 欄位名稱是否與關鍵字相符，可依需要增修關鍵字對應表。

## 最終建議報告

- 完成資料擷取後，可執行 `python final_analysis.py` 產出 `inbody_final_report.md`（會自動搜尋 `data/inbody_clean/` 或目前資料夾中的摘要檔）。
- 需要指定輸入時，可傳入 `--input`，例如 `python final_analysis.py --input data/inbody_clean/inbody_summary.json`。
- 可使用 `--output` 參數自訂輸出檔案路徑，輸出為 Markdown 格式，方便後續轉成 PDF 或分享。

### LLM 個人化分析（選用）

1. 建議先執行 `cp .env.example .env`，並將 `OPENAI_API_KEY` 改成你自己的 OpenAI API 金鑰（或設定在系統環境變數）。
2. 安裝相依套件：`pip install -r requirements.txt`（內含 `openai` 與 `python-dotenv`）。
3. 預設會由 OpenAI GPT-4.1 家族產生個人化洞察，可透過下列指令執行：
   ```bash
   python final_analysis.py \
     --input data/inbody_clean/inbody_summary.json \
     --model gpt-4.1 \
     --temperature 0.3
   ```
4. LLM 會參考 `reference/` 目錄下的文檔（例如 `InBody報告深度文獻分析.md`）作為 RAG 來源，自動生成完整報告；可用 `--reference` 指定其他檔案或資料夾。
5. 若切換到 GPT-5 系列，可加上 `--model gpt-5` 並搭配 `--reasoning-effort`, `--verbosity`, `--max-output-tokens` 控制輸出；溫度請設為 `-1`（或省略）。
6. 若未設置 API 金鑰或網路環境無法連線，程式會回退到內建的規則式分析並顯示錯誤訊息；如需完全停用 LLM，可加上 `--no-gpt`。

> 快速腳本：`./scripts/run_inbody_pipeline.sh` 會自動建立虛擬環境、安裝依賴並完成「CSV → 摘要 → 最終報告」整個流程。

> ⚠️ `.env` 已列入 `.gitignore`，請勿將含有金鑰的檔案提交至版本控制。

進階設定：如使用 `sk-proj-...` 專案金鑰，可在 `.env` 內補充 `OPENAI_PROJECT=proj_xxxxx`；若使用自訂端點，亦可設定 `OPENAI_BASE_URL`。

## 部署到 Zeabur

本專案已提供 Dockerfile，能直接在 Zeabur 建立容器服務：

1. 將專案推送到 Git 儲存庫，於 Zeabur 後台建立新服務並選擇「Dockerfile」作為部署來源。
2. Zeabur 會自動使用根目錄的 `Dockerfile` 建置映像檔，無需額外指定 Build/Run 指令。
3. 在服務的環境變數設定 `OPENAI_API_KEY`（必要時再加上 `OPENAI_BASE_URL`、`OPENAI_PROJECT` 等參數）。
4. Zeabur 預設會提供 `PORT` 環境變數，容器會以該埠啟動 Streamlit；若需自訂可以在面板覆寫。
5. 部署完成後，即可透過 Zeabur 產生的網址造訪 Streamlit 介面，進行 CSV 上傳與報告產出。

> 若想啟用持續部署，建議在 Zeabur 中啟用 auto deploy，或於 PR 合併後手動觸發重新部署。
