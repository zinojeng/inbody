"""Streamlit UI for generating InBody summary reports with optional GPT insights."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import streamlit as st

from final_analysis import run as generate_final_report
from inbody_processing import process_inbody_file


st.set_page_config(page_title="InBody Report Builder", layout="wide")


def _store_api_key(api_key: str) -> None:
    """Persist the provided API key in the current process environment."""
    if not api_key:
        return
    os.environ["OPENAI_API_KEY"] = api_key.strip()


def _format_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _load_summary(summary_path: Path) -> Dict[str, object]:
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _show_summary_table(summary: Dict[str, object]) -> None:
    rows = [(key, "" if value is None else str(value)) for key, value in summary.items()]
    df = pd.DataFrame(rows, columns=["項目", "數值"])
    st.dataframe(df, width='stretch', hide_index=True)


st.title("InBody 報告產生器")
st.write(
    "上傳原始 InBody CSV 檔案，提供 OpenAI API Key（可選），即可產出個人化 Markdown 報告。"
)

with st.sidebar:
    st.header("GPT 設定")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Key 僅保留於本地執行環境。",
        key="sidebar_api_key",
    )
    model = st.selectbox(
        "模型選擇",
        options=["gpt-5", "gpt-4.1", "gpt-4o-mini"],
        help="若主模型不可用會自動 fallback 至預設備援模型。",
        key="sidebar_model",
    )
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        key="sidebar_temperature",
    )
    st.caption("未提供 API Key 時，系統會改用內建規則式摘要。")


def _reset_outputs() -> None:
    st.session_state.pop("report_text", None)
    st.session_state.pop("report_generated", None)
    st.session_state.pop("summary_data", None)


with st.form("inbody_form"):
    uploaded_file = st.file_uploader(
        "上傳 InBody CSV",
        type=["csv"],
        accept_multiple_files=False,
        key="form_csv_uploader",
    )
    use_gpt = st.checkbox("啟用 GPT 強化分析", value=True, key="form_use_gpt")
    submitted = st.form_submit_button("產出報告", width='stretch')

    if submitted:
        _reset_outputs()
        if uploaded_file is None:
            st.warning("請先選擇 CSV 檔案。")
        else:
            with st.spinner("報告產生中，請稍候..."):
                try:
                    with tempfile.TemporaryDirectory() as temp_root:
                        tmp_dir = Path(temp_root)
                        raw_path = tmp_dir / "upload.csv"
                        raw_path.write_bytes(uploaded_file.getvalue())

                        output_dir = tmp_dir / "clean"
                        outputs = process_inbody_file(raw_path, output_dir)

                        _store_api_key(api_key)
                        has_api_key = bool(api_key or os.getenv("OPENAI_API_KEY"))
                        effective_use_gpt = use_gpt and has_api_key
                        if use_gpt and not effective_use_gpt:
                            st.info("未提供 API Key，改用內建規則式摘要。")

                        report_path = generate_final_report(
                            outputs["json"],
                            output_path=tmp_dir / "inbody_final_report.md",
                            use_gpt=effective_use_gpt,
                            model=model,
                            temperature=temperature,
                        )

                        report_text = report_path.read_text(encoding="utf-8")
                        summary = _load_summary(outputs["json"])

                    st.session_state["report_text"] = report_text
                    st.session_state["summary_data"] = summary
                    st.session_state["report_generated"] = _format_timestamp()
                    st.success("報告產出完成！")
                except Exception as exc:  # noqa: BLE001 - display friendly error
                    st.error(f"產生報告時發生錯誤：{exc}")


report_text: Optional[str] = st.session_state.get("report_text")
summary_data: Optional[Dict[str, object]] = st.session_state.get("summary_data")
report_generated: Optional[str] = st.session_state.get("report_generated")

if report_text and summary_data:
    st.subheader("摘要指標")
    _show_summary_table(summary_data)

    st.subheader("最終報告")
    st.download_button(
        "下載 Markdown 報告",
        data=report_text,
        file_name="inbody_final_report.md",
        mime="text/markdown",
        width='stretch',
    )
    st.markdown(report_text)
    if report_generated:
        st.caption(f"產出時間：{report_generated}")
else:
    st.info("產出的報告與指標會顯示在此區塊。")
