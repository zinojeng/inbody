"""Utilities for reading InBody CSV exports and emitting normalized summaries."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_ENCODINGS: Tuple[str, ...] = ("utf-8-sig", "big5", "cp950")


def try_read_csv(path: Path, encodings: Tuple[str, ...] = DEFAULT_ENCODINGS) -> pd.DataFrame:
    """Attempt to read the CSV using common encodings used by InBody exports."""
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:  # noqa: BLE001 - capture to retry with next encoding
            last_err = exc
    if last_err is None:
        raise FileNotFoundError(path)
    raise last_err


def _find_col(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    """Find the first column whose name contains any of the patterns (case-insensitive)."""
    candidates: List[str] = []
    normalized_patterns = [pattern.lower() for pattern in patterns]
    for column in df.columns:
        name = str(column)
        lower = name.lower()
        if not any(pattern in lower for pattern in normalized_patterns):
            continue

        # Strip numeric prefixes like "18. " for better comparisons.
        stripped = re.sub(r"^\s*\d+\.?\s*", "", lower)

        include = False
        for pattern in normalized_patterns:
            if pattern not in lower and pattern not in stripped:
                continue

            if "lower limit" in lower or "upper limit" in lower:
                if "limit" not in pattern:
                    continue
            if "control" in lower and "control" not in pattern:
                continue
            if "%" in name or "(%)" in name:
                if "%" not in pattern and "percent" not in pattern:
                    continue
            if "/" in name:
                if "/" not in pattern and "ratio" not in pattern:
                    continue
            include = True
            break
        if include:
            candidates.append(name)
    if not candidates:
        return None
    # Prefer shorter, cleaner column names after removing numbering.
    def quality(name: str) -> tuple[int, int, int]:
        lowered = name.lower()
        stripped_name = re.sub(r"^\s*\d+\.?\s*", "", name)
        stripped_lower = stripped_name.lower()
        penalty = 0
        if "lower limit" in lowered or "upper limit" in lowered:
            penalty += 20
        if "control" in lowered:
            penalty += 15
        if "(%)" in lowered or "%" in lowered:
            penalty += 10
        if "/" in lowered:
            penalty += 5
        specificity = 0
        for pattern in normalized_patterns:
            if pattern in stripped_lower or pattern in lowered:
                specificity = max(specificity, len(pattern))
        # Higher specificity should come first, so invert when sorting (negative)
        return (penalty, -specificity, len(stripped_name))

    candidates.sort(key=quality)
    return candidates[0]


def _safe_get(df: pd.DataFrame, column: Optional[str]) -> Any:
    if column is None or column not in df.columns:
        return None
    value = df.iloc[0][column]
    if isinstance(value, str) and value.strip() in {"", "-"}:
        return None
    return value


def extract_core_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    keys = {
        "Name": ["name"],
        "ID": [" id", "member id"],
        "TestDateTime": ["test date", "date / time", "date/time"],
        "Height_cm": ["height"],
        "Gender": ["gender"],
        "Age": ["age"],
        "Weight_kg": ["weight"],
        "TBW_kg": ["tbw", "total body water"],
        "ICW_kg": ["icw", "intracellular water"],
        "ECW_kg": ["ecw", "extracellular water"],
        "Protein_kg": ["protein"],
        "Minerals_kg": ["mineral", "minerals"],
        "SMM_kg": ["skeletal muscle mass", "smm"],
        "BFM_kg": ["body fat mass", "bfm"],
        "PBF_pct": ["percent body fat", "pbf", "body fat (%)"],
        "BMI": ["bmi"],
        "BMR_kcal": ["basal metabolic rate", "bmr"],
        "WHR": ["whr", "waist-hip ratio"],
        "VFA_cm2": ["visceral fat area", "vfa"],
        "ECW_TBW": ["ecw/tbw", "extracellular water ratio"],
        "SMI": ["smi", "skeletal muscle index"],
        "Score": ["inbody score", "score"],
        "TargetWeight_kg": ["target weight"],
        "FatControl_kg": ["fat control", "bfm control"],
        "MuscleControl_kg": ["muscle control", "ffm control"],
        "VFL_level": ["visceral fat level", "vfl"],
        "ObesityDegree_pct": ["obesity degree"],
        "BCM_kg": ["bcm", "body cell mass"],
        "TBW_FFM_pct": ["tbw/ffm"],
        "FFMI": ["ffmi", "fat free mass index"],
        "FMI": ["fmi", "fat mass index"],
    }
    region_map = {
        "RightArm_Lean_kg": ["right arm lean", "lean of right arm", "lean mass of right arm"],
        "LeftArm_Lean_kg": ["left arm lean", "lean of left arm", "lean mass of left arm"],
        "Trunk_Lean_kg": ["trunk lean", "lean of trunk", "lean mass of trunk"],
        "RightLeg_Lean_kg": ["right leg lean", "lean of right leg", "lean mass of right leg"],
        "LeftLeg_Lean_kg": ["left leg lean", "lean of left leg", "lean mass of left leg"],
        "RightArm_Fat_kg": ["right arm fat", "fat of right arm", "bfm of right arm"],
        "LeftArm_Fat_kg": ["left arm fat", "fat of left arm", "bfm of left arm"],
        "Trunk_Fat_kg": ["trunk fat", "fat of trunk", "bfm of trunk"],
        "RightLeg_Fat_kg": ["right leg fat", "fat of right leg", "bfm of right leg"],
        "LeftLeg_Fat_kg": ["left leg fat", "fat of left leg", "bfm of left leg"],
        "RightArm_Fat_pct": ["bfm% of right arm", "right arm fat %"],
        "LeftArm_Fat_pct": ["bfm% of left arm", "left arm fat %"],
        "Trunk_Fat_pct": ["bfm% of trunk", "trunk fat %"],
        "RightLeg_Fat_pct": ["bfm% of right leg", "right leg fat %"],
        "LeftLeg_Fat_pct": ["bfm% of left leg", "left leg fat %"],
        "RightArm_ECW_TBW": ["ecw/tbw of right arm"],
        "LeftArm_ECW_TBW": ["ecw/tbw of left arm"],
        "Trunk_ECW_TBW": ["ecw/tbw of trunk"],
        "RightLeg_ECW_TBW": ["ecw/tbw of right leg"],
        "LeftLeg_ECW_TBW": ["ecw/tbw of left leg"],
        "RightArm_TBW_kg": ["tbw of right arm"],
        "LeftArm_TBW_kg": ["tbw of left arm"],
        "Trunk_TBW_kg": ["tbw of trunk"],
        "RightLeg_TBW_kg": ["tbw of right leg"],
        "LeftLeg_TBW_kg": ["tbw of left leg"],
        "RightArm_PhaseAngle_deg": ["50khz-ra phase angle"],
        "LeftArm_PhaseAngle_deg": ["50khz-la phase angle"],
        "Trunk_PhaseAngle_deg": ["50khz-tr phase angle"],
        "RightLeg_PhaseAngle_deg": ["50khz-rl phase angle"],
        "LeftLeg_PhaseAngle_deg": ["50khz-ll phase angle"],
    }
    out: Dict[str, Any] = {}
    for key, patterns in keys.items():
        column = _find_col(df, patterns)
        out[key] = _safe_get(df, column)
    for key, patterns in region_map.items():
        column = _find_col(df, patterns)
        out[key] = _safe_get(df, column)
    return out


def _normalize_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _normalize_scalar(value.item())
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def normalize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {key: _normalize_scalar(value) for key, value in metrics.items()}


def generate_markdown_report(metrics: Dict[str, Any]) -> str:
    def fmt(value: Any, unit: str = "", digits: Optional[int] = None) -> str:
        if value is None:
            return "—"
        number: Any = value
        if digits is not None and isinstance(value, (int, float)):
            number = f"{float(value):.{digits}f}"
        elif digits is not None:
            try:
                number = f"{float(value):.{digits}f}"
            except (TypeError, ValueError):
                number = value
        text = str(number).strip()
        if not text:
            return "—"
        return f"{text}{unit}" if unit else text

    lines: List[str] = []
    lines.append("# InBody 正式報告\n")

    lines.append("## 基本資料")
    lines.append("| 項目 | 數值 |")
    lines.append("| --- | --- |")
    lines.append(f"| 姓名 | {fmt(metrics.get('Name'))} |")
    lines.append(f"| ID | {fmt(metrics.get('ID'))} |")
    lines.append(f"| 性別 | {fmt(metrics.get('Gender'))} |")
    lines.append(f"| 年齡 | {fmt(metrics.get('Age'))} |")
    lines.append(f"| 身高 | {fmt(metrics.get('Height_cm'), ' cm')} |")
    lines.append(f"| 測試時間 | {fmt(metrics.get('TestDateTime'))} |\n")

    lines.append("## 身體組成分析")
    lines.append("| 項目 | 數值 |")
    lines.append("| --- | --- |")
    lines.append(f"| 體重 | {fmt(metrics.get('Weight_kg'), ' kg', digits=1)} |")
    lines.append(f"| 骨骼肌量 (SMM) | {fmt(metrics.get('SMM_kg'), ' kg', digits=1)} |")
    lines.append(f"| 體脂肪量 (BFM) | {fmt(metrics.get('BFM_kg'), ' kg', digits=1)} |")
    lines.append(f"| 體脂率 (PBF) | {fmt(metrics.get('PBF_pct'), ' %', digits=1)} |")
    lines.append(f"| 體水分 (TBW) | {fmt(metrics.get('TBW_kg'), ' kg', digits=1)} |")
    lines.append(f"| 細胞內水分 (ICW) | {fmt(metrics.get('ICW_kg'), ' kg', digits=1)} |")
    lines.append(f"| 細胞外水分 (ECW) | {fmt(metrics.get('ECW_kg'), ' kg', digits=1)} |")
    lines.append(f"| 蛋白質 | {fmt(metrics.get('Protein_kg'), ' kg', digits=1)} |")
    lines.append(f"| 礦物質 | {fmt(metrics.get('Minerals_kg'), ' kg', digits=1)} |")
    lines.append(f"| BMI | {fmt(metrics.get('BMI'), digits=1)} |")
    lines.append(f"| BMR | {fmt(metrics.get('BMR_kcal'), ' kcal', digits=0)} |")
    lines.append(f"| 內臟脂肪面積 (VFA) | {fmt(metrics.get('VFA_cm2'), ' cm²', digits=1)} |")
    lines.append(f"| ECW/TBW | {fmt(metrics.get('ECW_TBW'), digits=3)} |")
    lines.append(f"| 腰臀比 (WHR) | {fmt(metrics.get('WHR'), digits=3)} |")
    lines.append(f"| SMI | {fmt(metrics.get('SMI'), digits=2)} |")
    lines.append(f"| InBody 分數 | {fmt(metrics.get('Score'), digits=0)} |\n")

    lines.append("## 體重控制建議")
    lines.append("| 項目 | 數值 |")
    lines.append("| --- | --- |")
    lines.append(f"| 目標體重 | {fmt(metrics.get('TargetWeight_kg'), ' kg', digits=1)} |")
    lines.append(f"| 建議減脂 | {fmt(metrics.get('FatControl_kg'), ' kg', digits=1)} |")
    lines.append(f"| 建議增肌 | {fmt(metrics.get('MuscleControl_kg'), ' kg', digits=1)} |\n")

    lines.append("## 部位肌肉/脂肪分析")
    lines.append("| 部位 | Lean (kg) | Fat (kg) |")
    lines.append("| --- | --- | --- |")
    lines.append(
        f"| 右手 | {fmt(metrics.get('RightArm_Lean_kg'), digits=1)} | {fmt(metrics.get('RightArm_Fat_kg'), digits=1)} |"
    )
    lines.append(
        f"| 左手 | {fmt(metrics.get('LeftArm_Lean_kg'), digits=1)} | {fmt(metrics.get('LeftArm_Fat_kg'), digits=1)} |"
    )
    lines.append(
        f"| 軀幹 | {fmt(metrics.get('Trunk_Lean_kg'), digits=1)} | {fmt(metrics.get('Trunk_Fat_kg'), digits=1)} |"
    )
    lines.append(
        f"| 右腿 | {fmt(metrics.get('RightLeg_Lean_kg'), digits=1)} | {fmt(metrics.get('RightLeg_Fat_kg'), digits=1)} |"
    )
    lines.append(
        f"| 左腿 | {fmt(metrics.get('LeftLeg_Lean_kg'), digits=1)} | {fmt(metrics.get('LeftLeg_Fat_kg'), digits=1)} |\n"
    )

    lines.append("## 其他指標與備註")
    lines.append("- 若需視覺化圖表，建議將以上數據餵入 `final_analysis.py` 或外部報表工具進行繪製。")
    lines.append("- 本報告由原始 CSV 自動轉換，缺漏值以 `—` 顯示。")

    return "\n".join(lines)


def write_outputs(metrics: Dict[str, Any], output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_metrics = normalize_metrics(metrics)
    summary_rows = [{"項目": key, "數值": value} for key, value in normalized_metrics.items()]
    summary_df = pd.DataFrame(summary_rows)

    summary_csv = output_dir / "inbody_summary.csv"
    summary_json = output_dir / "inbody_summary.json"
    report_md = output_dir / "inbody_report.md"

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(json.dumps(normalized_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md.write_text(generate_markdown_report(normalized_metrics), encoding="utf-8")

    return {
        "csv": summary_csv,
        "json": summary_json,
        "markdown": report_md,
    }


def process_inbody_file(
    input_path: Path,
    output_dir: Path,
    encodings: Tuple[str, ...] = DEFAULT_ENCODINGS,
) -> Dict[str, Path]:
    df = try_read_csv(input_path, encodings=encodings)
    metrics = extract_core_metrics(df)
    return write_outputs(metrics, output_dir)


__all__ = [
    "DEFAULT_ENCODINGS",
    "extract_core_metrics",
    "generate_markdown_report",
    "normalize_metrics",
    "process_inbody_file",
    "try_read_csv",
    "write_outputs",
]
