"""Generate a final recommendation report based on InBody summary metrics."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import textwrap
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    DOTENV_AVAILABLE = False

    def load_dotenv(*_args, **_kwargs):  # type: ignore[override]
        return False

try:
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    OpenAI = None


def normalize_key(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


def parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text in {"-", "NA", "N/A", "nan", "None"}:
        return None
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


@dataclass
class MetricEntry:
    key: str
    value: object


class MetricStore:
    def __init__(self, items: Dict[str, object]):
        grouped: Dict[str, List[MetricEntry]] = defaultdict(list)
        for key, value in items.items():
            grouped[normalize_key(key)].append(MetricEntry(key, value))
        self._items = grouped

    def get(self, *candidates: str) -> Optional[MetricEntry]:
        for candidate in candidates:
            normalized = normalize_key(candidate)
            entries = self._items.get(normalized)
            if not entries:
                continue
            candidate_lower = candidate.lower()
            for entry in entries:
                if entry.key.lower() == candidate_lower:
                    return entry
            for entry in entries:
                if candidate_lower in entry.key.lower():
                    return entry
            return min(entries, key=lambda e: len(e.key))
        return None

    def get_value(self, *candidates: str) -> Optional[object]:
        entry = self.get(*candidates)
        return entry.value if entry else None

    def get_number(self, *candidates: str) -> Optional[float]:
        return parse_float(self.get_value(*candidates))

    def get_text(self, *candidates: str) -> Optional[str]:
        value = self.get_value(*candidates)
        if value is None:
            return None
        text = str(value).strip()
        return text or None


REFERENCE_DEFAULT = Path("reference")
ASCII_CITATION_PATTERN = re.compile(r"\[(\d+)(?:[^\]]*)\]")
FULLWIDTH_CITATION_PATTERN = re.compile(r"［(\d+)(?:[^］]*)］")
DEFAULT_GPT_MODEL = os.getenv("DEFAULT_GPT_MODEL", "gpt-5")
FALLBACK_GPT_MODEL = os.getenv("FALLBACK_GPT_MODEL", "gpt-4o-mini")


KEYS = {
    "name": ["姓名", "name"],
    "id": ["id", "身份證", "身分證", "測試編號"],
    "gender": ["gender", "性別"],
    "age": ["age", "年齡"],
    "height_cm": ["height", "身高", "height_cm", "heightcm"],
    "weight_kg": ["weight", "體重", "weight_kg", "weightkg"],
    "test_time": ["測試時間", "test date / time", "test datetime"],
    "bmi": ["bmi", "體質指數"],
    "pbf": ["pbf", "percent body fat", "體脂率", "pbf_pct", "pbfpct"],
    "bfm": ["bfm", "body fat mass", "體脂肪量", "bfm_kg", "bfmkg"],
    "smm": ["smm", "skeletal muscle mass", "骨骼肌量", "smm_kg", "smmkg"],
    "smi": ["smi", "skeletal muscle index"],
    "smwt": ["smm/wt", "肌肉比"],
    "tbw": ["tbw", "total body water", "總水量", "tbw_kg", "tbwkg"],
    "icw": ["icw", "intracellular water", "icw_kg", "icwkg"],
    "ecw": ["ecw", "extracellular water", "ecw_kg", "ecwkg"],
    "ecw_tbw": ["ecw/tbw", "ecw 比 tbw", "水腫率", "ecw_tbw"],
    "ecw_tbw_ra": ["ecw/tbw of right arm", "右上肢 ecw/tbw", "rightarm_ecw_tbw"],
    "ecw_tbw_la": ["ecw/tbw of left arm", "左上肢 ecw/tbw", "leftarm_ecw_tbw"],
    "ecw_tbw_tr": ["ecw/tbw of trunk", "軀幹 ecw/tbw", "trunk_ecw_tbw"],
    "ecw_tbw_rl": ["ecw/tbw of right leg", "右下肢 ecw/tbw", "rightleg_ecw_tbw"],
    "ecw_tbw_ll": ["ecw/tbw of left leg", "左下肢 ecw/tbw", "leftleg_ecw_tbw"],
    "tbw_ra": ["tbw of right arm", "右上肢體水分", "rightarm_tbw", "rightarm_tbw_kg"],
    "tbw_la": ["tbw of left arm", "左上肢體水分", "leftarm_tbw", "leftarm_tbw_kg"],
    "tbw_tr": ["tbw of trunk", "軀幹體水分", "trunk_tbw", "trunk_tbw_kg"],
    "tbw_rl": ["tbw of right leg", "右下肢體水分", "rightleg_tbw", "rightleg_tbw_kg"],
    "tbw_ll": ["tbw of left leg", "左下肢體水分", "leftleg_tbw", "leftleg_tbw_kg"],
    "bmr": ["bmr", "basal metabolic rate", "基礎代謝", "bmr_kcal"],
    "whr": ["whr", "waist-hip ratio"],
    "vfa": ["vfa", "visceral fat area", "內臟脂肪面積", "vfa_cm2", "vfacm2"],
    "vfl": ["vfl", "visceral fat level", "vfl_level"],
    "inbody_score": ["inbody score", "inbody分數"],
    "weight_control": ["weight control", "建議體重控制", "weight_control", "weightcontrol", "weight_control_kg", "weightcontrolkg"],
    "bfm_control": ["bfm control", "fat control", "建議減脂", "fat_control", "fat_control_kg", "fatcontrolkg"],
    "ffm_control": ["ffm control", "muscle control", "建議增肌", "肌肉控制", "muscle_control", "muscle_control_kg", "musclecontrolkg"],
    "target_weight": ["target weight", "目標體重", "target_weight", "target_weight_kg", "targetweightkg"],
    "lean_ra": ["lean mass of right arm", "右上肢骨骼肌量", "rightarm_lean", "rightarm_lean_kg", "rightarmleankg"],
    "lean_la": ["lean mass of left arm", "左上肢骨骼肌量", "leftarm_lean", "leftarm_lean_kg", "leftarmleankg"],
    "lean_rl": ["lean mass of right leg", "右下肢骨骼肌量", "rightleg_lean", "rightleg_lean_kg", "rightlegleankg"],
    "lean_ll": ["lean mass of left leg", "左下肢骨骼肌量", "leftleg_lean", "leftleg_lean_kg", "leftlegleankg"],
    "lean_trunk": ["lean mass of trunk", "軀幹骨骼肌量", "trunk_lean", "trunk_lean_kg", "trunkleankg"],
    "lean_ra_pct": ["lean mass(%) of right arm", "右上肢肌肉%"],
    "lean_la_pct": ["lean mass(%) of left arm", "左上肢肌肉%"],
    "lean_rl_pct": ["lean mass(%) of right leg", "右下肢肌肉%"],
    "lean_ll_pct": ["lean mass(%) of left leg", "左下肢肌肉%"],
    "lean_trunk_pct": ["lean mass(%) of trunk", "軀幹肌肉%"],
    "bfm_ra": ["bfm of right arm", "右上肢脂肪量", "rightarm_fat", "rightarm_fat_kg", "rightarmfatkg"],
    "bfm_la": ["bfm of left arm", "左上肢脂肪量", "leftarm_fat", "leftarm_fat_kg", "leftarmfatkg"],
    "bfm_rl": ["bfm of right leg", "右下肢脂肪量", "rightleg_fat", "rightleg_fat_kg", "rightlegfatkg"],
    "bfm_ll": ["bfm of left leg", "左下肢脂肪量", "leftleg_fat", "leftleg_fat_kg", "leftlegfatkg"],
    "bfm_trunk": ["bfm of trunk", "軀幹脂肪量", "trunk_fat", "trunk_fat_kg", "trunkfatkg"],
    "bfm_ra_pct": ["bfm% of right arm", "右上肢脂肪%", "rightarm_fat_pct"],
    "bfm_la_pct": ["bfm% of left arm", "左上肢脂肪%", "leftarm_fat_pct"],
    "bfm_trunk_pct": ["bfm% of trunk", "軀幹脂肪%", "trunk_fat_pct"],
    "bfm_rl_pct": ["bfm% of right leg", "右下肢脂肪%", "rightleg_fat_pct"],
    "bfm_ll_pct": ["bfm% of left leg", "左下肢脂肪%", "leftleg_fat_pct"],
    "obesity_degree": ["obesity degree", "肥胖度", "obesitydegree_pct"],
    "ffmi": ["ffmi", "fat free mass index"],
    "fmi": ["fmi", "fat mass index"],
    "bcm": ["bcm", "body cell mass", "bcm_kg"],
    "tbw_ffm": ["tbw/ffm", "tbw_ffm", "tbw_ffm_pct"],
    "phase_ra": ["50khz-ra phase angle", "phase angle ra", "rightarm_phase_angle", "rightarm_phaseangle_deg"],
    "phase_la": ["50khz-la phase angle", "phase angle la", "leftarm_phase_angle", "leftarm_phaseangle_deg"],
    "phase_tr": ["50khz-tr phase angle", "phase angle trunk", "trunk_phase_angle", "trunk_phaseangle_deg"],
    "phase_rl": ["50khz-rl phase angle", "phase angle rl", "rightleg_phase_angle", "rightleg_phaseangle_deg"],
    "phase_ll": ["50khz-ll phase angle", "phase angle ll", "leftleg_phase_angle", "leftleg_phaseangle_deg"],
}


def load_reference_sections(reference_path: Path) -> List[str]:
    if not reference_path.exists():
        return []

    paths: List[Path] = []
    if reference_path.is_dir():
        for candidate in sorted(reference_path.rglob("*")):
            if candidate.suffix.lower() in {".md", ".txt"} and candidate.is_file():
                paths.append(candidate)
    else:
        paths.append(reference_path)

    sections: List[str] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        current: List[str] = []
        for line in text.splitlines():
            if line.strip().startswith("## ") and current:
                sections.append("\n".join(current).strip())
                current = [line]
            else:
                current.append(line)
        if current:
            sections.append("\n".join(current).strip())

    return [section for section in sections if section]


REFERENCE_LINE_PATTERN = re.compile(r"^(\d+)\.\s*(.+)$")


def load_reference_index(reference_path: Path) -> Dict[str, str]:
    if not reference_path.exists():
        return {}

    paths: List[Path] = []
    if reference_path.is_dir():
        for candidate in sorted(reference_path.rglob("*")):
            if candidate.is_file() and candidate.suffix.lower() in {".md", ".txt"}:
                paths.append(candidate)
    else:
        paths.append(reference_path)

    index: Dict[str, str] = {}
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for raw_line in text.splitlines():
            line = raw_line.strip()
            match = REFERENCE_LINE_PATTERN.match(line)
            if match:
                number, content = match.groups()
                index[number] = content.strip()
    return index


def extract_keywords_for_scoring(store: MetricStore) -> List[str]:
    terms: List[str] = []
    key_map = {
        "BMI": store.get_number(*KEYS["bmi"]) or compute_bmi(store),
        "體脂": store.get_number(*KEYS["pbf"]),
        "內臟脂肪": store.get_number(*KEYS["vfa"]),
        "ECW/TBW": store.get_number(*KEYS["ecw_tbw"]),
        "相位角": store.get_number(*KEYS.get("phase_tr", [])),
        "肌少": store.get_number(*KEYS.get("lean_ra", [])),
    }
    for name, value in key_map.items():
        if value is not None:
            terms.append(name)
    return terms


def select_reference_passages(store: MetricStore, sections: List[str], top_k: int = 3) -> List[str]:
    if not sections:
        return []
    terms = extract_keywords_for_scoring(store)
    if not terms:
        return sections[: top_k]
    scored: List[Tuple[int, str]] = []
    for section in sections:
        lower = section.lower()
        score = sum(1 for term in terms if term.lower() in lower)
        scored.append((score, section))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [section for score, section in scored[:top_k] if score > 0] or sections[: top_k]


def build_metric_profile(store: MetricStore) -> str:
    fields = {
        "姓名": store.get_text(*KEYS.get("name", [])) or "—",
        "年齡": store.get_text(*KEYS.get("age", [])) or "—",
        "性別": store.get_text(*KEYS.get("gender", [])) or "—",
        "身高(cm)": store.get_number(*KEYS.get("height_cm", [])),
        "體重(kg)": store.get_number(*KEYS.get("weight_kg", [])),
        "BMI": store.get_number(*KEYS["bmi"]) or compute_bmi(store),
        "體脂率(%)": store.get_number(*KEYS["pbf"]),
        "內臟脂肪面積(cm^2)": store.get_number(*KEYS["vfa"]),
        "內臟脂肪等級": store.get_number(*KEYS.get("vfl", [])),
        "骨骼肌量(kg)": store.get_number(*KEYS["smm"]),
        "體脂肪量(kg)": store.get_number(*KEYS["bfm"]),
        "ECW/TBW": store.get_number(*KEYS["ecw_tbw"]),
        "軀幹相位角(°)": store.get_number(*KEYS.get("phase_tr", [])),
        "左右上肢肌肉量(kg)": (
            store.get_number(*KEYS.get("lean_ra", [])),
            store.get_number(*KEYS.get("lean_la", [])),
        ),
        "左右下肢肌肉量(kg)": (
            store.get_number(*KEYS.get("lean_rl", [])),
            store.get_number(*KEYS.get("lean_ll", [])),
        ),
    }
    lines = []
    for key, value in fields.items():
        if isinstance(value, tuple):
            a, b = value
            if a is None and b is None:
                continue
            lines.append(f"{key}: {a or '—'} / {b or '—'}")
        else:
            if value is None:
                continue
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def generate_gpt_insights(
    store: MetricStore,
    reference_sections: List[str],
    model: str,
    temperature: Optional[float],
) -> Optional[str]:
    if OpenAI is None:
        raise RuntimeError("openai package未安裝，無法啟用 GPT 分析")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("未在環境變數或 .env 中找到 OPENAI_API_KEY")
    client_kwargs = {"api_key": api_key}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url
    project = os.getenv("OPENAI_PROJECT")
    if project:
        client_kwargs["project"] = project
    client = OpenAI(**client_kwargs)
    profile = build_metric_profile(store)
    context_sections = select_reference_passages(store, reference_sections)
    context = "\n\n".join(context_sections)
    prompt = textwrap.dedent(
        f"""
        你是一名運動醫學與臨床營養專家。請依照以下資訊，產出一份完整的 Markdown 報告。

        [客製化指標]
        {profile}

        [參考文獻節錄]
        {context}

        報告需求：
        - 使用繁體中文撰寫，語氣專業但易懂。
        - 以 Markdown 格式輸出，並包含下列章節：
          1. 一级標題：報告標題（可自訂）與產出日期（使用今天日期）。
          2. 基本資料：使用表格展示姓名、性別、年齡、身高、體重、測試日期、BMI、體脂率等關鍵指標。
          3. 臨床焦點摘要：2-3 條條列，聚焦個體目前最重要的生理風險或優勢。
          4. 代謝風險與病理機制：需解釋 InBody「C 型」輪廓的涵義，以及 59.6 cm²（第 5 級）內臟脂肪對代謝、發炎與胰島素敏感性的影響。
          5. 飲食策略與補充建議：至少 3 條具體建議，需引用個人數值（如體重 71.8 kg、蛋白質鎖定範圍）並說明該建議如何改善風險。
          6. 訓練與恢復處方：至少 3 條建議，需結合左右肢肌肉差異、相位角等資料，避免制式建議。
          7. 監測指標與追蹤計畫：列出主要/次要 KPI 與建議追蹤週期，需附上數值目標（例如 VFA 目標、PBF 目標）。
          8. 結語：至少 3 句話，呼應 C 型輪廓與內臟脂肪風險，點出下一步行動與複測節奏。
        - 若資料不足請明確註記「資料不足」。
        - 引用參考內容時以內文方式呈現（例如「[參考 22]」）。
        - 避免制式、泛用句型，每個段落須結合個人化數據，明確說明建議與體脂、肌肉、水分或相位角的關聯。
        """
    ).strip()
    system_prompt = "你是專業的運動醫學與營養顧問。"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    supports_temperature = True
    if model and str(model).lower().startswith("gpt-5"):
        supports_temperature = False

    if hasattr(client, "responses"):
        request_payload = {
            "model": model,
            "input": messages,
        }
        if temperature is not None and supports_temperature:
            request_payload["temperature"] = temperature
        response = client.responses.create(**request_payload)
        output = getattr(response, "output_text", None)
        if not output:
            # 對於新版 SDK，responses.create 會返回 content 字串列表
            try:
                output = "".join(part.text for part in response.output if part.type == "output_text")
            except Exception as exc:  # pragma: no cover - defensive fallback
                raise RuntimeError(f"GPT 回應解析失敗: {exc}")
        return output.strip() if output else None

    # 舊版 openai SDK 不支援 responses API，退回 chat.completions
    chat = getattr(client, "chat", None)
    completions = getattr(chat, "completions", None) if chat else None
    if completions is None:
        raise RuntimeError("目前的 openai 套件版本過舊，請升級到 1.55 以上或改用 responses API。")

    request_kwargs = {"model": model, "messages": messages}
    if temperature is not None and supports_temperature:
        request_kwargs["temperature"] = temperature
    completion = completions.create(**request_kwargs)
    choice_message = completion.choices[0].message if completion.choices else None
    output = getattr(choice_message, "content", None)
    if not output:
        raise RuntimeError("GPT 回應解析失敗（chat.completions 無內容）。")
    return output.strip()
def load_from_json(path: Path) -> Dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        converted = {}
        for entry in data:
            if not isinstance(entry, dict):
                continue
            key = entry.get("項目") or entry.get("metric") or entry.get("name")
            value = entry.get("數值") or entry.get("value")
            if key is not None:
                converted[str(key)] = value
        if converted:
            return converted
    raise ValueError("Unsupported JSON structure for summary metrics")


def load_from_csv(path: Path) -> Dict[str, object]:
    text = path.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return {}
    reader = csv.reader(lines)
    rows = list(reader)
    if not rows:
        return {}
    if len(rows[0]) == 2:
        return {row[0]: row[1] for row in rows if len(row) >= 2}
    header = rows[0]
    values = rows[1] if len(rows) > 1 else []
    return {header[i]: values[i] if i < len(values) else "" for i in range(len(header))}


def load_metrics(path: Path) -> MetricStore:
    if path.suffix.lower() == ".json":
        raw = load_from_json(path)
    elif path.suffix.lower() == ".csv":
        raw = load_from_csv(path)
    else:
        raise ValueError("Only JSON and CSV summaries are supported")
    return MetricStore(raw)


def classify_bmi(bmi: float) -> str:
    if bmi < 18.5:
        return "體重過輕"
    if bmi < 24:
        return "標準"
    if bmi < 27:
        return "過重"
    if bmi < 30:
        return "輕度肥胖"
    if bmi < 35:
        return "中度肥胖"
    return "重度肥胖"


def classify_pbf(pbf: float, gender: Optional[str]) -> str:
    if gender and gender.strip().lower().startswith("f"):
        if pbf < 18:
            return "偏低"
        if pbf <= 28:
            return "理想"
        if pbf <= 33:
            return "稍高"
        return "偏高"
    if pbf < 10:
        return "偏低"
    if pbf <= 20:
        return "理想"
    if pbf <= 25:
        return "稍高"
    return "偏高"


def format_number(value: Optional[float], unit: str = "", digits: int = 1) -> str:
    if value is None:
        return "—"
    formatted = f"{value:.{digits}f}" if isinstance(value, float) else str(value)
    return f"{formatted}{unit}" if unit else formatted


def format_test_timestamp(value: Optional[object]) -> str:
    if value is None:
        return "—"
    text = str(value).strip()
    if not text:
        return "—"
    digits = "".join(ch for ch in text if ch.isdigit())
    try:
        if len(digits) == 14:
            dt = datetime.strptime(digits, "%Y%m%d%H%M%S")
            return dt.strftime("%Y-%m-%d %H:%M")
        if len(digits) == 12:
            dt = datetime.strptime(digits, "%Y%m%d%H%M")
            return dt.strftime("%Y-%m-%d %H:%M")
        if len(digits) == 8:
            dt = datetime.strptime(digits, "%Y%m%d")
            return dt.strftime("%Y-%m-%d")
    except ValueError:
        return text
    return text


def analyze_weight(store: MetricStore) -> List[str]:
    bmi = store.get_number(*KEYS["bmi"]) or compute_bmi(store)
    weight = store.get_number(*KEYS["weight_kg"])
    height = store.get_number(*KEYS["height_cm"])
    gender = store.get_text(*KEYS["gender"])
    pbf = store.get_number(*KEYS["pbf"])
    weight_control = store.get_number(*KEYS["weight_control"])
    fat_control = store.get_number(*KEYS["bfm_control"])
    muscle_control = store.get_number(*KEYS["ffm_control"])
    lines: List[str] = []
    if bmi is not None:
        lines.append(f"BMI {bmi:.1f}，屬於{classify_bmi(bmi)}。")
    if weight is not None and height is not None:
        lines.append(f"體重 {weight:.1f} kg，身高 {height:.1f} cm。")
    if pbf is not None:
        lines.append(f"體脂率 {pbf:.1f}%（{classify_pbf(pbf, gender)}）。")
    target_weight = store.get_number(*KEYS["target_weight"])
    if target_weight is not None and weight is not None:
        delta = weight - target_weight
        direction = "增加" if delta < 0 else "減少"
        lines.append(f"建議目標體重 {target_weight:.1f} kg，與目前差異 {abs(delta):.1f} kg（需{direction}）。")
    elif weight_control is not None:
        if abs(weight_control) < 0.2:
            lines.append("體重控制目標已接近理想值，維持目前策略即可。")
        else:
            action = "降低" if weight_control < 0 else "增加"
            lines.append(f"建議體重調整 {weight_control:.1f} kg（朝向{action}）。")
    if fat_control is not None:
        action = "減脂" if fat_control < 0 else "增加脂肪"
        lines.append(f"脂肪控制指標：{action} {abs(fat_control):.1f} kg。")
    if muscle_control is not None:
        if muscle_control > 0:
            lines.append(f"肌肉量建議增加 {muscle_control:.1f} kg，安排阻力訓練與蛋白質補充。")
        elif muscle_control < 0:
            lines.append(f"肌肉量建議調降 {abs(muscle_control):.1f} kg，以保持整體平衡。")
        else:
            lines.append("肌肉量維持即可，重視恢復與蛋白質攝取。")
    return lines


def compute_bmi(store: MetricStore) -> Optional[float]:
    height_cm = store.get_number(*KEYS["height_cm"])
    weight = store.get_number(*KEYS["weight_kg"])
    if height_cm is None or weight is None or height_cm <= 0:
        return None
    return weight / ((height_cm / 100) ** 2)


def muscle_pair_differences(store: MetricStore) -> List[Tuple[str, float]]:
    pairs = [
        ("上肢", KEYS.get("lean_ra", []), KEYS.get("lean_la", [])),
        ("下肢", KEYS.get("lean_rl", []), KEYS.get("lean_ll", [])),
    ]
    diffs: List[Tuple[str, float]] = []
    for label, key_a, key_b in pairs:
        if not key_a or not key_b:
            continue
        value_a = store.get_number(*key_a)
        value_b = store.get_number(*key_b)
        if value_a is None or value_b is None:
            continue
        average = (value_a + value_b) / 2
        if average == 0:
            continue
        diff_pct = abs(value_a - value_b) / average * 100
        diffs.append((label, diff_pct))
    return diffs


def analyze_body_composition(store: MetricStore) -> List[str]:
    smm = store.get_number(*KEYS["smm"])
    smi = store.get_number(*KEYS["smi"])
    smwt = store.get_number(*KEYS["smwt"])
    bfm = store.get_number(*KEYS["bfm"])
    vfa = store.get_number(*KEYS["vfa"])
    vfl = store.get_number(*KEYS["vfl"])
    bmr = store.get_number(*KEYS["bmr"])
    inbody_score = store.get_number(*KEYS["inbody_score"])
    whr = store.get_number(*KEYS["whr"]) if "whr" in KEYS else store.get_number("WHR")
    obesity_degree = store.get_number(*KEYS.get("obesity_degree", []))
    ffmi = store.get_number(*KEYS.get("ffmi", []))
    fmi = store.get_number(*KEYS.get("fmi", []))
    lines: List[str] = []
    if smm is not None:
        lines.append(f"骨骼肌量 {smm:.1f} kg。")
    if smi is not None:
        status = "低於肌少症門檻" if (store.get_text(*KEYS["gender"]) or "").lower().startswith("m") and smi < 7.0 or (store.get_text(*KEYS["gender"]) or "").lower().startswith("f") and smi < 5.7 else "在健康範圍內"
        lines.append(f"SMI {smi:.2f}（{status}）。")
    elif smwt is not None:
        lines.append(f"肌肉占體重比例 {smwt:.2f}。")
    if bfm is not None:
        lines.append(f"體脂肪量 {bfm:.1f} kg。")
    if vfa is not None:
        remark = "偏高，需特別注意腹部脂肪" if vfa >= 100 else "位於建議範圍內"
        lines.append(f"內臟脂肪面積 {vfa:.0f} cm²（{remark}）。")
    ecw_tbw = store.get_number(*KEYS["ecw_tbw"])
    if ecw_tbw is not None:
        status = "疑似水腫" if ecw_tbw >= 0.390 else "水分平衡正常"
        lines.append(f"ECW/TBW {ecw_tbw:.3f}（{status}）。")
    tbw = store.get_number(*KEYS["tbw"])
    if tbw is not None:
        lines.append(f"總體水量 {tbw:.1f} L。")
    if bmr is not None:
        lines.append(f"基礎代謝率 {bmr:.0f} kcal。")
    if whr is not None:
        whr_status = "腹部肥胖風險" if whr >= 0.9 else "維持腰臀比" if whr >= 0.8 else "腰臀比健康"
        lines.append(f"腰臀比 {whr:.2f}（{whr_status}）。")
    if obesity_degree is not None:
        lines.append(f"肥胖度指數 {obesity_degree:.0f}%（100% 為標準體重基準）。")
    if ffmi is not None:
        lines.append(f"FFMI {ffmi:.1f}，反映無脂體重相對表現。")
    if fmi is not None:
        lines.append(f"FMI {fmi:.1f}，可作為長期體脂追蹤指標。")
    if inbody_score is not None:
        lines.append(f"InBody 分數 {inbody_score:.0f}。")
    if vfl is not None:
        remark = "需特別注意內臟脂肪堆積" if vfl >= 10 else "維持目前生活型態" if vfl <= 5 else "內臟脂肪尚可，但建議持續監測"
        lines.append(f"內臟脂肪等級 {vfl:.0f}（{remark}）。")
    return lines


def analyze_controls(store: MetricStore) -> List[str]:
    weight_control = store.get_number(*KEYS["weight_control"])
    bfm_control = store.get_number(*KEYS["bfm_control"])
    ffm_control = store.get_number(*KEYS["ffm_control"])
    if weight_control is None:
        current_weight = store.get_number(*KEYS["weight_kg"])
        target_weight = store.get_number(*KEYS["target_weight"])
        if current_weight is not None and target_weight is not None:
            weight_control = target_weight - current_weight
    lines: List[str] = []
    if weight_control is not None:
        if abs(weight_control) < 0.3:
            lines.append("體重已接近儀器建議，持續紀錄飲食與作息維持穩定。")
        else:
            direction = "熱量赤字" if weight_control < 0 else "熱量盈餘"
            daily_delta = abs(weight_control) * 7700 / 56  # 8 週調整目標
            lines.append(
                f"建立每週檢核的{direction}計畫，平均每日熱量調整約 {daily_delta:.0f} kcal 以逐步達成。"
            )
    if bfm_control is not None:
        if bfm_control < -0.3:
            lines.append("強化阻力與有氧混合訓練，每週至少 150 分鐘中高強度以加速減脂。")
        elif bfm_control > 0.3:
            lines.append("需增加脂肪量時，優先提高優質碳水與總熱量，避免過度運動消耗。")
        else:
            lines.append("體脂肪與建議值差距不大，維持現有訓練配置即可。")
    if ffm_control is not None:
        if ffm_control > 0.3:
            lines.append("每餐攝取 20-30g 蛋白質並搭配逐步超負荷訓練，以支撐肌肉量提升。")
        elif ffm_control < -0.3:
            lines.append("若需降低無脂體重，建議與專業教練評估期程，以避免過度流失肌肉。")
        else:
            lines.append("肌肉量調整幅度不大，維持訓練強度並注意恢復即可。")
    return lines


def analyze_segmental(store: MetricStore) -> List[str]:
    def diff_message(value_a: Optional[float], value_b: Optional[float], label_a: str, label_b: str) -> Optional[str]:
        if value_a is None or value_b is None:
            return None
        average = (value_a + value_b) / 2
        if average == 0:
            return None
        gap = abs(value_a - value_b) / average
        if gap < 0.1:
            return None
        stronger = label_a if value_a > value_b else label_b
        return f"{stronger} 肌肉量較另一側高出約 {gap * 100:.1f}% ，建議安排矯正訓練。"

    lines: List[str] = []
    lean_ra = store.get_number(*KEYS["lean_ra"])
    lean_la = store.get_number(*KEYS["lean_la"])
    lean_rl = store.get_number(*KEYS["lean_rl"])
    lean_ll = store.get_number(*KEYS["lean_ll"])
    lean_trunk = store.get_number(*KEYS["lean_trunk"])
    bfm_ra = store.get_number(*KEYS["bfm_ra"])
    bfm_la = store.get_number(*KEYS["bfm_la"])
    bfm_rl = store.get_number(*KEYS["bfm_rl"])
    bfm_ll = store.get_number(*KEYS["bfm_ll"])
    bfm_trunk = store.get_number(*KEYS["bfm_trunk"])

    for message in (
        diff_message(lean_ra, lean_la, "右上肢", "左上肢"),
        diff_message(lean_rl, lean_ll, "右下肢", "左下肢"),
    ):
        if message:
            lines.append(message)

    segments = [
        ("右上肢", lean_ra, bfm_ra),
        ("左上肢", lean_la, bfm_la),
        ("軀幹", lean_trunk, bfm_trunk),
        ("右下肢", lean_rl, bfm_rl),
        ("左下肢", lean_ll, bfm_ll),
    ]
    for label, lean_value, fat_value in segments:
        if lean_value is None and fat_value is None:
            continue
        lean_text = format_number(lean_value, " kg")
        fat_text = format_number(fat_value, " kg")
        lines.append(f"{label}：肌肉 {lean_text} / 脂肪 {fat_text}。")

    for label, key in (
        ("右上肢", "lean_ra_pct"),
        ("左上肢", "lean_la_pct"),
        ("右下肢", "lean_rl_pct"),
        ("左下肢", "lean_ll_pct"),
        ("軀幹", "lean_trunk_pct"),
    ):
        value = store.get_number(*KEYS.get(key, [])) if key in KEYS else None
        if value is None:
            continue
        if value < 90:
            lines.append(f"{label} 肌肉發展僅 {value:.0f}%（低於建議），可加強該部位訓練。")
        elif value > 110:
            lines.append(f"{label} 肌肉發展 {value:.0f}%（相對突出），注意左右協調。")

    if not lines and any(v is not None for v in (lean_ra, lean_la, lean_rl, lean_ll, lean_trunk)):
        lines.append("四肢與軀幹肌肉量分佈均衡，維持現有訓練即可。")
    return lines


def build_clinical_summary(store: MetricStore) -> List[str]:
    bmi = store.get_number(*KEYS["bmi"]) or compute_bmi(store)
    pbf = store.get_number(*KEYS["pbf"])
    vfa = store.get_number(*KEYS["vfa"])
    smm = store.get_number(*KEYS["smm"])
    bfm = store.get_number(*KEYS["bfm"])
    weight = store.get_number(*KEYS["weight_kg"])
    ecw_tbw = store.get_number(*KEYS["ecw_tbw"])
    trunk_phase = store.get_number(*KEYS.get("phase_tr", []))
    fat_control = store.get_number(*KEYS["bfm_control"])
    vfl = store.get_number(*KEYS.get("vfl", []))
    gender = store.get_text(*KEYS["gender"]) if "gender" in KEYS else None
    bmr = store.get_number(*KEYS.get("bmr", []))
    smi = store.get_number(*KEYS.get("smi", []))
    score = store.get_number(*KEYS.get("inbody_score", []))
    tbw = store.get_number(*KEYS.get("tbw", []))
    icw = store.get_number(*KEYS.get("icw", []))
    ecw = store.get_number(*KEYS.get("ecw", []))
    tbw_ffm = store.get_number(*KEYS.get("tbw_ffm", []))
    bcm = store.get_number(*KEYS.get("bcm", []))
    lean_ra = store.get_number(*KEYS.get("lean_ra", []))
    lean_la = store.get_number(*KEYS.get("lean_la", []))
    lean_rl = store.get_number(*KEYS.get("lean_rl", []))
    lean_ll = store.get_number(*KEYS.get("lean_ll", []))
    vfl_map = {0: "最低", 1: "極低", 2: "偏低", 3: "低", 4: "稍高", 5: "中等", 6: "偏高", 7: "高", 8: "很高", 9: "極高", 10: "危險"}
    lines: List[str] = []
    if smm is not None and bfm is not None:
        muscle_text = f"骨骼肌量 {smm:.1f} kg"
        fat_text = f"體脂肪量 {bfm:.1f} kg"
        muscle_gap: Optional[float]
        if weight is not None and weight > 0:
            smm_ratio = smm / weight * 100
            bfm_ratio = bfm / weight * 100
            muscle_text += f"（佔體重 {smm_ratio:.1f}%）"
            fat_text += f"（約 {bfm_ratio:.1f}% 體脂）"
            muscle_gap = weight - smm
            c_shape_flag = muscle_gap > 20 or (pbf is not None and pbf >= 25)
        else:
            muscle_gap = None
            c_shape_flag = pbf is not None and pbf >= 25
        details = f"{muscle_text}、{fat_text}"
        if muscle_gap is not None:
            details += f"，體重-骨骼肌差 {muscle_gap:.1f} kg"
        if c_shape_flag:
            details += (
                "，呈現 InBody C 型輪廓（肌肉量柱狀圖相對脂肪柱較短），代表軀幹與四肢肌肉量支撐不足；"
                "需以阻力訓練與熱量管理同步提升肌脂比。[1][14]"
            )
        else:
            details += "，為設定訓練與飲食目標的核心 KPI。[1][14]"
        lines.append(details)

    if bmi is not None and pbf is not None:
        bmi_class = classify_bmi(bmi)
        pbf_class = classify_pbf(pbf, gender)
        risk_clause = "需警覺內臟脂肪牽引的代謝風險" if bmi < 25 and pbf >= 20 else "建議持續穩定監測月度變化"
        lines.append(
            f"BMI {bmi:.1f}（{bmi_class}）與體脂率 {pbf:.1f}%（{pbf_class}），{risk_clause}。[2][12]"
        )

    if vfa is not None or vfl is not None:
        components: List[str] = []
        if vfa is not None:
            components.append(f"內臟脂肪面積 {vfa:.1f} cm²")
        if vfl is not None:
            components.append(f"等級 {vfl:.0f}")
        status: List[str] = []
        if vfa is not None:
            if vfa >= 120:
                status.append("臨床極高風險，建議医疗團隊密切監測")
            elif vfa >= 100:
                status.append("超過 100 cm² 高風險門檻")
            elif vfa >= 70:
                status.append("高於亞洲族群建議上限 70 cm²")
            elif vfa >= 50:
                status.append("突破 50 cm² 健康警戒，需要加速腰腹調整")
            else:
                status.append("維持在健康區間，持續觀察")
        if vfl is not None:
            if vfl >= 10:
                status.append("等級 ≥10，內臟脂肪堆積加劇")
            elif vfl >= 5:
                status.append(
                    f"等級位於監測上緣（VFL {vfl:.0f}，{vfl_map.get(int(vfl), '中等')}），須留意復胖風險"
                )
            else:
                status.append("等級仍在低風險範圍")
        visceral_priority = (vfa is not None and vfa >= 50) or (vfl is not None and vfl >= 5)
        action = "需把腰腹訓練與飲食控糖視為第一優先" if visceral_priority else "維持現有生活型態並定期複測"
        base_text = " / ".join(components)
        if status:
            lines.append(f"{base_text}，{'；'.join(status)}，{action}。[22]")
        else:
            lines.append(f"{base_text}，{action}。[22]")

    if ecw_tbw is not None or trunk_phase is not None:
        segments: List[str] = []
        if ecw_tbw is not None:
            if ecw_tbw >= 0.39:
                segments.append(f"ECW/TBW {ecw_tbw:.3f} 高於 0.39，暗示外液滯留或睡眠恢復不足")
            elif ecw_tbw <= 0.360:
                segments.append(f"ECW/TBW {ecw_tbw:.3f} 偏低，注意水分與電解質補充")
            else:
                segments.append(f"ECW/TBW {ecw_tbw:.3f} 落在 0.36-0.39，水分平衡穩定")
        if trunk_phase is not None:
            if trunk_phase < 5.5:
                segments.append(
                    f"軀幹相位角 {trunk_phase:.1f}° 偏低，推測細胞膜電阻下降；須加強蛋白質與睡眠修復"
                )
            elif trunk_phase >= 7.5:
                segments.append(
                    f"軀幹相位角 {trunk_phase:.1f}° 高於 7.5°，代表細胞活性與恢復效率佳"
                )
            else:
                segments.append(
                    f"軀幹相位角 {trunk_phase:.1f}° 居於中段（約 5.5-7.5° 被視為穩定範圍），維持規律訓練與恢復節奏"
                )
        if segments:
            lines.append("；".join(segments) + "。")

    limb_snapshots: List[str] = []
    if lean_ra is not None and lean_la is not None:
        arms_diff = abs(lean_ra - lean_la)
        avg_arms = (lean_ra + lean_la) / 2 if (lean_ra + lean_la) != 0 else 0
        if avg_arms > 0:
            diff_pct = arms_diff / avg_arms * 100
            balance_note = f"差 {diff_pct:.1f}%" if diff_pct >= 5 else "左右差 <5%"
        else:
            balance_note = "資料不足"
        limb_snapshots.append(f"上肢肌肉 {lean_ra:.2f}/{lean_la:.2f} kg（{balance_note}）")
    if lean_rl is not None and lean_ll is not None:
        legs_diff = abs(lean_rl - lean_ll)
        avg_legs = (lean_rl + lean_ll) / 2 if (lean_rl + lean_ll) != 0 else 0
        if avg_legs > 0:
            diff_pct = legs_diff / avg_legs * 100
            balance_note = f"差 {diff_pct:.1f}%" if diff_pct >= 5 else "左右差 <5%"
        else:
            balance_note = "資料不足"
        limb_snapshots.append(f"下肢肌肉 {lean_rl:.2f}/{lean_ll:.2f} kg（{balance_note}）")
    if limb_snapshots:
        lines.append("四肢肌肉平衡: " + "；".join(limb_snapshots) + "，依此調整單側與矯正訓練。")

    if fat_control is not None:
        target = abs(fat_control)
        direction = "減脂" if fat_control < 0 else "增脂"
        smm_text = f"維持 {smm:.1f} kg 骨骼肌量" if smm is not None else "維持既有骨骼肌量"
        lines.append(f"首要身體重組目標：在 {smm_text} 的前提下{direction}{target:.1f} kg，並壓低內臟脂肪。")

    if bmr is not None:
        lines.append(
            f"基礎代謝率 {bmr:.0f} kcal（同齡男性常見範圍 1500-1700 kcal），建議以阻力訓練搭配蛋白質補充維持代謝動力。"
        )

    if smi is not None:
        smi_threshold = 7.0 if gender and gender.lower().startswith("m") else 5.7
        status = "高於肌少症門檻" if smi >= smi_threshold else "低於肌少症門檻" if smi < smi_threshold else "資料不足"
        lines.append(f"SMI {smi:.1f}（{status}），持續維持下肢力量並定期評估步態。")

    if score is not None:
        lines.append(f"InBody 分數 {score:.0f}，距離 80 分健康門檻仍有差距，建議 8-12 週後複測追蹤。")

    if tbw is not None or (icw is not None and ecw is not None):
        water_parts: List[str] = []
        if tbw is not None:
            water_parts.append(f"TBW {tbw:.1f} L")
        if icw is not None and ecw is not None:
            water_parts.append(f"ICW/ECW {icw:.1f}/{ecw:.1f} L")
        if tbw_ffm is not None:
            water_parts.append(f"TBW/FFM {tbw_ffm:.1f}%")
        if water_parts:
            lines.append("體液指標：" + "，".join(water_parts) + "，維持水分與電解質平衡可支撐訓練恢復。")

    if bcm is not None:
        lines.append(f"BCM {bcm:.1f} kg 反映細胞活性與肌肉品質，是觀察訓練成效的重要補充 KPI。")

    if not lines:
        return ["目前資料不足以形成臨床摘要。"]
    return lines


def analyze_metabolic_risk(store: MetricStore) -> List[str]:
    bmi = store.get_number(*KEYS["bmi"]) or compute_bmi(store)
    pbf = store.get_number(*KEYS["pbf"])
    vfa = store.get_number(*KEYS["vfa"])
    vfl = store.get_number(*KEYS.get("vfl", []))
    whr = store.get_number(*KEYS.get("whr", []))
    obesity_degree = store.get_number(*KEYS.get("obesity_degree", []))
    score = store.get_number(*KEYS["inbody_score"])
    ecw_tbw = store.get_number(*KEYS["ecw_tbw"])
    lines: List[str] = []
    if bmi is not None and pbf is not None:
        lines.append(f"雖然 BMI {bmi:.1f} 位於健康範圍，但體脂率 {pbf:.1f}% 已逼近年齡上限，顯示正常體重肥胖的代謝弱點。[2][13]")
    if vfa is not None:
        lines.append(f"內臟脂肪面積 {vfa:.1f} cm² 與等級 {int(vfl) if vfl is not None else '—'} 為發炎與胰島素阻抗的主要引擎，需結合飲食與高強度訓練降低。[20][22]")
    if whr is not None or obesity_degree is not None:
        descriptor = []
        if whr is not None and whr >= 0.90:
            descriptor.append(f"腰臀比 {whr:.2f} 偏高")
        elif whr is not None:
            descriptor.append(f"腰臀比 {whr:.2f}")
        if obesity_degree is not None:
            descriptor.append(f"肥胖度指數 {obesity_degree:.0f}%")
        if descriptor:
            lines.append("、".join(descriptor) + " 指向中心性脂肪堆積與心代謝負荷。")
    if ecw_tbw is not None:
        lines.append(f"ECW/TBW {ecw_tbw:.3f} 維持在 0.36-0.38 區間，顯示目前仍無明顯水腫，是扭轉風險的最佳時機。[37]")
    if score is not None:
        lines.append(f"InBody 分數 {score:.0f} 仍低於 80，建議 12 週內複測以確認風險是否下降。")
    if not lines:
        lines.append("缺乏足夠的代謝風險資料。")
    return lines


def analyze_fluid_balance(store: MetricStore) -> List[str]:
    overall = store.get_number(*KEYS["ecw_tbw"])
    segments = [
        ("右上肢", store.get_number(*KEYS.get("ecw_tbw_ra", []))),
        ("左上肢", store.get_number(*KEYS.get("ecw_tbw_la", []))),
        ("軀幹", store.get_number(*KEYS.get("ecw_tbw_tr", []))),
        ("右下肢", store.get_number(*KEYS.get("ecw_tbw_rl", []))),
        ("左下肢", store.get_number(*KEYS.get("ecw_tbw_ll", []))),
    ]
    lines: List[str] = []
    if overall is not None:
        status = "偏高，可能有水腫" if overall >= 0.39 else "位於建議區間" if overall <= 0.38 else "需要持續觀察"
        lines.append(f"全身 ECW/TBW {overall:.3f}（{status}）。")
    deviations = []
    for label, value in segments:
        if value is None:
            continue
        if value >= 0.39:
            message = f"{label} ECW/TBW {value:.3f}（局部水份偏高）。"
        elif value <= 0.36:
            message = f"{label} ECW/TBW {value:.3f}（偏低，注意水分補充）。"
        else:
            message = f"{label} ECW/TBW {value:.3f}（維持在正常範圍）。"
        lines.append(message)
        deviations.append(value)
    valid_values = [v for v in deviations if v is not None]
    if len(valid_values) >= 2:
        gap = max(valid_values) - min(valid_values)
        if gap >= 0.015:
            lines.append("四肢水分分布差異超過 0.015，建議檢視姿勢或日常活動是否不平衡。")
    if not lines:
        lines.append("水分分布資訊不足或在標準範圍內。")
    return lines


def analyze_fat_distribution(store: MetricStore) -> List[str]:
    def pct_message(label: str, value: Optional[float]) -> Optional[str]:
        if value is None:
            return None
        if value >= 130:
            return f"{label} 脂肪百分比 {value:.0f}%（顯著高於標準）。"
        if value <= 80:
            return f"{label} 脂肪百分比 {value:.0f}%（低於標準，留意營養狀況）。"
        return f"{label} 脂肪百分比 {value:.0f}%（接近標準）。"

    lines: List[str] = []
    for label, key in (
        ("右上肢", "bfm_ra_pct"),
        ("左上肢", "bfm_la_pct"),
        ("軀幹", "bfm_trunk_pct"),
        ("右下肢", "bfm_rl_pct"),
        ("左下肢", "bfm_ll_pct"),
    ):
        message = pct_message(label, store.get_number(*KEYS.get(key, [])))
        if message:
            lines.append(message)
    if not lines:
        lines.append("缺少脂肪百分比資料，無法評估分佈情況。")
    return lines


def analyze_research_metrics(store: MetricStore) -> List[str]:
    bcm = store.get_number(*KEYS.get("bcm", []))
    tbw_ffm = store.get_number(*KEYS.get("tbw_ffm", []))
    ffmi = store.get_number(*KEYS.get("ffmi", []))
    fmi = store.get_number(*KEYS.get("fmi", []))
    phases = [
        ("右上肢", store.get_number(*KEYS.get("phase_ra", []))),
        ("左上肢", store.get_number(*KEYS.get("phase_la", []))),
        ("軀幹", store.get_number(*KEYS.get("phase_tr", []))),
        ("右下肢", store.get_number(*KEYS.get("phase_rl", []))),
        ("左下肢", store.get_number(*KEYS.get("phase_ll", []))),
    ]
    lines: List[str] = []
    if bcm is not None:
        lines.append(f"身體細胞量 BCM {bcm:.1f} kg，反映細胞活性與肌肉量。")
    if tbw_ffm is not None:
        lines.append(f"TBW/FFM {tbw_ffm:.1f}%，評估組織水分佔比。")
    if ffmi is not None and fmi is not None:
        ratio = ffmi / fmi if fmi else None
        lines.append(f"FFMI/FMI 比 {ffmi:.1f} / {fmi:.1f}{f'（比例 {ratio:.2f}）' if ratio else ''}，可做體態追蹤基準。")
    phase_values = [value for _, value in phases if value is not None]
    for label, value in phases:
        if value is None:
            continue
        status = "良好" if value >= 7 else "偏低" if value < 5.5 else "中等"
        lines.append(f"{label} 相位角 {value:.1f}°（{status}）。")
    if phase_values and (max(phase_values) - min(phase_values)) >= 1.0:
        lines.append("相位角左右差異超過 1 度，檢視訓練負荷是否不均。")
    if not lines:
        lines.append("目前缺少進階研究指標資料。")
    return lines


def recommend_nutrition_strategy(store: MetricStore) -> List[str]:
    weight = store.get_number(*KEYS["weight_kg"])
    lines: List[str] = []
    lines.append("設定每日 500-750 kcal 熱量赤字，並搭配每週體重與腰圍紀錄管控進度。[46]")
    if weight is not None:
        min_protein = weight * 1.6
        max_protein = weight * 2.0
        lines.append(
            f"蛋白質鎖定 {min_protein:.0f}-{max_protein:.0f} g/日（1.6-2.0 g/kg），支撐減脂過程的肌肉保存。[47][50]"
        )
        lines.append("將蛋白質均分 3-4 餐，每餐 25-30 g，協助最大化肌肉蛋白質合成。[50]")
    lines.append("飲食組成以高纖低糖、足量蔬菜與 ω-3 脂肪酸為核心，緩解內臟脂肪引起的慢性發炎。[22][46]")
    lines.append("保持 30-35 mL/kg 的飲水量並留意鈉攝取，支撐 0.36-0.38 的 ECW/TBW 水準。")
    return lines


def recommend_training_strategy(store: MetricStore) -> List[str]:
    diffs = muscle_pair_differences(store)
    lines: List[str] = []
    lines.append("每週安排 3-4 次阻力訓練，採用全身多關節動作並逐步超負荷，搭配 2 次 20-30 分鐘 HIIT 或中高強度有氧以降低 VFA。[22][54]")
    lines.append("訓練結束後加入 10-15 分鐘核心穩定與髖/肩等矯正動作，預防不對稱造成代償。")
    if diffs:
        focus_segments = [label for label, diff in diffs if diff >= 10]
        if focus_segments:
            lines.append("針對 " + "、".join(focus_segments) + " 啟動單側訓練與神經肌肉控制，避免代償與過度使用傷害。[39][42]")
    lines.append("確保每週至少 2 次 7-8 小時的高品質睡眠視窗並安排減壓活動，以維持相位角與荷爾蒙平衡。[27]")
    return lines


def build_monitoring_targets(store: MetricStore) -> List[str]:
    vfa = store.get_number(*KEYS["vfa"])
    pbf = store.get_number(*KEYS["pbf"])
    smm = store.get_number(*KEYS["smm"])
    phase_values = [
        store.get_number(*KEYS.get("phase_ra", [])),
        store.get_number(*KEYS.get("phase_la", [])),
        store.get_number(*KEYS.get("phase_tr", [])),
        store.get_number(*KEYS.get("phase_rl", [])),
        store.get_number(*KEYS.get("phase_ll", [])),
    ]
    ecw_tbw = store.get_number(*KEYS["ecw_tbw"])
    diffs = muscle_pair_differences(store)
    lines: List[str] = []
    major_targets: List[str] = []
    if vfa is not None:
        major_targets.append(f"VFA {vfa:.1f} → <50 cm²")
    if pbf is not None:
        major_targets.append(f"PBF {pbf:.1f}% → 18-20%")
    if smm is not None:
        major_targets.append(f"SMM 維持 ≥ {smm:.1f} kg")
    if major_targets:
        lines.append("主要指標：" + "；".join(major_targets))
    secondary: List[str] = []
    if phase_values and any(value is not None for value in phase_values):
        secondary.append("相位角 ↑0.3-0.5°，尤其是左右上肢")
    if ecw_tbw is not None:
        secondary.append("ECW/TBW 維持 0.360-0.380")
    for label, diff in diffs:
        if diff >= 10:
            secondary.append(f"{label} 肌肉差距 <5%")
    if secondary:
        lines.append("次要指標：" + "；".join(secondary))
    lines.append("量測節奏：每 12 週重測 InBody 佐以腰圍與血壓紀錄，評估策略成效。")
    return lines


def renumber_citations(lines: List[str], _reference_index: Dict[str, str]) -> Tuple[List[str], List[str]]:
    def strip_reference_labels(text: str) -> str:
        text = re.sub(r"\[\s*參考\s*([^\]]+)\]", lambda m: f"[{m.group(1).strip()}]", text)
        text = re.sub(r"［\s*參考\s*([^］]+)］", lambda m: f"[{m.group(1).strip()}]", text)
        return text

    cleaned_lines = [strip_reference_labels(line) for line in lines]
    updated_lines: List[str] = []
    for line in cleaned_lines:
        line = ASCII_CITATION_PATTERN.sub("", line)
        line = FULLWIDTH_CITATION_PATTERN.sub("", line)
        line = re.sub(r"【\s*參考[^】]*】", "", line)
        line = re.sub(r"內文已以「」標註[：:]*\s*", "", line)
        line = re.sub(r"\s{2,}", " ", line)
        updated_lines.append(line)
    return updated_lines, []


def build_report(
    store: MetricStore,
    gpt_insights: Optional[str] = None,
    reference_index: Optional[Dict[str, str]] = None,
) -> str:
    if gpt_insights:
        report = gpt_insights.strip()
        if not report.endswith("\n"):
            report += "\n"
        return report

    name = store.get_text(*KEYS["name"]) or ""
    gender = store.get_text(*KEYS["gender"]) or ""
    age = store.get_text(*KEYS["age"]) or ""
    height = store.get_number(*KEYS["height_cm"])
    weight = store.get_number(*KEYS["weight_kg"])
    test_time_raw = store.get_value(*KEYS["test_time"]) if KEYS.get("test_time") else None
    test_time = format_test_timestamp(test_time_raw)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: List[str] = []
    lines.append("# InBody 最終建議報告")
    lines.append("")
    lines.append(f"報告產出時間：{now}")
    lines.append("")
    lines.append("## 基本資料")
    profile = [
        f"姓名：{name or '—'}",
        f"性別：{gender or '—'}",
        f"年齡：{age or '—'}",
        f"身高：{format_number(height, ' cm')}",
        f"體重：{format_number(weight, ' kg')}",
        f"測試時間：{test_time}",
    ]
    lines.extend(f"- {item}" for item in profile)
    lines.append("")

    sections: List[Tuple[str, List[str]]] = [
        ("臨床執行摘要", build_clinical_summary(store)),
        ("代謝風險解析", analyze_metabolic_risk(store)),
        ("體重與體脂分析", analyze_weight(store)),
        ("身體組成重點", analyze_body_composition(store)),
        ("體重控制建議", analyze_controls(store)),
        ("部位肌肉與脂肪", analyze_segmental(store)),
        ("水分平衡分析", analyze_fluid_balance(store)),
        ("脂肪分佈評估", analyze_fat_distribution(store)),
        ("研究指標補充", analyze_research_metrics(store)),
        ("營養策略", recommend_nutrition_strategy(store)),
        ("訓練與修復策略", recommend_training_strategy(store)),
        ("階段性監測指標", build_monitoring_targets(store)),
        ("附註說明", build_appendix_notes(store)),
    ]

    if gpt_insights:
        sections.append(("GPT-5 個人化洞察", [gpt_insights]))

    for title, content in sections:
        lines.append(f"## {title}")
        if content:
            lines.extend(f"- {item}" for item in content)
        else:
            lines.append("- 資料不足，無法評估。")
        lines.append("")

    lines.append("## 總結與下一步")
    summary_points = build_summary(store)
    lines.extend(f"- {point}" for point in summary_points)
    lines.append("")

    final_lines, reference_entries = renumber_citations(lines, reference_index or {})
    if reference_entries:
        final_lines.append("## References")
        final_lines.extend(f"- {entry}" for entry in reference_entries)
        final_lines.append("")
    report_text = "\n".join(final_lines).strip() + "\n"
    report_text = re.sub(r"文中引用之數字編號[^\n]+\n?", "", report_text)
    report_text = re.sub(r"\[\s*參考[^\]]*\]", "", report_text)
    report_text = re.sub(r"［\s*參考[^］]*］", "", report_text)
    report_text = re.sub(r"【\s*參考[^】]*】", "", report_text)
    report_text = re.sub(r"\[[^\]]*\d+[^\]]*\]", "", report_text)
    report_text = re.sub(r"［[^］]*\d+[^］]*］", "", report_text)
    report_text = re.sub(r"（\s*見\s*）", "", report_text)
    report_text = re.sub(r"\(\s*見\s*\)", "", report_text)
    report_text = re.sub(r"見\s*$", "", report_text, flags=re.MULTILINE)
    report_text = re.sub(r"\s{2,}", " ", report_text)
    report_text = re.sub(r"\(\s*\)", "", report_text)
    report_text = re.sub(r"（\s*）", "", report_text)
    return report_text


def build_summary(store: MetricStore) -> List[str]:
    summary: List[str] = []
    weight_control = store.get_number(*KEYS["weight_control"])
    bfm_control = store.get_number(*KEYS["bfm_control"])
    ffm_control = store.get_number(*KEYS["ffm_control"])
    pbf = store.get_number(*KEYS["pbf"])
    bmi = store.get_number(*KEYS["bmi"]) or compute_bmi(store)
    ecw_tbw = store.get_number(*KEYS["ecw_tbw"])
    vfa = store.get_number(*KEYS["vfa"])
    score = store.get_number(*KEYS["inbody_score"])
    vfl = store.get_number(*KEYS.get("vfl", []))
    whr = store.get_number(*KEYS.get("whr", []))
    phase_values = [
        store.get_number(*KEYS.get("phase_ra", [])),
        store.get_number(*KEYS.get("phase_la", [])),
        store.get_number(*KEYS.get("phase_tr", [])),
        store.get_number(*KEYS.get("phase_rl", [])),
        store.get_number(*KEYS.get("phase_ll", [])),
    ]
    phase_min = min([value for value in phase_values if value is not None], default=None)
    segment_ecw = [
        store.get_number(*KEYS.get("ecw_tbw_ra", [])),
        store.get_number(*KEYS.get("ecw_tbw_la", [])),
        store.get_number(*KEYS.get("ecw_tbw_tr", [])),
        store.get_number(*KEYS.get("ecw_tbw_rl", [])),
        store.get_number(*KEYS.get("ecw_tbw_ll", [])),
    ]
    segment_fat_pct = {
        "右上肢": store.get_number(*KEYS.get("bfm_ra_pct", [])),
        "左上肢": store.get_number(*KEYS.get("bfm_la_pct", [])),
        "軀幹": store.get_number(*KEYS.get("bfm_trunk_pct", [])),
        "右下肢": store.get_number(*KEYS.get("bfm_rl_pct", [])),
        "左下肢": store.get_number(*KEYS.get("bfm_ll_pct", [])),
    }
    muscle_pairs = [
        ("上肢", store.get_number(*KEYS.get("lean_ra", [])), store.get_number(*KEYS.get("lean_la", []))),
        ("下肢", store.get_number(*KEYS.get("lean_rl", [])), store.get_number(*KEYS.get("lean_ll", []))),
    ]

    if bmi is not None:
        summary.append(f"持續追蹤 BMI {bmi:.1f}，透過均衡飲食及運動維持在標準範圍。")
    if pbf is not None and pbf > 25:
        summary.append("提高阻力訓練與有氧運動頻率，以降低體脂率。")
    elif pbf is not None and pbf < 10:
        summary.append("適度提高總熱量與蛋白質，避免體脂率過低影響免疫與荷爾蒙。")
    if ffm_control is not None and ffm_control > 0:
        summary.append("安排每週 2-3 次全身性重量訓練，並補足每公斤體重 1.6-2.0g 蛋白質。")
    if bfm_control is not None and bfm_control < -0.5:
        summary.append("掌握熱量赤字時維持高纖飲食，搭配 120~150 分鐘中強度有氧以促進減脂。")
    if ffm_control is not None and ffm_control < -0.5:
        summary.append("若需要刻意降低無脂體重，務必在專業指導下循序進行避免代謝下滑。")
    if weight_control is not None and abs(weight_control) >= 0.5:
        summary.append("搭配飲食日誌及每週量測，追蹤體重控制進度。")

    segment_high_ecw = [value for value in segment_ecw if value is not None and value >= 0.39]
    if ecw_tbw is not None and ecw_tbw >= 0.39:
        summary.append("注意鈉攝取與睡眠品質，必要時諮詢專業醫師評估水腫。")
    elif segment_ecw and all(value is not None and 0.36 <= value <= 0.39 for value in segment_ecw if value is not None):
        summary.append("四肢與軀幹 ECW/TBW 均在建議範圍，維持當前水分管理。")
    if segment_high_ecw:
        summary.append("某些部位 ECW/TBW 偏高，留意該側的負荷與循環狀況。")

    if vfa is not None and vfa >= 100:
        summary.append("加入核心訓練與高強度間歇，以降低內臟脂肪風險。")
    if score is not None and score < 80:
        summary.append("整體 InBody 分數仍有改善空間，建議三個月後回測以檢視進步幅度。")
    if vfl is not None and vfl >= 10:
        summary.append("內臟脂肪等級偏高，建議從高纖低糖飲食與規律有氧著手。")
    elif vfl is not None and vfl <= 5:
        summary.append("內臟脂肪等級維持在安全範圍，持續目前飲食節奏。")
    if whr is not None and whr >= 0.90:
        summary.append("腰臀比偏高，改善腹部脂肪堆積有助降低代謝症候群風險。")

    if phase_min is not None and phase_min < 5.5:
        summary.append("相位角偏低，補足蛋白質並確保睡眠可提升細胞活性。")
    elif phase_min is not None and phase_min >= 7.0:
        summary.append("相位角表現良好，代表細胞活性與恢復狀態穩定。")
    phase_clean = [value for value in phase_values if value is not None]
    if phase_clean:
        phase_spread = max(phase_clean) - min(phase_clean)
        if phase_spread >= 1.0:
            summary.append(f"相位角左右最大差異約 {phase_spread:.1f}°，調整姿勢與訓練負荷以維持平衡。")

    high_fat_segments = [label for label, value in segment_fat_pct.items() if value is not None and value >= 130]
    if high_fat_segments:
        summary.append("脂肪分佈以 " + "、".join(high_fat_segments) + " 為主，建議加入局部肌力搭配有氧訓練。")
    low_fat_segments = [label for label, value in segment_fat_pct.items() if value is not None and value <= 80]
    if low_fat_segments:
        summary.append("、".join(low_fat_segments) + " 脂肪百分比偏低，確保攝取足夠能量避免過度消耗。")

    imbalances: List[str] = []
    for label, value_a, value_b in muscle_pairs:
        if value_a is None or value_b is None:
            continue
        average = (value_a + value_b) / 2
        if average == 0:
            continue
        diff_pct = abs(value_a - value_b) / average * 100
        if diff_pct >= 10:
            imbalances.append(f"{label}左右肌肉差距約 {diff_pct:.1f}%")
    if imbalances:
        summary.append("、".join(imbalances) + "，建議安排矯正訓練與單側負重。")
    elif any(value_a is not None and value_b is not None for _, value_a, value_b in muscle_pairs):
        summary.append("四肢肌肉量左右差距都在 10% 內，維持目前訓練即可。")

    if not summary:
        summary.append("維持目前的飲食與訓練節奏，定期複測以掌握趨勢。")
    return summary


def build_appendix_notes(store: MetricStore) -> List[str]:
    smm = store.get_number(*KEYS.get("smm", []))
    bfm = store.get_number(*KEYS.get("bfm", []))
    weight = store.get_number(*KEYS.get("weight_kg", []))
    vfa = store.get_number(*KEYS.get("vfa", []))
    vfl = store.get_number(*KEYS.get("vfl", []))
    trunk_phase = store.get_number(*KEYS.get("phase_tr", []))

    notes: List[str] = []
    notes.append(
        "**C 型輪廓**：InBody 透過五大柱狀圖比較肌肉與脂肪；當肌肉柱顯著低於脂肪柱時即稱 C 型，代表肌肉支撐不足且脂肪偏高，必須同步強化阻力訓練與熱量管理。"
    )
    notes.append(
        "**內臟脂肪分級**：<50 cm² 為安全、50-70 cm² 為警戒、70-100 cm² 為高風險、≥100 cm² 屬極高風險；若 VFL ≥10 則需醫療團隊密切追蹤。"
    )
    notes.append(
        "**相位角判讀**：5.5° 以下視為偏低、5.5-7.5° 為穩定範圍、7.5° 以上代表細胞活性佳；相位角受睡眠、營養與訓練恢復影響。"
    )
    if vfa is not None or vfl is not None:
        notes.append(
            f"**本次內臟脂肪**：面積 {format_number(vfa, ' cm²')}、等級 {format_number(vfl)}，建議透過核心訓練與控糖飲食持續下降。"
        )
    if trunk_phase is not None:
        notes.append(
            f"**本次相位角**：軀幹 {format_number(trunk_phase, '°')}，可作為細胞恢復與營養補充成效的追蹤指標。"
        )
    if smm is not None and bfm is not None:
        clause = "，顯示肌肉柱略短於脂肪柱" if weight is not None and weight - smm > 20 else ""
        notes.append(
            f"**肌脂對照**：骨骼肌量 {format_number(smm, ' kg')} / 體脂肪量 {format_number(bfm, ' kg')}{clause}，為評估 C 型改善幅度的重要指標。"
        )
    return notes


def default_input_path(base: Path) -> Optional[Path]:
    search_dirs = [
        base,
        base / "data",
        base / "data" / "inbody_clean",
    ]
    filenames = ("inbody_summary.json", "inbody_summary.csv")
    for directory in search_dirs:
        for name in filenames:
            path = directory / name
            if path.exists():
                return path
    return None


def run(
    input_path: Path,
    output_path: Optional[Path],
    *,
    use_gpt: bool = False,
    reference_path: Optional[Path] = None,
    model: str = DEFAULT_GPT_MODEL,
    temperature: Optional[float] = 0.3,
) -> Path:
    store = load_metrics(input_path)
    reference_base = reference_path or REFERENCE_DEFAULT
    reference_sections = load_reference_sections(reference_base)
    reference_index = load_reference_index(reference_base)
    gpt_text: Optional[str] = None
    if use_gpt:
        candidate_models: List[str] = []
        if model:
            candidate_models.append(model)
        if FALLBACK_GPT_MODEL and FALLBACK_GPT_MODEL not in candidate_models:
            candidate_models.append(FALLBACK_GPT_MODEL)
        last_error: Optional[Exception] = None
        for candidate in candidate_models:
            try:
                gpt_text = generate_gpt_insights(store, reference_sections, candidate, temperature)
                if candidate != model:
                    print(f"[GPT] 主模型 '{model}' 失敗，已改用 '{candidate}'。")
                break
            except Exception as exc:  # pragma: no cover - network/credentials issues
                print(f"[GPT] 無法使用模型 '{candidate}'：{exc}")
                last_error = exc
        if gpt_text is None and last_error is not None:
            print("[GPT] 無法產生個人化分析，改用內建規則式摘要。")
    report = build_report(store, gpt_text, reference_index)
    destination = output_path or input_path.with_name("inbody_final_report.md")
    destination.write_text(report, encoding="utf-8")
    return destination


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a final recommendation report from InBody summary data.")
    parser.add_argument("--input", type=Path, help="Path to inbody_summary.json or inbody_summary.csv")
    parser.add_argument("--output", type=Path, help="Path to write the final Markdown report", default=None)
    parser.add_argument("--no-gpt", action="store_true", help="Disable GPT-5 augmented analysis")
    parser.add_argument("--reference", type=Path, help="Path to reference markdown for RAG", default=None)
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_GPT_MODEL,
        help=f"OpenAI model id (default: {DEFAULT_GPT_MODEL})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature for GPT generation (omit with value -1 to disable)",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    load_dotenv()
    base_dir = Path.cwd()
    input_path: Optional[Path]
    if args.input:
        raw_input = args.input
        input_path = raw_input
        if not input_path.is_absolute():
            input_path = (base_dir / input_path).resolve()
        if not input_path.exists() and not raw_input.is_absolute():
            candidate = (base_dir / "data" / raw_input).resolve()
            if candidate.exists():
                input_path = candidate
    else:
        input_path = default_input_path(base_dir)
    if input_path is None:
        raise SystemExit("找不到輸入檔案，請使用 --input 指定 inbody_summary.json 或 inbody_summary.csv")
    if not input_path.exists():
        raise SystemExit(f"找不到輸入檔案：{input_path}")
    output_path = args.output
    if output_path and not output_path.is_absolute():
        output_path = base_dir / output_path
    reference_path = args.reference
    if reference_path and not reference_path.is_absolute():
        reference_path = (base_dir / reference_path).resolve()
    temperature = None if args.temperature is not None and args.temperature < 0 else args.temperature
    destination = run(
        input_path,
        output_path,
        use_gpt=not args.no_gpt,
        reference_path=reference_path,
        model=args.model,
        temperature=temperature,
    )
    print(f"Final report written to {destination}")


if __name__ == "__main__":
    main()
