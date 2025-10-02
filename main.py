"""CLI entry point for generating normalized InBody summaries."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

from inbody_processing import DEFAULT_ENCODINGS, process_inbody_file


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract key metrics from an InBody CSV export.")
    parser.add_argument("input", type=Path, help="Path to the raw InBody CSV file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to store the generated summary files (default: <input parent>/inbody_clean)",
    )
    parser.add_argument(
        "--encoding",
        nargs="+",
        help="Override the encoding detection order (e.g. --encoding utf-8 big5)",
    )
    return parser.parse_args(argv)


def resolve_output_dir(input_path: Path, output_dir: Optional[Path]) -> Path:
    if output_dir is not None:
        if not output_dir.is_absolute():
            return (Path.cwd() / output_dir).resolve()
        return output_dir
    return (input_path.parent / "inbody_clean").resolve()


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    input_path = args.input
    if not input_path.is_absolute():
        input_path = (Path.cwd() / input_path).resolve()
    if not input_path.exists():
        raise SystemExit(f"找不到輸入檔案：{input_path}")

    output_dir = resolve_output_dir(input_path, args.output_dir)
    encodings: Tuple[str, ...] = tuple(args.encoding) if args.encoding else DEFAULT_ENCODINGS
    outputs = process_inbody_file(input_path, output_dir, encodings=encodings)

    print("已產出：")
    print(f"- 整理後 CSV：{outputs['csv']}")
    print(f"- 整理後 JSON：{outputs['json']}")
    print(f"- Markdown 報告：{outputs['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
