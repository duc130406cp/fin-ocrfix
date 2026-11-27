#!/usr/bin/env python3
"""R3 · Phân tích kết quả, thống kê 5 nhãn lỗi OCR, trích xuất ví dụ điển hình & vẽ biểu đồ RQ1–RQ4.

Tuân thủ đặc tả Plan4project Phần A4, A5, A6, B4, C5, C6, C8:
- Đọc bảng tổng hợp results/summary.csv (của R5) làm nguồn số liệu chuẩn; tự động chuẩn hoá tên cột
  và đơn vị phần trăm để tương thích linh hoạt với evaluate.py.
- Kết hợp đọc results/predictions/*.jsonl, data/{vi,en}_test.jsonl (và tùy chọn results/metrics/*.json,
  data/terms.csv) để:
  1. Bổ sung thống kê chi tiết theo 5 nhãn lỗi OCR (DAU, KT, SO, TACH, MAT) trên 100 dòng gắn nhãn.
  2. Vẽ biểu đồ phân tán CER từng dòng trước/sau sửa (theo phong cách No Free Lunches, 2025).
  3. Trích xuất danh sách ví dụ lỗi điển hình đa mô hình (typical_examples.md & .json) cho báo cáo/slide.
  4. Tự tính dự phòng các metric chuẩn C5 nếu chạy thử trước khi có summary.csv đầy đủ.
- Xuất toàn bộ biểu đồ ở cả 2 định dạng: .png (300 DPI cho slide/README) và .pdf (vector cho Overleaf).

Ví dụ chạy chuẩn (C6):
    python src/analyze.py --summary results/summary.csv --pred_dir results/predictions/ --out results/figures/
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    HAS_DATA_STACK = True
except ImportError:  # Cho phép import các hàm xử lý chuỗi/metric thuần Python trên môi trường tối giản
    matplotlib = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    pd = None  # type: ignore[assignment]
    HAS_DATA_STACK = False

ERROR_LABELS = ("DAU", "KT", "SO", "TACH", "MAT")
ERROR_LABEL_DESCRIPTIONS = {
    "DAU": "Sai/mất dấu (DAU)",
    "KT": "Nhầm ký tự chữ (KT)",
    "SO": "Sai ký tự số/đơn vị (SO)",
    "TACH": "Tách/gộp từ (TACH)",
    "MAT": "Mất/thừa ký tự, từ (MAT)",
}

NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*%?")

# Ánh xạ linh hoạt các biến thể tên cột có thể xuất hiện trong summary.csv của R5
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "model": ("model", "model_name", "system", "method", "checkpoint", "model_id", "run"),
    "lang": ("lang", "language"),
    "noise": ("noise", "noise_level"),
    "prompt_variant": ("prompt_variant", "prompt", "variant"),
    "cer": ("cer", "pred_cer", "cer_pred", "cer_prediction"),
    "wer": ("wer", "pred_wer", "wer_pred", "wer_prediction"),
    "cer_source": ("cer_source", "source_cer", "baseline_cer", "orig_cer"),
    "wer_source": ("wer_source", "source_wer", "baseline_wer", "orig_wer"),
    "cer_improvement": (
        "cer_improvement",
        "cer_reduction",
        "improvement",
        "rel_cer_improvement",
        "cer_imp",
    ),
    "worse_rate": (
        "worse_rate",
        "degraded_rate",
        "overcorrection_rate",
        "over_correction_rate",
        "worsened_rate",
        "lines_worse_rate",
        "pct_worse",
    ),
    "clean_corrupted_rate": (
        "clean_corrupted_rate",
        "corrupt_clean_rate",
        "broken_clean_rate",
        "damaged_correct_rate",
        "clean_damage_rate",
        "false_alarm_rate",
    ),
    "number_preservation": (
        "number_preservation",
        "num_preservation",
        "number_preserve_rate",
        "num_preserve_rate",
        "preserve_number",
        "number_exact_rate",
    ),
    "broken_valid_numbers": (
        "broken_valid_numbers",
        "corrupted_source_numbers",
        "damaged_valid_numbers",
        "broken_numbers_count",
    ),
    "term_preservation": (
        "term_preservation",
        "terminology_preservation",
        "term_preserve_rate",
        "preserve_term",
    ),
    "diacritic_error_rate": (
        "diacritic_error_rate",
        "tone_error_rate",
        "accent_error_rate",
        "dau_error_rate",
        "diacritic_err",
    ),
    "fix_DAU": ("fix_dau", "dau_fix_rate", "err_dau", "exact_dau", "dau"),
    "fix_KT": ("fix_kt", "kt_fix_rate", "err_kt", "exact_kt", "kt"),
    "fix_SO": ("fix_so", "so_fix_rate", "err_so", "exact_so", "so"),
    "fix_TACH": ("fix_tach", "tach_fix_rate", "err_tach", "exact_tach", "tach"),
    "fix_MAT": ("fix_mat", "mat_fix_rate", "err_mat", "exact_mat", "mat"),
}

RATE_COLUMNS = (
    "cer",
    "wer",
    "cer_source",
    "wer_source",
    "cer_improvement",
    "worse_rate",
    "clean_corrupted_rate",
    "number_preservation",
    "term_preservation",
    "diacritic_error_rate",
    "fix_DAU",
    "fix_KT",
    "fix_SO",
    "fix_TACH",
    "fix_MAT",
)

MODEL_DISPLAY_ORDER = [
    "baseline",
    "bartpho-v2",
    "vn-spell-small",
    "coedit-large",
    "qwen3.5-2b-basic",
    "qwen3.5-2b-strict",
    "qwen3.5-4b-basic",
    "qwen3.5-4b-strict",
    "qwen2.5-1.5b-basic",
    "qwen2.5-1.5b-strict",
]

MODEL_DISPLAY_NAMES = {
    "baseline": "Baseline (OCR gốc)",
    "bartpho-v2": "BARTpho-v2",
    "vn-spell-small": "VN-Spell-Small",
    "coedit-large": "CoEdIT-Large",
    "qwen3.5-2b-basic": "Qwen3.5-2B (basic)",
    "qwen3.5-2b-strict": "Qwen3.5-2B (strict)",
    "qwen3.5-4b-basic": "Qwen3.5-4B (basic)",
    "qwen3.5-4b-strict": "Qwen3.5-4B (strict)",
    "qwen2.5-1.5b-basic": "Qwen2.5-1.5B (basic)",
    "qwen2.5-1.5b-strict": "Qwen2.5-1.5B (strict)",
}

MODEL_COLORS = {
    "baseline": "#64748B",          # Slate gray
    "bartpho-v2": "#0EA5E9",        # Sky blue
    "vn-spell-small": "#14B8A6",    # Teal
    "coedit-large": "#2563EB",      # Royal blue
    "qwen3.5-2b-basic": "#F59E0B",  # Amber
    "qwen3.5-2b-strict": "#7C3AED", # Violet
    "qwen3.5-4b-basic": "#EA580C",  # Orange
    "qwen3.5-4b-strict": "#4F46E5", # Indigo
    "qwen2.5-1.5b-basic": "#D97706",
    "qwen2.5-1.5b-strict": "#6D28D9",
}


def configure_matplotlib_style() -> None:
    """Cấu hình font hỗ trợ đầy đủ tiếng Việt có dấu và xuất vector PDF chuẩn LaTeX."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial", "Tahoma"],
            "axes.unicode_minus": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "axes.axisbelow": True,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    """Lưu đồng thời ảnh .png (300 DPI) và .pdf (vector cho báo cáo Overleaf)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


# ==============================================================================
# 1. Chuẩn hoá văn bản & các hàm tính Metric dự phòng đúng đặc tả C5
# ==============================================================================


def normalize_eval_text(text: str) -> str:
    """Chuẩn hoá trước khi chấm theo C5: Unicode NFC + gộp khoảng trắng thừa."""
    nfc = unicodedata.normalize("NFC", str(text))
    return " ".join(nfc.split())


def edit_distance_seq(seq_a: list[Any] | str, seq_b: list[Any] | str) -> int:
    """Khoảng cách Levenshtein chuẩn (Wagner–Fischer) dùng làm fallback khi thiếu jiwer."""
    n, m = len(seq_a), len(seq_b)
    if n == 0:
        return m
    if m == 0:
        return n
    if n < m:
        seq_a, seq_b = seq_b, seq_a
        n, m = m, n

    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        ca = seq_a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ca == seq_b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


def compute_line_cer(prediction: str, target: str) -> float:
    pred_norm = normalize_eval_text(prediction)
    tgt_norm = normalize_eval_text(target)
    if not tgt_norm:
        return 0.0 if not pred_norm else 1.0
    try:
        import jiwer  # type: ignore

        return float(jiwer.cer(tgt_norm, pred_norm))
    except ImportError:
        return edit_distance_seq(pred_norm, tgt_norm) / len(tgt_norm)


def compute_corpus_cer_wer(predictions: list[str], targets: list[str]) -> tuple[float, float]:
    preds = [normalize_eval_text(p) for p in predictions]
    tgts = [normalize_eval_text(t) for t in targets]
    non_empty_pairs = [(p, t) for p, t in zip(preds, tgts) if t]
    if not non_empty_pairs:
        return 0.0, 0.0

    preds_clean = [p for p, _ in non_empty_pairs]
    tgts_clean = [t for _, t in non_empty_pairs]

    try:
        import jiwer  # type: ignore

        return float(jiwer.cer(tgts_clean, preds_clean)), float(
            jiwer.wer(tgts_clean, preds_clean)
        )
    except ImportError:
        char_edits = sum(edit_distance_seq(p, t) for p, t in non_empty_pairs)
        total_chars = sum(len(t) for _, t in non_empty_pairs)
        word_edits = sum(edit_distance_seq(p.split(), t.split()) for p, t in non_empty_pairs)
        total_words = sum(len(t.split()) for _, t in non_empty_pairs)
        cer = char_edits / total_chars if total_chars else 0.0
        wer = word_edits / total_words if total_words else 0.0
        return cer, wer


def extract_numbers(text: str) -> list[str]:
    return NUMBER_RE.findall(normalize_eval_text(text))


def strip_vi_diacritics(text: str) -> str:
    """Bỏ dấu thanh và dấu phụ tiếng Việt để lấy chữ gốc phục vụ metric lỗi dấu (C5)."""
    replaced = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", replaced)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", stripped)


def tokenize_vi_syllables(text: str) -> list[str]:
    """Tách âm tiết chữ cái tiếng Việt (bỏ dấu câu ở rìa token)."""
    tokens = normalize_eval_text(text).split()
    cleaned = [tok.strip(".,;:!?()[]{}\"'“”‘’/-–—") for tok in tokens]
    return [tok for tok in cleaned if tok and any(ch.isalpha() for ch in tok)]


def compute_vi_diacritic_error_rate(predictions: list[str], targets: list[str]) -> float:
    """Tính tỷ lệ lỗi dấu tiếng Việt theo đúng C5 dùng difflib.SequenceMatcher."""
    matched_base_pairs = 0
    diff_diacritic_pairs = 0

    for pred, tgt in zip(predictions, targets):
        tgt_syls = tokenize_vi_syllables(tgt)
        pred_syls = tokenize_vi_syllables(pred)
        tgt_bases = [strip_vi_diacritics(s).lower() for s in tgt_syls]
        pred_bases = [strip_vi_diacritics(s).lower() for s in pred_syls]

        matcher = difflib.SequenceMatcher(None, tgt_bases, pred_bases)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for t_tok, p_tok in zip(tgt_syls[i1:i2], pred_syls[j1:j2]):
                    matched_base_pairs += 1
                    if t_tok.lower() != p_tok.lower():
                        diff_diacritic_pairs += 1

    if matched_base_pairs == 0:
        return 0.0
    return diff_diacritic_pairs / matched_base_pairs


def load_terms_by_lang(terms_path: Path | None) -> dict[str, list[str]]:
    """Đọc danh sách thuật ngữ tài chính data/terms.csv (nếu đã có)."""
    terms: dict[str, list[str]] = {"vi": [], "en": []}
    if terms_path is None or not terms_path.exists():
        return terms

    with terms_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return terms
        field_map = {name.strip().lower(): name for name in reader.fieldnames}
        term_col = (
            field_map.get("term")
            or field_map.get("thuat_ngu")
            or field_map.get("thuật ngữ")
            or reader.fieldnames[0]
        )
        lang_col = (
            field_map.get("lang")
            or field_map.get("language")
            or field_map.get("ngon_ngu")
            or field_map.get("ngôn ngữ")
        )
        for row in reader:
            raw_term = normalize_eval_text(row.get(term_col, ""))
            if not raw_term:
                continue
            raw_lang = str(row.get(lang_col, "") if lang_col else "").strip().lower()
            if raw_lang in ("vi", "vie", "vietnamese"):
                terms["vi"].append(raw_term)
            elif raw_lang in ("en", "eng", "english"):
                terms["en"].append(raw_term)
            else:
                terms["vi"].append(raw_term)
                terms["en"].append(raw_term)
    return terms


def compute_term_preservation(
    predictions: list[str],
    targets: list[str],
    lang_terms: list[str],
) -> float:
    """Bảo toàn thuật ngữ theo C5: kiểm tra xuất hiện nguyên văn (chuẩn NFC) trong prediction."""
    if not lang_terms:
        return float("nan")

    total_hits = 0
    preserved_hits = 0
    for pred, tgt in zip(predictions, targets):
        tgt_norm = normalize_eval_text(tgt)
        pred_norm = normalize_eval_text(pred)
        for term in lang_terms:
            term_norm = normalize_eval_text(term)
            if not term_norm:
                continue
            count_in_tgt = tgt_norm.count(term_norm)
            if count_in_tgt > 0:
                total_hits += count_in_tgt
                preserved_hits += min(count_in_tgt, pred_norm.count(term_norm))

    if total_hits == 0:
        return float("nan")
    return preserved_hits / total_hits


def compute_metrics_for_run(
    gold_rows: list[dict[str, Any]],
    pred_map: dict[str, str],
    lang: str,
    lang_terms: list[str],
) -> dict[str, Any]:
    """Tính bộ metric tham chiếu theo C5 (chỉ dùng khi chạy với --allow_fallback_metrics)."""
    sources = [str(r["source"]) for r in gold_rows]
    targets = [str(r["target"]) for r in gold_rows]
    preds = [pred_map.get(str(r["id"]), str(r["source"])) for r in gold_rows]

    cer_pred, wer_pred = compute_corpus_cer_wer(preds, targets)
    cer_src, wer_src = compute_corpus_cer_wer(sources, targets)
    cer_imp = (cer_src - cer_pred) / cer_src if cer_src > 0 else 0.0

    worse_count = 0
    clean_total = 0
    clean_corrupted = 0
    num_total = 0
    num_preserved = 0
    broken_valid_nums = 0

    label_totals: Counter[str] = Counter()
    label_fixed: Counter[str] = Counter()

    for rec, src, tgt, pred in zip(gold_rows, sources, targets, preds):
        src_norm = normalize_eval_text(src)
        tgt_norm = normalize_eval_text(tgt)
        pred_norm = normalize_eval_text(pred)

        if compute_line_cer(pred_norm, tgt_norm) > compute_line_cer(src_norm, tgt_norm) + 1e-12:
            worse_count += 1

        if src_norm == tgt_norm:
            clean_total += 1
            if pred_norm != tgt_norm:
                clean_corrupted += 1

        # Đúng đặc tả C5: chỉ xét các dòng has_number và so sánh tập số (set)
        if bool(rec.get("has_number")):
            tgt_nums = set(extract_numbers(tgt_norm))
            if tgt_nums:
                num_total += 1
                pred_nums = set(extract_numbers(pred_norm))
                src_nums = set(extract_numbers(src_norm))
                if pred_nums == tgt_nums:
                    num_preserved += 1
                valid_in_src = src_nums & tgt_nums
                if not valid_in_src.issubset(pred_nums):
                    broken_valid_nums += 1

        err_types = rec.get("error_types") or []
        if isinstance(err_types, str):
            err_types = [e.strip() for e in err_types.split(",") if e.strip()]
        for lbl in err_types:
            lbl_up = str(lbl).strip().upper()
            if lbl_up in ERROR_LABELS:
                label_totals[lbl_up] += 1
                if pred_norm == tgt_norm:
                    label_fixed[lbl_up] += 1

    n_lines = len(gold_rows)
    result: dict[str, Any] = {
        "cer": cer_pred * 100.0,
        "wer": wer_pred * 100.0,
        "cer_source": cer_src * 100.0,
        "wer_source": wer_src * 100.0,
        "cer_improvement": cer_imp * 100.0,
        "worse_rate": (worse_count / n_lines * 100.0) if n_lines else 0.0,
        "clean_corrupted_rate": (clean_corrupted / clean_total * 100.0)
        if clean_total
        else float("nan"),
        "number_preservation": (num_preserved / num_total * 100.0) if num_total else float("nan"),
        "broken_valid_numbers": broken_valid_nums,
        "term_preservation": compute_term_preservation(preds, targets, lang_terms) * 100.0,
        "diacritic_error_rate": (
            compute_vi_diacritic_error_rate(preds, targets) * 100.0
            if lang == "vi"
            else float("nan")
        ),
    }
    for lbl in ERROR_LABELS:
        result[f"fix_{lbl}"] = (
            (label_fixed[lbl] / label_totals[lbl] * 100.0)
            if label_totals[lbl] > 0
            else float("nan")
        )
    return result


# ==============================================================================
# 2. Đọc dữ liệu, chuẩn hoá tên mô hình & hợp nhất bảng tổng hợp
# ==============================================================================


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Lỗi JSON ở {path}:{line_no}: {exc}") from exc
    return rows


def canonicalize_model_key(raw_model: str, prompt_variant: str | None = None) -> str:
    """Đưa các cách viết tên model/checkpoint khác nhau về một khóa chuẩn duy nhất."""
    m = str(raw_model or "").strip().lower()
    m = m.split("/")[-1]
    p = str(prompt_variant or "").strip().lower()

    if m in ("baseline", "source", "none", "no_correction", "identity", "ocr"):
        return "baseline"
    if "vietnamese-correction" in m or "bartpho" in m and "spell" not in m:
        return "bartpho-v2"
    if "vn-spell-correction" in m or "spell-correction" in m or "vn-spell" in m:
        return "vn-spell-small"
    if "coedit" in m:
        return "coedit-large"

    variant_suffix = ""
    if m.endswith("-basic") or m.endswith("_basic") or p == "basic":
        variant_suffix = "-basic"
        m = re.sub(r"[-_]basic$", "", m)
    elif m.endswith("-strict") or m.endswith("_strict") or p == "strict":
        variant_suffix = "-strict"
        m = re.sub(r"[-_]strict$", "", m)

    if "qwen3.5-2b" in m or "qwen3_5-2b" in m:
        return f"qwen3.5-2b{variant_suffix or '-basic'}"
    if "qwen3.5-4b" in m or "qwen3_5-4b" in m:
        return f"qwen3.5-4b{variant_suffix or '-basic'}"
    if "qwen2.5-1.5b" in m:
        return f"qwen2.5-1.5b{variant_suffix or '-basic'}"

    return f"{m}{variant_suffix}" if variant_suffix and not m.endswith(variant_suffix) else m


def display_model_name(model_key: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_key, model_key)


def model_sort_rank(model_key: str) -> tuple[int, str]:
    if model_key in MODEL_DISPLAY_ORDER:
        return (MODEL_DISPLAY_ORDER.index(model_key), model_key)
    return (len(MODEL_DISPLAY_ORDER), model_key)


def model_color(model_key: str, fallback_idx: int = 0) -> str:
    if model_key in MODEL_COLORS:
        return MODEL_COLORS[model_key]
    palette = ["#0284C7", "#059669", "#D97706", "#DC2626", "#9333EA", "#475569"]
    return palette[fallback_idx % len(palette)]


def parse_prediction_filename(path: Path) -> tuple[str, str, str] | None:
    """Tách {model}_{lang}_{noise}.jsonl -> (model_key, lang, noise)."""
    stem = path.stem
    match = re.match(r"^(.+)_(vi|en)_(light|heavy)$", stem, flags=re.IGNORECASE)
    if not match:
        return None
    raw_model, lang, noise = match.group(1), match.group(2).lower(), match.group(3).lower()
    return canonicalize_model_key(raw_model), lang, noise


def detect_summary_is_ratio_scale(df: pd.DataFrame, explicit_pct_cols: set[str]) -> bool:
    """Xác định xem bảng summary.csv đang lưu các chỉ số ở thang tỷ lệ [0, 1] hay phần trăm [0, 100].

    Kiểm tra đồng thời trên toàn bảng để tránh lỗi nhân nhầm 100 lần cho một cột có giá trị phần trăm
    nhỏ (ví dụ clean_corrupted_rate = 0.5% trong một bảng vốn đã ở thang 0..100).
    """
    probe_cols = [
        c
        for c in (
            "number_preservation",
            "term_preservation",
            "wer",
            "cer",
            "cer_source",
            "wer_source",
            "worse_rate",
            "clean_corrupted_rate",
            "diacritic_error_rate",
            "fix_DAU",
            "fix_KT",
            "fix_SO",
            "fix_TACH",
            "fix_MAT",
        )
        if c in df.columns and c not in explicit_pct_cols
    ]
    all_vals: list[float] = []
    for col in probe_cols:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        all_vals.extend(float(abs(v)) for v in vals)

    if not all_vals:
        return False
    # Nếu bất kỳ cột tỷ lệ nào vượt quá 1.5 (vd CER=6.2, WER=18.5, bảo toàn số=94.0) -> bảng đã ở thang %
    if max(all_vals) > 1.5:
        return False
    return max(all_vals) > 0.0


def load_and_standardize_summary(summary_path: Path | None) -> pd.DataFrame:
    """Đọc summary.csv của R5 và chuẩn hoá về schema cột thống nhất (đơn vị %)."""
    if summary_path is None or not summary_path.exists():
        return pd.DataFrame()

    raw_df = pd.read_csv(summary_path)
    if raw_df.empty:
        return pd.DataFrame()

    lower_to_orig = {col.strip().lower(): col for col in raw_df.columns}
    std: dict[str, Any] = {}
    explicit_pct_cols: set[str] = set()

    for canonical, aliases in COLUMN_ALIASES.items():
        found_col = None
        is_explicit_pct = False
        for alias in aliases:
            if alias.lower() in lower_to_orig:
                found_col = lower_to_orig[alias.lower()]
                is_explicit_pct = "%" in found_col or found_col.lower().endswith("_pct")
                break
            if f"{alias.lower()}_pct" in lower_to_orig:
                found_col = lower_to_orig[f"{alias.lower()}_pct"]
                is_explicit_pct = True
                break
        if found_col is not None:
            std[canonical] = raw_df[found_col]
            if is_explicit_pct:
                explicit_pct_cols.add(canonical)

    df = pd.DataFrame(std)
    if "model" not in df.columns:
        raise ValueError(
            f"File {summary_path} thiếu cột định danh mô hình (model / model_name / system)."
        )

    # Tách trường hợp cột model gộp sẵn _{lang}_{noise}
    if "lang" not in df.columns or "noise" not in df.columns:
        parsed_langs = []
        parsed_noises = []
        parsed_models = []
        for val in df["model"].astype(str):
            m = re.match(r"^(.+)_(vi|en)_(light|heavy)$", val.strip(), flags=re.IGNORECASE)
            if m:
                parsed_models.append(m.group(1))
                parsed_langs.append(m.group(2).lower())
                parsed_noises.append(m.group(3).lower())
            else:
                parsed_models.append(val)
                parsed_langs.append("")
                parsed_noises.append("")
        df["model"] = parsed_models
        if "lang" not in df.columns:
            df["lang"] = parsed_langs
        if "noise" not in df.columns:
            df["noise"] = parsed_noises

    prompt_col = df["prompt_variant"] if "prompt_variant" in df.columns else [None] * len(df)
    df["model_key"] = [
        canonicalize_model_key(m, p) for m, p in zip(df["model"], prompt_col)
    ]
    df["lang"] = df["lang"].astype(str).str.strip().str.lower()
    df["noise"] = df["noise"].astype(str).str.strip().str.lower()

    for col in RATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    is_ratio_scale = detect_summary_is_ratio_scale(df, explicit_pct_cols)
    if is_ratio_scale:
        for col in RATE_COLUMNS:
            if col in df.columns and col not in explicit_pct_cols:
                df[col] = df[col] * 100.0

    # Nếu chưa có dòng baseline riêng nhưng có cer_source/wer_source, tự bổ sung dòng baseline để vẽ biểu đồ đối chứng
    existing_keys = set(zip(df["model_key"], df["lang"], df["noise"]))
    extra_baselines: list[dict[str, Any]] = []
    if "cer_source" in df.columns:
        for (lang, noise), grp in df.groupby(["lang", "noise"]):
            if not lang or not noise or ("baseline", lang, noise) in existing_keys:
                continue
            cer_s = grp["cer_source"].dropna()
            wer_s = grp["wer_source"].dropna() if "wer_source" in grp.columns else pd.Series(dtype=float)
            if not cer_s.empty:
                extra_baselines.append(
                    {
                        "model": "baseline",
                        "model_key": "baseline",
                        "lang": lang,
                        "noise": noise,
                        "cer": float(cer_s.iloc[0]),
                        "wer": float(wer_s.iloc[0]) if not wer_s.empty else float("nan"),
                        "cer_improvement": 0.0,
                        "worse_rate": 0.0,
                        "clean_corrupted_rate": 0.0,
                    }
                )
    if extra_baselines:
        df = pd.concat([pd.DataFrame(extra_baselines), df], ignore_index=True)

    return df


def load_gold_datasets(data_dir: Path | None) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Đọc data/{vi,en}_test.jsonl và tách theo (lang, noise)."""
    gold_by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    if data_dir is None or not data_dir.exists():
        return {}
    for fname, default_lang in (("vi_test.jsonl", "vi"), ("en_test.jsonl", "en")):
        fpath = data_dir / fname
        for rec in read_jsonl(fpath):
            lang = str(rec.get("lang") or default_lang).lower()
            noise = str(rec.get("noise") or "light").lower()
            gold_by_group[(lang, noise)].append(rec)
    return dict(gold_by_group)


def load_predictions_dir(pred_dir: Path | None) -> dict[tuple[str, str, str], dict[str, str]]:
    """Đọc mọi file {model}_{lang}_{noise}.jsonl trong pred_dir -> {(model_key, lang, noise): {id: pred}}."""
    preds: dict[tuple[str, str, str], dict[str, str]] = {}
    if pred_dir is None or not pred_dir.exists():
        return preds
    for path in sorted(pred_dir.glob("*.jsonl")):
        parsed = parse_prediction_filename(path)
        if parsed is None:
            continue
        id_map: dict[str, str] = {}
        for rec in read_jsonl(path):
            rec_id = str(rec.get("id", ""))
            if rec_id:
                id_map[rec_id] = str(rec.get("prediction", ""))
        if id_map:
            preds[parsed] = id_map
    return preds


def enrich_summary_with_predictions(
    summary_df: pd.DataFrame,
    gold_by_group: dict[tuple[str, str], list[dict[str, Any]]],
    preds_by_run: dict[tuple[str, str, str], dict[str, str]],
    terms_by_lang: dict[str, list[str]],
    allow_fallback_metrics: bool = False,
) -> pd.DataFrame:
    """Ưu tiên tuyệt đối results/summary.csv của R5 (theo nguyên tắc A6).

    - Khi summary.csv đã có dữ liệu: giữ nguyên 100% các con số tổng hợp từ R5, tuyệt đối không
      âm thầm trộn metric tự tính vào bảng biểu đồ.
    - Chỉ khi chưa có summary.csv VÀ người dùng bật --allow_fallback_metrics (để chạy nháp sớm)
      mới tính bộ metric tham chiếu trực tiếp từ gold + predictions.
    """
    if not summary_df.empty:
        return summary_df.copy()

    if not allow_fallback_metrics:
        return pd.DataFrame()

    print(
        "[R3][Cảnh báo] Chưa có results/summary.csv từ R5 — đang tính metric nháp bằng "
        "--allow_fallback_metrics (chỉ dùng xem trước, báo cáo chính thức phải dùng summary.csv của R5)."
    )
    computed_rows: list[dict[str, Any]] = []
    for (lang, noise), gold_rows in sorted(gold_by_group.items()):
        if not gold_rows:
            continue
        baseline_map = {str(r["id"]): str(r["source"]) for r in gold_rows}
        runs_for_group: list[tuple[str, dict[str, str]]] = [("baseline", baseline_map)]
        for (m_key, p_lang, p_noise), p_map in sorted(preds_by_run.items()):
            if p_lang == lang and p_noise == noise and m_key != "baseline":
                runs_for_group.append((m_key, p_map))

        for m_key, p_map in runs_for_group:
            metrics = compute_metrics_for_run(
                gold_rows=gold_rows,
                pred_map=p_map,
                lang=lang,
                lang_terms=terms_by_lang.get(lang, []),
            )
            metrics.update(
                {
                    "model": m_key,
                    "model_key": m_key,
                    "lang": lang,
                    "noise": noise,
                }
            )
            computed_rows.append(metrics)

    return pd.DataFrame(computed_rows)


# ==============================================================================
# 3. Các hàm vẽ biểu đồ RQ1 – RQ4 & Thống kê 5 nhãn lỗi OCR
# ==============================================================================


def _annotate_bars(ax: plt.Axes, bars, fmt: str = "{:.1f}", fontsize: int = 8) -> None:
    for bar in bars:
        height = bar.get_height()
        if math.isnan(height):
            continue
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 2.5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def plot_rq1_cer_wer(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    """RQ1: So sánh CER (%) và WER (%) theo ngôn ngữ (VI, EN) và mức nhiễu (Light, Heavy)."""
    if df.empty or "cer" not in df.columns:
        return []

    langs = [lg for lg in ("vi", "en") if lg in set(df["lang"])]
    if not langs:
        return []

    has_wer = "wer" in df.columns and df["wer"].notna().any()
    metrics = [("cer", "CER (%) ↓"), ("wer", "WER (%) ↓")] if has_wer else [("cer", "CER (%) ↓")]

    fig, axes = plt.subplots(
        len(langs),
        len(metrics),
        figsize=(6.4 * len(metrics), 4.3 * len(langs)),
        squeeze=False,
    )

    lang_titles = {"vi": "Tiếng Việt (VI)", "en": "Tiếng Anh (EN)"}
    noises = ["light", "heavy"]
    noise_labels = ["Nhiễu nhẹ (Light)", "Nhiễu nặng (Heavy)"]

    for r_idx, lang in enumerate(langs):
        sub_lang = df[df["lang"] == lang].copy()
        models = sorted(sub_lang["model_key"].unique(), key=model_sort_rank)
        n_models = max(1, len(models))
        width = min(0.16, 0.78 / n_models)
        x = np.arange(len(noises))

        for c_idx, (m_col, m_label) in enumerate(metrics):
            ax = axes[r_idx][c_idx]
            for m_i, m_key in enumerate(models):
                vals = []
                for ns in noises:
                    cell = sub_lang[(sub_lang["model_key"] == m_key) & (sub_lang["noise"] == ns)]
                    vals.append(
                        float(cell[m_col].iloc[0])
                        if not cell.empty and pd.notna(cell[m_col].iloc[0])
                        else float("nan")
                    )
                offset = (m_i - (n_models - 1) / 2) * width
                bars = ax.bar(
                    x + offset,
                    vals,
                    width=width * 0.92,
                    label=display_model_name(m_key),
                    color=model_color(m_key, m_i),
                    edgecolor="white",
                    linewidth=0.6,
                )
                _annotate_bars(ax, bars, fmt="{:.1f}", fontsize=7.5)

            ax.set_xticks(x)
            ax.set_xticklabels(noise_labels)
            ax.set_ylabel(m_label)
            ax.set_title(f"RQ1 · {lang_titles.get(lang, lang.upper())} — {m_label}")
            ymax = sub_lang[m_col].dropna().max() if m_col in sub_lang.columns else 10.0
            if pd.notna(ymax) and ymax > 0:
                ax.set_ylim(0, float(ymax) * 1.22)
            ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper left")

    fig.suptitle(
        "RQ1: So sánh tỷ lệ lỗi ký tự (CER) và lỗi từ (WER) giữa Seq2Seq, LLM và Baseline",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    return save_figure(fig, out_dir, "rq1_cer_wer_comparison")


def plot_rq2_safety_and_preservation(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    """RQ2: Bảo toàn số liệu, bảo toàn thuật ngữ, tỷ lệ dòng tệ hơn & làm hỏng dòng đúng."""
    panel_specs = [
        ("number_preservation", "Bảo toàn số liệu (%) ↑", (0, 112)),
        ("term_preservation", "Bảo toàn thuật ngữ (%) ↑", (0, 112)),
        ("worse_rate", "Tỷ lệ dòng bị làm tệ hơn / Sửa quá tay (%) ↓", None),
        ("clean_corrupted_rate", "Tỷ lệ làm hỏng dòng vốn đã đúng (%) ↓", None),
    ]
    non_baseline = df[df["model_key"] != "baseline"].copy() if not df.empty else pd.DataFrame()
    available_specs = [
        spec
        for spec in panel_specs
        if not non_baseline.empty and spec[0] in non_baseline.columns and non_baseline[spec[0]].notna().any()
    ]
    if not available_specs:
        return []

    groups = [
        (lg, ns)
        for lg in ("vi", "en")
        for ns in ("light", "heavy")
        if not non_baseline[(non_baseline["lang"] == lg) & (non_baseline["noise"] == ns)].empty
    ]
    if not groups:
        return []

    n_panels = len(available_specs)
    ncols = 2 if n_panels > 1 else 1
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.8 * ncols, 4.3 * nrows), squeeze=False)

    models = sorted(non_baseline["model_key"].unique(), key=model_sort_rank)
    n_models = max(1, len(models))
    width = min(0.16, 0.78 / n_models)
    x = np.arange(len(groups))
    group_labels = [f"{lg.upper()}\n({ns})" for lg, ns in groups]

    for idx, (col, title, fixed_ylim) in enumerate(available_specs):
        ax = axes[idx // ncols][idx % ncols]
        for m_i, m_key in enumerate(models):
            vals = []
            for lg, ns in groups:
                cell = non_baseline[
                    (non_baseline["model_key"] == m_key)
                    & (non_baseline["lang"] == lg)
                    & (non_baseline["noise"] == ns)
                ]
                vals.append(
                    float(cell[col].iloc[0])
                    if not cell.empty and pd.notna(cell[col].iloc[0])
                    else float("nan")
                )
            offset = (m_i - (n_models - 1) / 2) * width
            bars = ax.bar(
                x + offset,
                vals,
                width=width * 0.92,
                label=display_model_name(m_key),
                color=model_color(m_key, m_i),
                edgecolor="white",
                linewidth=0.6,
            )
            _annotate_bars(ax, bars, fmt="{:.1f}", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(group_labels)
        ax.set_ylabel("%")
        ax.set_title(f"RQ2 · {title}")
        if fixed_ylim is not None:
            ax.set_ylim(*fixed_ylim)
        else:
            ymax = non_baseline[col].dropna().max()
            ax.set_ylim(0, max(10.0, float(ymax) * 1.25) if pd.notna(ymax) else 25.0)
        if idx == 0:
            ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="best")

    for empty_idx in range(n_panels, nrows * ncols):
        axes[empty_idx // ncols][empty_idx % ncols].set_visible(False)

    fig.suptitle(
        "RQ2: Độ an toàn miền tài chính — Bảo toàn số liệu, thuật ngữ và rủi ro sửa quá tay",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    return save_figure(fig, out_dir, "rq2_safety_and_preservation")


def plot_rq2_cer_scatter(
    gold_by_group: dict[tuple[str, str], list[dict[str, Any]]],
    preds_by_run: dict[tuple[str, str, str], dict[str, str]],
    out_dir: Path,
) -> list[Path]:
    """Biểu đồ phân tán CER từng dòng trước vs sau sửa so với đường y=x (No Free Lunches, 2025)."""
    candidate_runs = [
        (m_key, lg, ns)
        for (m_key, lg, ns) in sorted(
            preds_by_run.keys(), key=lambda item: (item[1], item[2], model_sort_rank(item[0]))
        )
        if m_key != "baseline" and (lg, ns) in gold_by_group
    ]
    if not candidate_runs:
        return []

    # Ưu tiên hiển thị tối đa 6 cấu hình tiêu biểu (Seq2Seq chính & LLM basic/strict ở mức nhiễu heavy/light)
    if len(candidate_runs) > 6:
        heavy_runs = [r for r in candidate_runs if r[2] == "heavy"]
        candidate_runs = (heavy_runs if len(heavy_runs) >= 3 else candidate_runs)[:6]

    n_plots = len(candidate_runs)
    ncols = min(3, n_plots)
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.4 * nrows), squeeze=False)

    for idx, (m_key, lg, ns) in enumerate(candidate_runs):
        ax = axes[idx // ncols][idx % ncols]
        gold_rows = gold_by_group[(lg, ns)]
        p_map = preds_by_run[(m_key, lg, ns)]

        src_cers = []
        pred_cers = []
        colors = []
        improved = worsened = unchanged = 0

        for rec in gold_rows:
            s_cer = min(1.0, compute_line_cer(str(rec["source"]), str(rec["target"]))) * 100.0
            pred_txt = p_map.get(str(rec["id"]), str(rec["source"]))
            p_cer = min(1.0, compute_line_cer(pred_txt, str(rec["target"]))) * 100.0
            src_cers.append(s_cer)
            pred_cers.append(p_cer)
            if p_cer < s_cer - 1e-6:
                colors.append("#16A34A")  # Cải thiện
                improved += 1
            elif p_cer > s_cer + 1e-6:
                colors.append("#DC2626")  # Bị làm tệ hơn
                worsened += 1
            else:
                colors.append("#94A3B8")  # Không đổi
                unchanged += 1

        max_val = max(max(src_cers, default=10.0), max(pred_cers, default=10.0), 10.0) * 1.05
        ax.plot([0, max_val], [0, max_val], color="#1E293B", linestyle="--", linewidth=1.0, label="y = x (không đổi)")
        ax.scatter(src_cers, pred_cers, c=colors, alpha=0.55, s=18, edgecolors="none")
        ax.set_xlim(-1, max_val)
        ax.set_ylim(-1, max_val)
        ax.set_xlabel("CER gốc trước sửa (%)")
        ax.set_ylabel("CER sau khi sửa (%)")
        total = max(1, len(gold_rows))
        ax.set_title(
            f"{display_model_name(m_key)} · {lg.upper()} ({ns})\n"
            f"Tốt lên: {improved / total * 100:.1f}% | Tệ đi: {worsened / total * 100:.1f}%",
            fontsize=9.5,
        )

    for empty_idx in range(n_plots, nrows * ncols):
        axes[empty_idx // ncols][empty_idx % ncols].set_visible(False)

    fig.suptitle(
        "Phân bố CER từng dòng trước và sau hậu xử lý (dưới đường chéo y = x là cải thiện, trên là sửa quá tay)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    return save_figure(fig, out_dir, "rq2_cer_scatter_before_after")


def compute_error_type_table(
    df: pd.DataFrame,
    gold_by_group: dict[tuple[str, str], list[dict[str, Any]]],
    preds_by_run: dict[tuple[str, str, str], dict[str, str]],
) -> pd.DataFrame:
    """Tổng hợp bảng thống kê theo 5 nhãn lỗi OCR (DAU, KT, SO, TACH, MAT)."""
    rows: list[dict[str, Any]] = []

    # Ưu tiên 1: tính trực tiếp trên tập gold có error_types + pred_dir để có cả số lượng mẫu n_labeled
    labeled_groups = {
        (lg, ns): [r for r in recs if r.get("error_types")]
        for (lg, ns), recs in gold_by_group.items()
    }
    has_direct_labels = any(len(v) > 0 for v in labeled_groups.values())

    if has_direct_labels and preds_by_run:
        for (m_key, lg, ns), p_map in sorted(
            preds_by_run.items(), key=lambda item: (item[0][1], item[0][2], model_sort_rank(item[0][0]))
        ):
            if m_key == "baseline" or (lg, ns) not in labeled_groups:
                continue
            labeled_recs = labeled_groups[(lg, ns)]
            if not labeled_recs:
                continue
            for lbl in ERROR_LABELS:
                matching = []
                for rec in labeled_recs:
                    et = rec.get("error_types") or []
                    if isinstance(et, str):
                        et = [x.strip().upper() for x in et.split(",") if x.strip()]
                    else:
                        et = [str(x).strip().upper() for x in et]
                    if lbl in et:
                        matching.append(rec)
                if not matching:
                    continue
                fixed = sum(
                    1
                    for rec in matching
                    if normalize_eval_text(p_map.get(str(rec["id"]), ""))
                    == normalize_eval_text(str(rec["target"]))
                )
                rows.append(
                    {
                        "model_key": m_key,
                        "model_display": display_model_name(m_key),
                        "lang": lg,
                        "noise": ns,
                        "error_label": lbl,
                        "n_lines": len(matching),
                        "n_exact_fixed": fixed,
                        "exact_fix_rate_pct": round(fixed / len(matching) * 100.0, 2),
                    }
                )
        if rows:
            return pd.DataFrame(rows)

    # Ưu tiên 2: đọc từ các cột fix_DAU .. fix_MAT trong summary_df
    fix_cols = [f"fix_{lbl}" for lbl in ERROR_LABELS if f"fix_{lbl}" in df.columns]
    if fix_cols:
        for _, r in df[df["model_key"] != "baseline"].iterrows():
            for lbl in ERROR_LABELS:
                col = f"fix_{lbl}"
                if col in df.columns and pd.notna(r.get(col)):
                    rows.append(
                        {
                            "model_key": r["model_key"],
                            "model_display": display_model_name(r["model_key"]),
                            "lang": r["lang"],
                            "noise": r["noise"],
                            "error_label": lbl,
                            "n_lines": np.nan,
                            "n_exact_fixed": np.nan,
                            "exact_fix_rate_pct": round(float(r[col]), 2),
                        }
                    )
    return pd.DataFrame(rows)


def plot_rq3_diacritic_and_error_types(
    df: pd.DataFrame,
    err_df: pd.DataFrame,
    out_dir: Path,
) -> list[Path]:
    """RQ3: Phục hồi lỗi dấu tiếng Việt & tỷ lệ sửa đúng hoàn toàn theo 5 nhãn lỗi OCR."""
    vi_df = (
        df[(df["lang"] == "vi") & (df["diacritic_error_rate"].notna())].copy()
        if not df.empty and "diacritic_error_rate" in df.columns
        else pd.DataFrame()
    )
    has_vi_diacritic = not vi_df.empty
    has_err_types = not err_df.empty

    if not has_vi_diacritic and not has_err_types:
        return []

    ncols = 2 if (has_vi_diacritic and has_err_types) else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6.8 * ncols, 4.6), squeeze=False)
    ax_idx = 0

    if has_vi_diacritic:
        ax = axes[0][ax_idx]
        ax_idx += 1
        models = sorted(vi_df["model_key"].unique(), key=model_sort_rank)
        noises = ["light", "heavy"]
        n_models = max(1, len(models))
        width = min(0.16, 0.78 / n_models)
        x = np.arange(len(noises))

        for m_i, m_key in enumerate(models):
            vals = []
            for ns in noises:
                cell = vi_df[(vi_df["model_key"] == m_key) & (vi_df["noise"] == ns)]
                vals.append(
                    float(cell["diacritic_error_rate"].iloc[0])
                    if not cell.empty and pd.notna(cell["diacritic_error_rate"].iloc[0])
                    else float("nan")
                )
            offset = (m_i - (n_models - 1) / 2) * width
            bars = ax.bar(
                x + offset,
                vals,
                width=width * 0.92,
                label=display_model_name(m_key),
                color=model_color(m_key, m_i),
                edgecolor="white",
                linewidth=0.6,
            )
            _annotate_bars(ax, bars, fmt="{:.1f}", fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels(["Nhiễu nhẹ (Light)", "Nhiễu nặng (Heavy)"])
        ax.set_ylabel("Tỷ lệ lỗi dấu (%) ↓")
        ax.set_title("RQ3a · Tỷ lệ lỗi dấu âm tiết tiếng Việt (%) ↓")
        ymax = vi_df["diacritic_error_rate"].dropna().max()
        ax.set_ylim(0, max(5.0, float(ymax) * 1.25))
        ax.legend(frameon=True, facecolor="white", framealpha=0.9)

    if has_err_types:
        ax = axes[0][ax_idx]
        # Gộp trung bình theo model_key × error_label (ưu tiên tiếng Việt cho RQ3 nếu có, hoặc toàn tập gắn nhãn)
        agg = (
            err_df.groupby(["model_key", "error_label"], as_index=False)
            .agg(
                exact_fix_rate_pct=("exact_fix_rate_pct", "mean"),
                n_lines=("n_lines", "sum"),
            )
        )
        models = sorted(agg["model_key"].unique(), key=model_sort_rank)
        n_models = max(1, len(models))
        width = min(0.16, 0.80 / n_models)
        x = np.arange(len(ERROR_LABELS))

        for m_i, m_key in enumerate(models):
            vals = []
            for lbl in ERROR_LABELS:
                cell = agg[(agg["model_key"] == m_key) & (agg["error_label"] == lbl)]
                vals.append(
                    float(cell["exact_fix_rate_pct"].iloc[0])
                    if not cell.empty and pd.notna(cell["exact_fix_rate_pct"].iloc[0])
                    else float("nan")
                )
            offset = (m_i - (n_models - 1) / 2) * width
            bars = ax.bar(
                x + offset,
                vals,
                width=width * 0.92,
                label=display_model_name(m_key),
                color=model_color(m_key, m_i),
                edgecolor="white",
                linewidth=0.6,
            )
            _annotate_bars(ax, bars, fmt="{:.0f}", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(list(ERROR_LABELS))
        ax.set_ylabel("Tỷ lệ sửa đúng hoàn toàn (%) ↑")
        ax.set_ylim(0, 112)
        ax.set_title("RQ3b · Tỷ lệ sửa đúng hoàn toàn theo 5 nhãn lỗi OCR (%) ↑")
        ax.legend(frameon=True, facecolor="white", framealpha=0.9)

    fig.suptitle(
        "RQ3: Khả năng phục hồi lỗi dấu tiếng Việt và hiệu quả sửa theo 5 loại lỗi OCR",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    return save_figure(fig, out_dir, "rq3_vi_diacritic_and_error_types")


def plot_error_label_distribution(
    gold_by_group: dict[tuple[str, str], list[dict[str, Any]]],
    out_dir: Path,
) -> list[Path]:
    """Thống kê phân bố số lượng 5 nhãn lỗi OCR (DAU, KT, SO, TACH, MAT) trên tập 100 dòng gắn nhãn."""
    counts_by_lang: dict[str, Counter[str]] = {"vi": Counter(), "en": Counter()}
    seen_ids: set[tuple[str, str]] = set()

    for (lg, _ns), recs in gold_by_group.items():
        for rec in recs:
            et = rec.get("error_types") or []
            if not et:
                continue
            # Tránh đếm trùng cùng một dòng ở cả 2 file light/heavy nếu mang cùng base id; nếu id riêng thì đếm chuẩn
            rec_key = (lg, str(rec.get("id", "")))
            if rec_key in seen_ids:
                continue
            seen_ids.add(rec_key)
            if isinstance(et, str):
                et = [x.strip().upper() for x in et.split(",") if x.strip()]
            for lbl in et:
                lbl_up = str(lbl).strip().upper()
                if lbl_up in ERROR_LABELS:
                    counts_by_lang.setdefault(lg, Counter())[lbl_up] += 1

    total_all = sum(sum(c.values()) for c in counts_by_lang.values())
    if total_all == 0:
        return []

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(ERROR_LABELS))
    width = 0.36

    vi_vals = [counts_by_lang.get("vi", Counter())[lbl] for lbl in ERROR_LABELS]
    en_vals = [counts_by_lang.get("en", Counter())[lbl] for lbl in ERROR_LABELS]

    b1 = ax.bar(x - width / 2, vi_vals, width=width, label="Tiếng Việt (VI)", color="#0EA5E9")
    b2 = ax.bar(x + width / 2, en_vals, width=width, label="Tiếng Anh (EN)", color="#6366F1")
    _annotate_bars(ax, b1, fmt="{:.0f}", fontsize=8.5)
    _annotate_bars(ax, b2, fmt="{:.0f}", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels([ERROR_LABEL_DESCRIPTIONS[lbl] for lbl in ERROR_LABELS], rotation=12, ha="right")
    ax.set_ylabel("Số dòng gắn nhãn")
    ax.set_ylim(0, max(max(vi_vals + en_vals, default=5) * 1.25, 5))
    ax.set_title("Phân bố tần suất 5 nhãn lỗi OCR trên tập kiểm tra chất lượng")
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)
    fig.tight_layout()
    return save_figure(fig, out_dir, "error_types_distribution")


def plot_rq4_basic_vs_strict(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    """RQ4: Đối chứng trực tiếp prompt basic vs strict trên LLM (độ chính xác vs bảo toàn số liệu)."""
    if df.empty:
        return []

    llm_df = df[
        df["model_key"].str.endswith("-basic") | df["model_key"].str.endswith("-strict")
    ].copy()
    if llm_df.empty:
        return []

    llm_df["base_llm"] = llm_df["model_key"].str.replace(r"-(basic|strict)$", "", regex=True)
    llm_df["variant"] = np.where(llm_df["model_key"].str.endswith("-strict"), "strict", "basic")

    # Chỉ giữ các cặp (base_llm, lang, noise) có đủ cả basic và strict
    pair_counts = llm_df.groupby(["base_llm", "lang", "noise"])["variant"].nunique()
    valid_pairs = pair_counts[pair_counts >= 2].index.tolist()
    if not valid_pairs:
        return []

    panel_specs = [
        ("cer", "CER (%) ↓ (càng thấp càng tốt)", None),
        ("number_preservation", "Bảo toàn số liệu (%) ↑ (càng cao càng tốt)", (0, 112)),
        ("clean_corrupted_rate", "Làm hỏng dòng đúng (%) ↓", None),
        ("worse_rate", "Dòng bị làm tệ hơn (%) ↓", None),
    ]
    available_specs = [
        spec for spec in panel_specs if spec[0] in llm_df.columns and llm_df[spec[0]].notna().any()
    ]
    if not available_specs:
        return []

    valid_pairs_sorted = sorted(valid_pairs, key=lambda t: (t[0], t[1], t[2]))
    multi_llm = len({t[0] for t in valid_pairs_sorted}) > 1
    x_labels = [
        f"{b.upper()}\n{lg.upper()}-{ns}" if multi_llm else f"{lg.upper()}\n({ns})"
        for b, lg, ns in valid_pairs_sorted
    ]
    x = np.arange(len(valid_pairs_sorted))
    width = 0.35

    ncols = 2 if len(available_specs) > 1 else 1
    nrows = math.ceil(len(available_specs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.2 * nrows), squeeze=False)

    for idx, (col, title, fixed_ylim) in enumerate(available_specs):
        ax = axes[idx // ncols][idx % ncols]
        basic_vals = []
        strict_vals = []
        for base_llm, lg, ns in valid_pairs_sorted:
            b_cell = llm_df[
                (llm_df["base_llm"] == base_llm)
                & (llm_df["lang"] == lg)
                & (llm_df["noise"] == ns)
                & (llm_df["variant"] == "basic")
            ]
            s_cell = llm_df[
                (llm_df["base_llm"] == base_llm)
                & (llm_df["lang"] == lg)
                & (llm_df["noise"] == ns)
                & (llm_df["variant"] == "strict")
            ]
            basic_vals.append(float(b_cell[col].iloc[0]) if not b_cell.empty else float("nan"))
            strict_vals.append(float(s_cell[col].iloc[0]) if not s_cell.empty else float("nan"))

        b_bars = ax.bar(
            x - width / 2,
            basic_vals,
            width=width,
            label="Prompt cơ bản (basic)",
            color="#F59E0B",
            edgecolor="white",
        )
        s_bars = ax.bar(
            x + width / 2,
            strict_vals,
            width=width,
            label="Prompt ràng buộc số liệu (strict)",
            color="#7C3AED",
            edgecolor="white",
        )
        _annotate_bars(ax, b_bars, fmt="{:.1f}", fontsize=8)
        _annotate_bars(ax, s_bars, fmt="{:.1f}", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel("%")
        ax.set_title(f"RQ4 · {title}")
        if fixed_ylim is not None:
            ax.set_ylim(*fixed_ylim)
        else:
            ymax = max([v for v in basic_vals + strict_vals if not math.isnan(v)], default=10.0)
            ax.set_ylim(0, max(8.0, ymax * 1.28))
        if idx == 0:
            ax.legend(frameon=True, facecolor="white", framealpha=0.9)

    for empty_idx in range(len(available_specs), nrows * ncols):
        axes[empty_idx // ncols][empty_idx % ncols].set_visible(False)

    fig.suptitle(
        "RQ4: Đánh đổi giữa độ chính xác (CER) và bảo toàn số liệu khi dùng prompt basic vs strict",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    return save_figure(fig, out_dir, "rq4_basic_vs_strict_tradeoff")


# ==============================================================================
# 4. Trích xuất danh sách ví dụ lỗi điển hình (Typical Examples) cho Báo cáo & Slide
# ==============================================================================


def extract_typical_examples(
    gold_by_group: dict[tuple[str, str], list[dict[str, Any]]],
    preds_by_run: dict[tuple[str, str, str], dict[str, str]],
    max_per_category: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    """Lọc tự động các ví dụ đối chiếu đa mô hình theo 6 nhóm hiện tượng quan trọng."""
    categories: dict[str, list[dict[str, Any]]] = {
        "improved_exact": [],
        "number_corrupted": [],
        "clean_line_corrupted": [],
        "rq4_strict_saves_number": [],
        "en_coedit_vs_llm": [],
        "by_error_label": [],
    }
    if not gold_by_group or not preds_by_run:
        return categories

    seen_label_examples: Counter[str] = Counter()

    for (lg, ns), gold_rows in sorted(gold_by_group.items()):
        models_here = sorted(
            [m for (m, p_lg, p_ns) in preds_by_run if p_lg == lg and p_ns == ns and m != "baseline"],
            key=model_sort_rank,
        )
        if not models_here:
            continue

        for rec in gold_rows:
            rec_id = str(rec.get("id", ""))
            src = normalize_eval_text(str(rec.get("source", "")))
            tgt = normalize_eval_text(str(rec.get("target", "")))
            src_cer = compute_line_cer(src, tgt)

            model_preds: dict[str, str] = {}
            model_cers: dict[str, float] = {}
            for m_key in models_here:
                p_txt = normalize_eval_text(preds_by_run[(m_key, lg, ns)].get(rec_id, src))
                model_preds[m_key] = p_txt
                model_cers[m_key] = compute_line_cer(p_txt, tgt)

            entry_base = {
                "id": rec_id,
                "lang": lg,
                "noise": ns,
                "doc_id": rec.get("doc_id", ""),
                "page": rec.get("page", ""),
                "has_number": bool(rec.get("has_number")),
                "error_types": rec.get("error_types", []),
                "source": src,
                "target": tgt,
                "source_cer_pct": round(src_cer * 100.0, 2),
                "predictions": {
                    display_model_name(m): {
                        "text": model_preds[m],
                        "cer_pct": round(model_cers[m] * 100.0, 2),
                    }
                    for m in models_here
                },
            }

            # 1. Khôi phục hoàn hảo dòng có lỗi OCR (source != target -> prediction == target)
            if (
                len(categories["improved_exact"]) < max_per_category
                and src != tgt
                and any(p == tgt for p in model_preds.values())
            ):
                categories["improved_exact"].append(entry_base)

            # 2. Mô hình làm hỏng số liệu vốn đã đúng trong source (RQ2)
            tgt_nums = Counter(extract_numbers(tgt))
            src_nums = Counter(extract_numbers(src))
            valid_src_nums = src_nums & tgt_nums
            if len(categories["number_corrupted"]) < max_per_category and valid_src_nums:
                corrupting_models = [
                    display_model_name(m)
                    for m, p_txt in model_preds.items()
                    if any(Counter(extract_numbers(p_txt))[k] < cnt for k, cnt in valid_src_nums.items())
                ]
                if corrupting_models:
                    item = dict(entry_base)
                    item["note"] = f"Làm hỏng số liệu đúng ở OCR bởi: {', '.join(corrupting_models)}"
                    categories["number_corrupted"].append(item)

            # 3. Làm hỏng dòng vốn đã đúng hoàn toàn (source == target)
            if len(categories["clean_line_corrupted"]) < max_per_category and src == tgt:
                breakers = [display_model_name(m) for m, p_txt in model_preds.items() if p_txt != tgt]
                if breakers:
                    item = dict(entry_base)
                    item["note"] = f"Sửa sai dòng vốn đã đúng bởi: {', '.join(breakers)}"
                    categories["clean_line_corrupted"].append(item)

            # 4. Đối chứng RQ4: basic làm hỏng số hoặc tệ hơn, còn strict giữ đúng
            for llm_prefix in ("qwen3.5-2b", "qwen3.5-4b", "qwen2.5-1.5b"):
                b_key = f"{llm_prefix}-basic"
                s_key = f"{llm_prefix}-strict"
                if (
                    len(categories["rq4_strict_saves_number"]) < max_per_category
                    and b_key in model_preds
                    and s_key in model_preds
                ):
                    b_txt, s_txt = model_preds[b_key], model_preds[s_key]
                    b_num_ok = Counter(extract_numbers(b_txt)) == tgt_nums if tgt_nums else True
                    s_num_ok = Counter(extract_numbers(s_txt)) == tgt_nums if tgt_nums else True
                    if (not b_num_ok and s_num_ok) or (model_cers[s_key] + 0.02 < model_cers[b_key]):
                        item = dict(entry_base)
                        item["note"] = (
                            f"{display_model_name(s_key)} bảo toàn số/chính xác hơn "
                            f"{display_model_name(b_key)}"
                        )
                        categories["rq4_strict_saves_number"].append(item)
                        break

            # 5. Đặc trưng tiếng Anh: so sánh CoEdIT-Large vs LLM
            if (
                lg == "en"
                and len(categories["en_coedit_vs_llm"]) < max_per_category
                and "coedit-large" in model_preds
                and src != tgt
            ):
                llm_keys = [m for m in models_here if m != "coedit-large"]
                if llm_keys and any(model_preds["coedit-large"] != model_preds[k] for k in llm_keys):
                    categories["en_coedit_vs_llm"].append(entry_base)

            # 6. Ví dụ đại diện cho từng nhãn trong 5 loại lỗi OCR (DAU, KT, SO, TACH, MAT)
            et = rec.get("error_types") or []
            if isinstance(et, str):
                et = [x.strip().upper() for x in et.split(",") if x.strip()]
            for lbl in et:
                lbl_up = str(lbl).strip().upper()
                if lbl_up in ERROR_LABELS and seen_label_examples[lbl_up] < 2:
                    seen_label_examples[lbl_up] += 1
                    item = dict(entry_base)
                    item["highlight_label"] = lbl_up
                    categories["by_error_label"].append(item)
                    break

    return categories


def write_typical_examples_markdown(
    examples: dict[str, list[dict[str, Any]]],
    out_dir: Path,
) -> list[Path]:
    """Xuất danh sách ví dụ lỗi điển hình ra Markdown (dễ đọc/copy vào LaTeX) và JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "typical_examples.json"
    md_path = out_dir / "typical_examples.md"

    json_path.write_text(json.dumps(examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    section_titles = [
        ("improved_exact", "1. Khôi phục hoàn toàn dòng nhiễu OCR (`prediction == target`)"),
        ("number_corrupted", "2. Sửa quá tay làm hỏng số liệu tài chính (RQ2)"),
        ("clean_line_corrupted", "3. Làm hỏng dòng OCR vốn đã đúng (`source == target`)"),
        ("rq4_strict_saves_number", "4. Đối chứng RQ4: Prompt `strict` bảo toàn số liệu tốt hơn `basic`"),
        ("en_coedit_vs_llm", "5. Tiếng Anh: So sánh Seq2Seq (`CoEdIT-Large`) và LLM"),
        ("by_error_label", "6. Ví dụ minh hoạ theo 5 nhãn lỗi OCR (`DAU`, `KT`, `SO`, `TACH`, `MAT`)"),
    ]

    lines = [
        "# Danh sách ví dụ lỗi điển hình phục vụ Báo cáo (`results.tex`) & Slide",
        "",
        "> Sinh tự động bởi `src/analyze.py` từ tập predictions và ground truth.",
        "",
    ]

    total_written = 0
    for key, title in section_titles:
        items = examples.get(key, [])
        lines.append(f"## {title}")
        lines.append("")
        if not items:
            lines.append("_Chưa có mẫu phù hợp trong tập dữ liệu hiện tại._")
            lines.append("")
            continue

        for idx, item in enumerate(items, start=1):
            total_written += 1
            lbl_tag = (
                f" | Nhãn: `{','.join(item['error_types'])}`" if item.get("error_types") else ""
            )
            hl_tag = (
                f" | Nhãn tiêu điểm: **`{item['highlight_label']}`**"
                if item.get("highlight_label")
                else ""
            )
            lines.append(
                f"### Ví dụ {idx} — `{item['id']}` ({item['lang'].upper()}, nhiễu `{item['noise']}`{lbl_tag}{hl_tag})"
            )
            if item.get("note"):
                lines.append(f"- **Nhận xét:** {item['note']}")
            lines.append(f"- **Ground truth (`target`):** `{item['target']}`")
            lines.append(
                f"- **OCR gốc (`source`, CER={item['source_cer_pct']}%):** `{item['source']}`"
            )
            for m_name, p_info in item["predictions"].items():
                lines.append(
                    f"- **{m_name} (CER={p_info['cer_pct']}%):** `{p_info['text']}`"
                )
            lines.append("")

    if total_written == 0:
        lines.append(
            "> Lưu ý: Để trích xuất ví dụ từng dòng, hãy đảm bảo `data/vi_test.jsonl`, `data/en_test.jsonl` "
            "và các file trong `results/predictions/` đã có mặt."
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [md_path, json_path]


# ==============================================================================
# 5. CLI Entrypoint
# ==============================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sinh biểu đồ RQ1–RQ4, thống kê 5 nhãn lỗi OCR và trích xuất ví dụ điển hình (R3)."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/summary.csv"),
        help="File bảng tổng hợp kết quả từ R5 (mặc định: results/summary.csv).",
    )
    parser.add_argument(
        "--pred_dir",
        type=Path,
        default=Path("results/predictions"),
        help="Thư mục chứa các file predictions .jsonl (mặc định: results/predictions).",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data"),
        help="Thư mục chứa vi_test.jsonl và en_test.jsonl (mặc định: data).",
    )
    parser.add_argument(
        "--terms",
        type=Path,
        default=Path("data/terms.csv"),
        help="File danh sách thuật ngữ tài chính (mặc định: data/terms.csv).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/figures"),
        help="Thư mục xuất biểu đồ và ví dụ lỗi (mặc định: results/figures).",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=5,
        help="Số lượng ví dụ tối đa trích xuất cho mỗi nhóm hiện tượng (mặc định: 5).",
    )
    parser.add_argument(
        "--allow_fallback_metrics",
        action="store_true",
        help="Chỉ dùng khi chạy nháp trước khi R5 chốt results/summary.csv.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> list[Path]:
    if not HAS_DATA_STACK:
        raise RuntimeError(
            "Cần cài đặt numpy, pandas và matplotlib để chạy src/analyze.py "
            "(vd: pip install numpy pandas matplotlib)."
        )
    args = parse_args(argv)
    configure_matplotlib_style()
    args.out.mkdir(parents=True, exist_ok=True)

    summary_df = load_and_standardize_summary(args.summary)
    gold_by_group = load_gold_datasets(args.data_dir)
    preds_by_run = load_predictions_dir(args.pred_dir)
    terms_by_lang = load_terms_by_lang(args.terms)

    full_df = enrich_summary_with_predictions(
        summary_df=summary_df,
        gold_by_group=gold_by_group,
        preds_by_run=preds_by_run,
        terms_by_lang=terms_by_lang,
        allow_fallback_metrics=args.allow_fallback_metrics,
    )

    if full_df.empty:
        raise FileNotFoundError(
            f"Chưa tìm thấy bảng tổng hợp hợp lệ tại --summary ({args.summary}). "
            "Hãy đợi R5 chạy evaluate.py tạo results/summary.csv, hoặc truyền thêm --allow_fallback_metrics nếu muốn vẽ thử."
        )

    full_df["model_display"] = full_df["model_key"].map(display_model_name)
    full_df = full_df.sort_values(
        by=["lang", "noise", "model_key"],
        key=lambda col: col.map(lambda k: model_sort_rank(k)[0]) if col.name == "model_key" else col,
    ).reset_index(drop=True)

    generated: list[Path] = []

    norm_csv_path = args.out / "normalized_summary.csv"
    full_df.to_csv(norm_csv_path, index=False, encoding="utf-8-sig")
    generated.append(norm_csv_path)

    err_df = compute_error_type_table(full_df, gold_by_group, preds_by_run)
    if not err_df.empty:
        err_csv_path = args.out / "error_types_breakdown.csv"
        err_df.to_csv(err_csv_path, index=False, encoding="utf-8-sig")
        generated.append(err_csv_path)

    generated.extend(plot_rq1_cer_wer(full_df, args.out))
    generated.extend(plot_rq2_safety_and_preservation(full_df, args.out))
    generated.extend(plot_rq2_cer_scatter(gold_by_group, preds_by_run, args.out))
    generated.extend(plot_rq3_diacritic_and_error_types(full_df, err_df, args.out))
    generated.extend(plot_error_label_distribution(gold_by_group, args.out))
    generated.extend(plot_rq4_basic_vs_strict(full_df, args.out))

    examples = extract_typical_examples(
        gold_by_group=gold_by_group,
        preds_by_run=preds_by_run,
        max_per_category=args.max_examples,
    )
    generated.extend(write_typical_examples_markdown(examples, args.out))

    print(f"[R3][analyze] Đã sinh {len(generated)} file biểu đồ & báo cáo phân tích tại: {args.out}")
    for p in generated:
        print(f"  - {p}")
    return generated


if __name__ == "__main__":
    main()
