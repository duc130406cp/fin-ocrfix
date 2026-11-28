"""R5 · Bộ metric chấm hậu xử lý OCR theo Plan4project Phần C5.

Mọi hàm trả về TỶ LỆ trong [0, 1] (riêng cer_improvement có thể âm); việc đổi sang % là của
src/evaluate.py. Trước khi chấm, mọi chuỗi được chuẩn hoá bằng normalize_text (Unicode NFC,
gộp khoảng trắng thừa). Các danh sách song song (preds, sources, targets, ...) phải cùng độ dài,
ngược lại hàm raise ValueError.

Chỉ dùng thư viện chuẩn + jiwer (jiwer được import bên trong hàm tính CER/WER).

Ánh xạ sang các dòng metric của C5:
    CER, WER                    -> corpus_cer, corpus_wer, line_cer
    Mức cải thiện               -> cer_improvement
    Tỷ lệ dòng bị làm tệ hơn    -> worse_rate
    Tỷ lệ làm hỏng dòng đúng    -> clean_corrupted_rate
    Bảo toàn số liệu            -> extract_numbers, number_preservation
    Bảo toàn thuật ngữ          -> Term, load_terms, find_terms, term_preservation
    Tỷ lệ lỗi dấu (chỉ VI)      -> strip_vi_diacritics, vi_syllables, diacritic_error_rate
    Theo loại lỗi               -> ERROR_LABELS, error_type_fix_rates
    Ghép prediction với gold    -> align_predictions
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ==============================================================================
# Hằng số
# ==============================================================================

# Chuỗi số theo C5: chữ số, có thể có dấu chấm/phẩy phân cách và dấu % ở cuối.
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*%?")

# 5 nhãn loại lỗi OCR (B4).
ERROR_LABELS: tuple[str, ...] = ("DAU", "KT", "SO", "TACH", "MAT")

# Các nhóm thuật ngữ hợp lệ trong data/terms.csv (B3).
TERM_GROUPS: tuple[str, ...] = ("chỉ tiêu", "viết tắt", "đơn vị")
ABBREVIATION_GROUP = "viết tắt"

# Các dấu nháy đơn được coi là cùng một ký tự khi khớp thuật ngữ; đều quy về APOSTROPHE.
APOSTROPHE = "'"
APOSTROPHE_VARIANTS: tuple[str, ...] = ("’", "‘", "'")  # ’ ‘ '

# Dấu câu bị bỏ ở hai đầu token khi tách âm tiết tiếng Việt.
VI_TOKEN_STRIP_CHARS = ".,;:!?()[]{}\"'“”‘’/-–—…"


# ==============================================================================
# Chuẩn hoá
# ==============================================================================


def normalize_text(text: str) -> str:
    """Chuẩn hoá trước khi chấm (C5).

    Unicode NFC, gộp mọi chuỗi khoảng trắng (space, tab, xuống dòng, ...) thành một dấu cách,
    bỏ khoảng trắng ở hai đầu. Không hạ chữ thường, không bỏ dấu câu.

    Ví dụ: "Gia\\u0301  bán \\n" -> "Giá bán".
    """
    raise NotImplementedError


# ==============================================================================
# CER / WER
# ==============================================================================


def corpus_cer(preds: Sequence[str], targets: Sequence[str]) -> float:
    """CER mức corpus bằng jiwer trên văn bản đã normalize_text.

    Tổng số phép sửa ký tự trên toàn tập chia tổng số ký tự của targets (không phải trung bình
    CER từng dòng). Ví dụ preds=["ax", "cdef"], targets=["ab", "cdef"] -> 1/6.
    ValueError nếu độ dài lệch nhau hoặc có target rỗng sau chuẩn hoá.
    """
    raise NotImplementedError


def corpus_wer(preds: Sequence[str], targets: Sequence[str]) -> float:
    """WER mức corpus bằng jiwer trên văn bản đã normalize_text.

    Từ được tách theo khoảng trắng; với tiếng Việt mỗi "từ" là một âm tiết.
    ValueError nếu độ dài lệch nhau hoặc có target rỗng sau chuẩn hoá.
    """
    raise NotImplementedError


def line_cer(pred: str, target: str) -> float:
    """CER của một dòng (jiwer) trên văn bản đã normalize_text.

    ValueError nếu target rỗng sau chuẩn hoá.
    """
    raise NotImplementedError


def cer_improvement(cer_source: float, cer_pred: float) -> float:
    """Mức cải thiện tương đối (C5): (cer_source - cer_pred) / cer_source.

    Âm nếu mô hình làm tệ hơn bản OCR gốc. Trả về NaN nếu cer_source == 0.
    """
    raise NotImplementedError


# ==============================================================================
# Sửa quá tay & làm hỏng dòng đúng
# ==============================================================================


def worse_rate(
    preds: Sequence[str], sources: Sequence[str], targets: Sequence[str]
) -> float:
    """Tỷ lệ dòng bị làm tệ hơn (C5), chỉ số "sửa quá tay" chính.

    Số dòng có line_cer(pred, target) > line_cer(source, target) (lớn hơn NGHIÊM NGẶT; CER bằng
    nhau không tính) chia tổng số dòng. NaN nếu không có dòng nào.
    """
    raise NotImplementedError


def clean_corrupted_rate(
    preds: Sequence[str], sources: Sequence[str], targets: Sequence[str]
) -> float:
    """Tỷ lệ làm hỏng dòng đúng (C5).

    Dòng sạch: normalize_text(source) == normalize_text(target). Trong các dòng sạch, tỷ lệ dòng
    có normalize_text(pred) != normalize_text(target). NaN nếu không có dòng sạch.
    """
    raise NotImplementedError


# ==============================================================================
# Bảo toàn số liệu
# ==============================================================================


def extract_numbers(text: str) -> list[str]:
    """Mọi chuỗi số khớp NUMBER_RE trong normalize_text(text), theo thứ tự xuất hiện.

    Ví dụ: "Doanh thu 1.250.000 triệu, tăng 15% so với 2024." -> ["1.250.000", "15%", "2024"];
    "1.25O.000" -> ["1.25", "000"].
    """
    raise NotImplementedError


def number_preservation(
    preds: Sequence[str],
    sources: Sequence[str],
    targets: Sequence[str],
    has_number: Sequence[bool],
) -> dict[str, Any]:
    """Bảo toàn số liệu (C5).

    Chỉ xét dòng có has_number=True VÀ target có ít nhất một số. Trên các dòng đó:
      - dòng được bảo toàn khi set(extract_numbers(pred)) == set(extract_numbers(target));
      - broken_valid_numbers: số dòng mà (set(số source) ∩ set(số target)) KHÔNG nằm trọn trong
        set(số pred), tức mô hình đã đổi một số vốn đã đúng trong source.

    Giới hạn đã biết: so theo tập nên bỏ qua số lần lặp và thứ tự số.

    Trả về {"rate": float, "n_lines": int, "broken_valid_numbers": int};
    rate = số dòng bảo toàn / n_lines, NaN nếu n_lines = 0.
    """
    raise NotImplementedError


# ==============================================================================
# Bảo toàn thuật ngữ
# ==============================================================================


@dataclass(frozen=True)
class Term:
    """Một dòng của data/terms.csv."""

    term: str
    lang: str
    group: str


def load_terms(path: str | Path) -> dict[str, list[Term]]:
    """Đọc data/terms.csv (UTF-8, cột term,lang,group) thành {lang: [Term, ...]}.

    term được normalize_text; giữ thứ tự dòng trong file.
    """
    raise NotImplementedError


def find_terms(text: str, terms: Sequence[Term]) -> Counter[str]:
    """Đếm số lần xuất hiện của từng thuật ngữ trong text. Khoá của Counter là Term.term.

    Quy tắc khớp:
      - trên normalize_text(text); các dấu nháy ’ ‘ ' coi là cùng một ký tự (cả text lẫn term);
      - thuật ngữ nhóm "viết tắt" hoặc viết toàn chữ HOA (term == term.upper()): phân biệt hoa
        thường; các thuật ngữ khác: không phân biệt (so sánh sau casefold);
      - khớp trọn từ: ký tự liền trước và liền sau đoạn khớp không được là \\w (Unicode);
      - ưu tiên cụm dài nhất, không chồng lấn: xét thuật ngữ theo độ dài giảm dần, đoạn văn bản
        đã được một thuật ngữ khớp thì không được khớp lại bởi thuật ngữ ngắn hơn.

    Ví dụ: "Lợi nhuận sau thuế đạt 10 tỷ VND" -> {"lợi nhuận sau thuế": 1, "VND": 1}
    (không đếm "lợi nhuận" vì nằm trong cụm dài hơn). Thuật ngữ không xuất hiện không có khoá.
    """
    raise NotImplementedError


def term_preservation(
    preds: Sequence[str], targets: Sequence[str], terms: Sequence[Term]
) -> dict[str, Any]:
    """Bảo toàn thuật ngữ (C5): tỷ lệ lần xuất hiện trong target cũng có mặt trong prediction.

    Với mỗi dòng và mỗi thuật ngữ (đếm bằng find_terms):
      preserved     += min(đếm trong target, đếm trong pred)
      n_occurrences += đếm trong target
    gộp chung toàn tập. terms là danh sách thuật ngữ của đúng ngôn ngữ đang chấm.

    Trả về {"rate": float, "n_occurrences": int}; rate = preserved / n_occurrences,
    NaN nếu n_occurrences = 0.
    """
    raise NotImplementedError


# ==============================================================================
# Lỗi dấu tiếng Việt
# ==============================================================================


def strip_vi_diacritics(text: str) -> str:
    """Bỏ dấu thanh và dấu phụ: đ->d, Đ->D, rồi NFD và bỏ ký tự loại Mn, cuối cùng NFC.

    Giữ nguyên hoa thường. Ví dụ "Đồng thuần" -> "Dong thuan".
    """
    raise NotImplementedError


def vi_syllables(text: str) -> list[str]:
    """Tách âm tiết: normalize_text, tách theo khoảng trắng, bỏ dấu câu ở hai đầu mỗi token
    (VI_TOKEN_STRIP_CHARS), chỉ giữ token còn lại có ít nhất một chữ cái.

    Ví dụ "Doanh thu (thuần), 2024" -> ["Doanh", "thu", "thuần"].
    """
    raise NotImplementedError


def diacritic_error_rate(preds: Sequence[str], targets: Sequence[str]) -> dict[str, Any]:
    """Tỷ lệ lỗi dấu (C5, chỉ dùng cho tiếng Việt).

    Mỗi dòng: tách âm tiết target và pred bằng vi_syllables; chữ gốc = strip_vi_diacritics(...)
    .lower(); căn hai dãy chữ gốc bằng difflib.SequenceMatcher (autojunk=False). Chỉ các cặp âm
    tiết thuộc khối "equal" được đếm vào n_pairs; cặp lỗi khi hai âm tiết khác nhau sau .lower()
    (khác dấu; khác hoa thường không tính). Gộp chung toàn tập.

    Trả về {"rate": float, "n_pairs": int}; rate = số cặp lỗi / n_pairs, NaN nếu n_pairs = 0.
    """
    raise NotImplementedError


# ==============================================================================
# Theo loại lỗi
# ==============================================================================


def error_type_fix_rates(
    preds: Sequence[str],
    targets: Sequence[str],
    error_types: Sequence[Sequence[str] | str | None],
) -> dict[str, dict[str, Any]]:
    """Tỷ lệ dòng được sửa đúng hoàn toàn theo từng nhãn lỗi (C5, trên 100 dòng có error_types).

    error_types của mỗi dòng là list nhãn, chuỗi "DAU, SO" (tách theo dấu phẩy), hoặc None/rỗng;
    mỗi nhãn được strip + upper, nhãn ngoài ERROR_LABELS bị bỏ qua. Dòng được sửa khi
    normalize_text(pred) == normalize_text(target).

    Trả về {label: {"rate": float, "n": int}} cho đủ mọi label trong ERROR_LABELS;
    n = số dòng mang nhãn, rate = số dòng mang nhãn được sửa / n, NaN nếu n = 0.
    """
    raise NotImplementedError


# ==============================================================================
# Ghép prediction với gold
# ==============================================================================


def align_predictions(
    gold_rows: Sequence[Mapping[str, Any]], pred_rows: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Trả về list prediction theo đúng thứ tự gold_rows, ghép theo khoá "id".

    pred_rows là các dòng của results/predictions/*.jsonl (có "id" và "prediction").
    Raise ValueError, thông báo nêu rõ các id liên quan, nếu:
      - có id của gold không có trong pred_rows (thiếu dòng);
      - một id xuất hiện nhiều lần trong pred_rows (trùng);
      - pred_rows có id không thuộc gold_rows (id lạ).
    KHÔNG thay dòng thiếu bằng source.
    """
    raise NotImplementedError
