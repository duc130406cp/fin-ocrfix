"""Unit tests cho data/terms.csv (R5): kiểm tra schema danh sách thuật ngữ tài chính theo B3 & C5."""

from __future__ import annotations

import csv
import sys
import unicodedata
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TERMS_PATH = ROOT / "data" / "terms.csv"

EXPECTED_HEADER = ["term", "lang", "group"]
ALLOWED_LANGS = {"vi", "en"}
ALLOWED_GROUPS = {"chỉ tiêu", "viết tắt", "đơn vị"}
MIN_TERMS_PER_LANG = 45
MAX_TERMS_PER_LANG = 60


def read_raw_text() -> str:
    raw = TERMS_PATH.read_bytes()
    return raw.decode("utf-8")


def read_rows() -> tuple[list[str], list[list[str]]]:
    with TERMS_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    return header, rows


class TestTermsCsvSchema(unittest.TestCase):
    def test_file_exists_utf8_without_bom(self) -> None:
        self.assertTrue(TERMS_PATH.exists(), f"Thiếu file {TERMS_PATH}")
        raw = TERMS_PATH.read_bytes()
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"), "terms.csv không được có BOM")
        raw.decode("utf-8")  # ném UnicodeDecodeError nếu không phải UTF-8

    def test_header(self) -> None:
        header, _ = read_rows()
        self.assertEqual(header, EXPECTED_HEADER)

    def test_no_blank_lines(self) -> None:
        lines = read_raw_text().splitlines()
        for idx, line in enumerate(lines, start=1):
            self.assertTrue(line.strip(), f"Dòng {idx} rỗng")

    def test_each_row_has_three_non_empty_fields(self) -> None:
        _, rows = read_rows()
        for idx, row in enumerate(rows, start=2):
            self.assertEqual(len(row), 3, f"Dòng {idx} không có đúng 3 cột: {row}")
            for value in row:
                self.assertTrue(value.strip(), f"Dòng {idx} có ô rỗng: {row}")

    def test_lang_and_group_allowed(self) -> None:
        _, rows = read_rows()
        for idx, (term, lang, group) in enumerate(rows, start=2):
            self.assertIn(lang, ALLOWED_LANGS, f"Dòng {idx} ({term}): lang={lang!r}")
            self.assertIn(group, ALLOWED_GROUPS, f"Dòng {idx} ({term}): group={group!r}")

    def test_all_fields_nfc_and_no_extra_whitespace(self) -> None:
        _, rows = read_rows()
        for idx, row in enumerate(rows, start=2):
            for value in row:
                self.assertEqual(
                    value, unicodedata.normalize("NFC", value), f"Dòng {idx} chưa ở dạng NFC: {value!r}"
                )
                self.assertEqual(
                    value, " ".join(value.split()), f"Dòng {idx} có khoảng trắng thừa: {value!r}"
                )

    def test_no_duplicate_term_lang(self) -> None:
        _, rows = read_rows()
        counts = Counter((term, lang) for term, lang, _ in rows)
        duplicates = [key for key, n in counts.items() if n > 1]
        self.assertEqual(duplicates, [], f"Trùng (term, lang): {duplicates}")

    def test_term_count_per_lang(self) -> None:
        _, rows = read_rows()
        counts = Counter(lang for _, lang, _ in rows)
        for lang in sorted(ALLOWED_LANGS):
            self.assertGreaterEqual(counts[lang], MIN_TERMS_PER_LANG, f"lang={lang}")
            self.assertLessEqual(counts[lang], MAX_TERMS_PER_LANG, f"lang={lang}")

    def test_abbreviations_are_uppercase(self) -> None:
        _, rows = read_rows()
        for term, lang, group in rows:
            if group == "viết tắt":
                self.assertEqual(term, term.upper(), f"Viết tắt phải viết HOA: {term!r} ({lang})")
                self.assertNotIn(" ", term, f"Viết tắt không chứa khoảng trắng: {term!r}")

    def test_indicator_phrases_are_lowercase(self) -> None:
        _, rows = read_rows()
        for term, lang, group in rows:
            if group == "chỉ tiêu":
                self.assertEqual(term, term.lower(), f"Chỉ tiêu nên viết thường: {term!r} ({lang})")


class TestTermsCsvWithAnalyzeLoader(unittest.TestCase):
    def test_load_terms_by_lang_reads_all_terms(self) -> None:
        sys.path.insert(0, str(ROOT / "src"))
        try:
            import analyze  # noqa: PLC0415
        except ImportError as exc:
            self.skipTest(f"Không import được src/analyze.py: {exc}")
        if not getattr(analyze, "HAS_DATA_STACK", False):
            self.skipTest("Thiếu matplotlib/pandas, bỏ qua kiểm tra với analyze.load_terms_by_lang")

        _, rows = read_rows()
        expected = Counter(lang for _, lang, _ in rows)
        loaded = analyze.load_terms_by_lang(TERMS_PATH)
        self.assertEqual(len(loaded["vi"]), expected["vi"])
        self.assertEqual(len(loaded["en"]), expected["en"])
        self.assertIn("lợi nhuận sau thuế", loaded["vi"])
        self.assertIn("profit after tax", loaded["en"])


if __name__ == "__main__":
    unittest.main()
