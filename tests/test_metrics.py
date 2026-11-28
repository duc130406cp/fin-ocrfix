"""Unit tests cho src/metrics.py (R5): các metric C5, giá trị kỳ vọng tính tay."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import metrics  # noqa: E402
from metrics import Term  # noqa: E402

TERMS_PATH = ROOT / "data" / "terms.csv"

# Danh sách thuật ngữ tạm cho nhóm 8 (không phụ thuộc data/terms.csv).
T_LOI_NHUAN = Term("lợi nhuận", "vi", "chỉ tiêu")
T_LNST_FULL = Term("lợi nhuận sau thuế", "vi", "chỉ tiêu")
T_LNST = Term("LNST", "vi", "viết tắt")
T_VND = Term("VND", "vi", "đơn vị")
T_OWNERS_EQUITY = Term("owners' equity", "en", "chỉ tiêu")
VI_TERMS = [T_LOI_NHUAN, T_LNST_FULL, T_LNST, T_VND]
ALL_TERMS = VI_TERMS + [T_OWNERS_EQUITY]


# ------------------------------------------------------------------------------
# 1. Chuẩn hoá
# ------------------------------------------------------------------------------
class Test01Normalize(unittest.TestCase):
    def test_nfc_and_whitespace(self) -> None:
        self.assertEqual(metrics.normalize_text("Giá  bán \n"), "Giá bán")

    def test_line_cer_ignores_unicode_form(self) -> None:
        self.assertAlmostEqual(metrics.line_cer("Giá bán", "Giá bán"), 0.0, places=6)


# ------------------------------------------------------------------------------
# 2. CER / WER
# ------------------------------------------------------------------------------
class Test02CerWer(unittest.TestCase):
    def test_cer_single_substitution(self) -> None:
        self.assertAlmostEqual(metrics.corpus_cer(["abcf"], ["abcd"]), 0.25, places=6)

    def test_cer_is_corpus_level(self) -> None:
        # 1 lỗi / 6 ký tự gộp toàn tập (không phải trung bình (0.5 + 0) / 2).
        self.assertAlmostEqual(
            metrics.corpus_cer(["ax", "cdef"], ["ab", "cdef"]), 1 / 6, places=6
        )

    def test_wer_vietnamese_syllables(self) -> None:
        self.assertAlmostEqual(
            metrics.corpus_wer(["lợi nhuan sau thuế"], ["lợi nhuận sau thuế"]), 0.25, places=6
        )

    def test_perfect_prediction_zero_cer_wer(self) -> None:
        """Case bắt buộc của C5: prediction = target thì CER = WER = 0."""
        targets = ["Doanh thu thuần đạt 1.250.000 triệu đồng", "Net revenue rose 15%"]
        self.assertAlmostEqual(metrics.corpus_cer(targets, targets), 0.0, places=6)
        self.assertAlmostEqual(metrics.corpus_wer(targets, targets), 0.0, places=6)


# ------------------------------------------------------------------------------
# 3. Mức cải thiện
# ------------------------------------------------------------------------------
class Test03CerImprovement(unittest.TestCase):
    def test_positive_improvement(self) -> None:
        self.assertAlmostEqual(metrics.cer_improvement(0.2, 0.05), 0.75, places=6)

    def test_negative_improvement(self) -> None:
        self.assertAlmostEqual(metrics.cer_improvement(0.1, 0.2), -1.0, places=6)

    def test_zero_source_cer_is_nan(self) -> None:
        self.assertTrue(math.isnan(metrics.cer_improvement(0.0, 0.0)))


# ------------------------------------------------------------------------------
# 4. Tỷ lệ dòng bị làm tệ hơn
# ------------------------------------------------------------------------------
class Test04WorseRate(unittest.TestCase):
    def test_one_of_two_lines_worse(self) -> None:
        rate = metrics.worse_rate(
            preds=["abcd", "abce"], sources=["abcx", "abcd"], targets=["abcd", "abcd"]
        )
        self.assertAlmostEqual(rate, 0.5, places=6)

    def test_equal_cer_is_not_worse(self) -> None:
        self.assertAlmostEqual(metrics.worse_rate(["abcy"], ["abcx"], ["abcd"]), 0.0, places=6)


# ------------------------------------------------------------------------------
# 5. Tỷ lệ làm hỏng dòng đúng
# ------------------------------------------------------------------------------
class Test05CleanCorruptedRate(unittest.TestCase):
    SOURCES = ["a  b", "x y", "p"]
    TARGETS = ["a b", "x z", "r"]

    def test_clean_line_kept(self) -> None:
        # Chỉ dòng 1 sạch ("a  b" ~ "a b" sau chuẩn hoá) và được giữ nguyên.
        rate = metrics.clean_corrupted_rate(["a b", "x y", "q"], self.SOURCES, self.TARGETS)
        self.assertAlmostEqual(rate, 0.0, places=6)

    def test_clean_line_corrupted(self) -> None:
        rate = metrics.clean_corrupted_rate(["a c", "x y", "q"], self.SOURCES, self.TARGETS)
        self.assertAlmostEqual(rate, 1.0, places=6)

    def test_no_clean_line_is_nan(self) -> None:
        rate = metrics.clean_corrupted_rate(["x y", "q"], ["x y", "p"], ["x z", "r"])
        self.assertTrue(math.isnan(rate))


# ------------------------------------------------------------------------------
# 6. Trích số
# ------------------------------------------------------------------------------
class Test06ExtractNumbers(unittest.TestCase):
    def test_thousand_separators_percent_and_year(self) -> None:
        self.assertEqual(
            metrics.extract_numbers("Doanh thu 1.250.000 triệu, tăng 15% so với 2024."),
            ["1.250.000", "15%", "2024"],
        )

    def test_letter_o_breaks_number(self) -> None:
        self.assertEqual(metrics.extract_numbers("1.25O.000"), ["1.25", "000"])


# ------------------------------------------------------------------------------
# 7. Bảo toàn số liệu
# ------------------------------------------------------------------------------
class Test07NumberPreservation(unittest.TestCase):
    TARGET = "lãi 1.250 tỷ năm 2024"
    SOURCE = "lãi 1.25O tỷ năm 2024"

    def run_one(self, pred: str) -> dict:
        return metrics.number_preservation([pred], [self.SOURCE], [self.TARGET], [True])

    def test_numbers_preserved(self) -> None:
        res = self.run_one("lãi 1.250 tỷ năm 2024")
        self.assertAlmostEqual(res["rate"], 1.0, places=6)
        self.assertEqual(res["n_lines"], 1)
        self.assertEqual(res["broken_valid_numbers"], 0)

    def test_single_digit_change_is_caught(self) -> None:
        """Case bắt buộc của C5: đổi một chữ số thì metric bảo toàn số liệu phải bắt được."""
        res = self.run_one("lãi 1.205 tỷ năm 2024")
        self.assertAlmostEqual(res["rate"], 0.0, places=6)
        self.assertEqual(res["broken_valid_numbers"], 0)  # 1.250 không đúng sẵn trong source

    def test_breaking_a_valid_source_number(self) -> None:
        res = self.run_one("lãi 1.250 tỷ năm 2023")
        self.assertAlmostEqual(res["rate"], 0.0, places=6)
        self.assertEqual(res["broken_valid_numbers"], 1)  # 2024 đúng sẵn trong source bị đổi

    def test_has_number_false_is_skipped(self) -> None:
        res = metrics.number_preservation(
            ["lãi 1.250 tỷ năm 2024", "lãi 9 tỷ"],
            [self.SOURCE, "lãi 5 tỷ"],
            [self.TARGET, "lãi 5 tỷ"],
            [True, False],
        )
        self.assertEqual(res["n_lines"], 1)
        self.assertAlmostEqual(res["rate"], 1.0, places=6)

    def test_set_comparison_ignores_repeats(self) -> None:
        # Giới hạn đã biết của phép so tập (C5): "10 và 10" vs "10" vẫn tính là bảo toàn.
        res = metrics.number_preservation(["10"], ["10 và 10"], ["10 và 10"], [True])
        self.assertAlmostEqual(res["rate"], 1.0, places=6)


# ------------------------------------------------------------------------------
# 8. Thuật ngữ
# ------------------------------------------------------------------------------
class Test08Terms(unittest.TestCase):
    def test_longest_match_wins(self) -> None:
        self.assertEqual(
            dict(metrics.find_terms("Lợi nhuận sau thuế đạt 10 tỷ VND", ALL_TERMS)),
            {"lợi nhuận sau thuế": 1, "VND": 1},
        )

    def test_short_and_long_terms_non_overlapping(self) -> None:
        self.assertEqual(
            dict(metrics.find_terms("lợi nhuận và lợi nhuận sau thuế", ALL_TERMS)),
            {"lợi nhuận": 1, "lợi nhuận sau thuế": 1},
        )

    def test_uppercase_terms_are_case_sensitive(self) -> None:
        self.assertEqual(metrics.find_terms("giá trị vnd", ALL_TERMS)["VND"], 0)
        self.assertEqual(metrics.find_terms("giá trị Vnd", ALL_TERMS)["VND"], 0)

    def test_whole_word_match(self) -> None:
        self.assertEqual(metrics.find_terms("LNSTT năm nay", ALL_TERMS)["LNST"], 0)

    def test_curly_apostrophe_matches(self) -> None:
        self.assertEqual(
            metrics.find_terms("Owners’ equity increased", ALL_TERMS)["owners' equity"], 1
        )

    def test_term_preservation(self) -> None:
        res = metrics.term_preservation(
            preds=["Lợi nhuận sau thue tăng", "LNST và VND"],
            targets=["Lợi nhuận sau thuế tăng", "LNST và VND"],
            terms=VI_TERMS,
        )
        self.assertAlmostEqual(res["rate"], 2 / 3, places=6)
        self.assertEqual(res["n_occurrences"], 3)

    def test_term_preservation_no_occurrence_is_nan(self) -> None:
        res = metrics.term_preservation(["không có gì"], ["không có gì"], VI_TERMS)
        self.assertTrue(math.isnan(res["rate"]))
        self.assertEqual(res["n_occurrences"], 0)

    def test_load_real_terms_csv(self) -> None:
        loaded = metrics.load_terms(TERMS_PATH)
        self.assertEqual(set(loaded), {"vi", "en"})
        for lang in ("vi", "en"):
            self.assertTrue(loaded[lang], f"Không có thuật ngữ {lang}")
            self.assertTrue(all(isinstance(t, Term) and t.lang == lang for t in loaded[lang]))


# ------------------------------------------------------------------------------
# 9. Lỗi dấu
# ------------------------------------------------------------------------------
class Test09DiacriticErrorRate(unittest.TestCase):
    def test_one_wrong_of_five(self) -> None:
        res = metrics.diacritic_error_rate(["doanh thu thuan năm nay"], ["doanh thu thuần năm nay"])
        self.assertAlmostEqual(res["rate"], 0.2, places=6)
        self.assertEqual(res["n_pairs"], 5)

    def test_d_vs_dj(self) -> None:
        res = metrics.diacritic_error_rate(["dồng"], ["đồng"])
        self.assertAlmostEqual(res["rate"], 1.0, places=6)

    def test_case_difference_is_not_error(self) -> None:
        res = metrics.diacritic_error_rate(["doanh"], ["Doanh"])
        self.assertAlmostEqual(res["rate"], 0.0, places=6)

    def test_missing_syllable_not_counted(self) -> None:
        res = metrics.diacritic_error_rate(["lợi"], ["lợi nhuận"])
        self.assertAlmostEqual(res["rate"], 0.0, places=6)
        self.assertEqual(res["n_pairs"], 1)

    def test_edge_punctuation_stripped(self) -> None:
        res = metrics.diacritic_error_rate(["thuần"], ["thuần,"])
        self.assertAlmostEqual(res["rate"], 0.0, places=6)


# ------------------------------------------------------------------------------
# 10. Theo loại lỗi
# ------------------------------------------------------------------------------
class Test10ErrorTypeFixRates(unittest.TestCase):
    def test_error_labels_constant(self) -> None:
        self.assertEqual(metrics.ERROR_LABELS, ("DAU", "KT", "SO", "TACH", "MAT"))

    def test_fix_rates_per_label(self) -> None:
        res = metrics.error_type_fix_rates(
            preds=["a", "x", "c"],
            targets=["a", "b", "c"],
            error_types=[["DAU"], ["DAU", "SO"], []],
        )
        self.assertAlmostEqual(res["DAU"]["rate"], 0.5, places=6)
        self.assertEqual(res["DAU"]["n"], 2)
        self.assertAlmostEqual(res["SO"]["rate"], 0.0, places=6)
        self.assertEqual(res["SO"]["n"], 1)
        self.assertTrue(math.isnan(res["KT"]["rate"]))
        self.assertEqual(res["KT"]["n"], 0)

    def test_string_labels_parsed(self) -> None:
        from_str = metrics.error_type_fix_rates(["a", "x"], ["a", "b"], ["dau, so", []])
        from_list = metrics.error_type_fix_rates(["a", "x"], ["a", "b"], [["DAU", "SO"], []])
        for label in metrics.ERROR_LABELS:
            self.assertEqual(from_str[label]["n"], from_list[label]["n"], label)
        self.assertEqual(from_str["DAU"]["n"], 1)
        self.assertEqual(from_str["SO"]["n"], 1)
        self.assertAlmostEqual(from_str["DAU"]["rate"], 1.0, places=6)


# ------------------------------------------------------------------------------
# 11. Ghép prediction với gold
# ------------------------------------------------------------------------------
class Test11AlignPredictions(unittest.TestCase):
    GOLD = [
        {"id": "vi_0001", "source": "s1", "target": "t1"},
        {"id": "vi_0002", "source": "s2", "target": "t2"},
        {"id": "vi_0003", "source": "s3", "target": "t3"},
    ]

    def test_missing_id_raises(self) -> None:
        preds = [{"id": "vi_0001", "prediction": "p1"}, {"id": "vi_0002", "prediction": "p2"}]
        with self.assertRaisesRegex(ValueError, "vi_0003"):
            metrics.align_predictions(self.GOLD, preds)

    def test_duplicate_id_raises(self) -> None:
        preds = [
            {"id": "vi_0001", "prediction": "p1"},
            {"id": "vi_0002", "prediction": "p2"},
            {"id": "vi_0002", "prediction": "p2b"},
            {"id": "vi_0003", "prediction": "p3"},
        ]
        with self.assertRaisesRegex(ValueError, "vi_0002"):
            metrics.align_predictions(self.GOLD, preds)

    def test_unknown_id_raises(self) -> None:
        preds = [
            {"id": "vi_0001", "prediction": "p1"},
            {"id": "vi_0002", "prediction": "p2"},
            {"id": "vi_0003", "prediction": "p3"},
            {"id": "vi_9999", "prediction": "p?"},
        ]
        with self.assertRaisesRegex(ValueError, "vi_9999"):
            metrics.align_predictions(self.GOLD, preds)

    def test_reorders_to_gold_order(self) -> None:
        preds = [
            {"id": "vi_0003", "prediction": "p3"},
            {"id": "vi_0001", "prediction": "p1"},
            {"id": "vi_0002", "prediction": "p2"},
        ]
        self.assertEqual(metrics.align_predictions(self.GOLD, preds), ["p1", "p2", "p3"])


if __name__ == "__main__":
    unittest.main()
