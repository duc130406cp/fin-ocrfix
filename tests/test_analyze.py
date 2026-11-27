"""Unit tests cho src/analyze.py (R3): kiểm tra công thức C5, chuẩn hoá bảng, ví dụ lỗi & vẽ biểu đồ."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import analyze  # noqa: E402


class TestAnalyzeCoreMetricsAndExamples(unittest.TestCase):
    def test_perfect_prediction_cer_zero_and_preservation_100(self) -> None:
        """Kiểm thử bắt buộc ở C5: prediction = target -> CER = 0 và bảo toàn = 100%."""
        gold = [
            {
                "id": "vi_01",
                "lang": "vi",
                "noise": "light",
                "has_number": True,
                "source": "Doanh thu thuan dat 1.25O.000 trieu dong, tang l5%",
                "target": "Doanh thu thuần đạt 1.250.000 triệu đồng, tăng 15%",
                "error_types": ["DAU", "SO"],
            },
            {
                "id": "vi_02",
                "lang": "vi",
                "noise": "light",
                "has_number": False,
                "source": "Lợi nhuận sau thuế tăng trưởng ổn định",
                "target": "Lợi nhuận sau thuế tăng trưởng ổn định",
                "error_types": [],
            },
        ]
        pred_map = {r["id"]: r["target"] for r in gold}
        metrics = analyze.compute_metrics_for_run(
            gold_rows=gold,
            pred_map=pred_map,
            lang="vi",
            lang_terms=["Doanh thu thuần", "Lợi nhuận sau thuế"],
        )
        self.assertAlmostEqual(metrics["cer"], 0.0)
        self.assertAlmostEqual(metrics["wer"], 0.0)
        self.assertAlmostEqual(metrics["number_preservation"], 100.0)
        self.assertAlmostEqual(metrics["term_preservation"], 100.0)
        self.assertAlmostEqual(metrics["clean_corrupted_rate"], 0.0)
        self.assertAlmostEqual(metrics["diacritic_error_rate"], 0.0)
        self.assertAlmostEqual(metrics["fix_DAU"], 100.0)
        self.assertAlmostEqual(metrics["fix_SO"], 100.0)

    def test_single_digit_change_caught_by_number_preservation(self) -> None:
        """Kiểm thử bắt buộc ở C5: đổi một chữ số thì metric bảo toàn số liệu phải bắt được."""
        gold = [
            {
                "id": "vi_01",
                "lang": "vi",
                "noise": "light",
                "has_number": True,
                "source": "Doanh thu thuần đạt 1.250 tỷ đồng",
                "target": "Doanh thu thuần đạt 1.250 tỷ đồng",
                "error_types": [],
            }
        ]
        # Mô hình sửa quá tay 1.250 -> 1.205
        pred_map = {"vi_01": "Doanh thu thuần đạt 1.205 tỷ đồng"}
        metrics = analyze.compute_metrics_for_run(
            gold_rows=gold,
            pred_map=pred_map,
            lang="vi",
            lang_terms=["Doanh thu thuần"],
        )
        self.assertAlmostEqual(metrics["number_preservation"], 0.0)
        self.assertEqual(metrics["broken_valid_numbers"], 1)
        self.assertAlmostEqual(metrics["worse_rate"], 100.0)
        self.assertAlmostEqual(metrics["clean_corrupted_rate"], 100.0)

    def test_vi_diacritic_error_rate_sequence_matcher(self) -> None:
        target = ["Doanh thu thuần tăng mạnh"]
        # 5 âm tiết cùng chữ gốc; sai dấu 2 âm tiết ("thuan", "manh") -> 2/5 = 0.4
        pred = ["Doanh thu thuan tăng manh"]
        rate = analyze.compute_vi_diacritic_error_rate(pred, target)
        self.assertAlmostEqual(rate, 2 / 5)

    def test_extract_typical_examples_categories(self) -> None:
        gold_by_group = {
            ("en", "light"): [
                {
                    "id": "en_1",
                    "lang": "en",
                    "noise": "light",
                    "has_number": True,
                    "source": "Cost of goods sold decreased by 8% in 2024",
                    "target": "Cost of goods sold decreased by 8% in 2024",
                    "error_types": [],
                },
                {
                    "id": "en_2",
                    "lang": "en",
                    "noise": "light",
                    "has_number": True,
                    "source": "Net revenue reached VND 1,25O billion",
                    "target": "Net revenue reached VND 1,250 billion",
                    "error_types": ["SO"],
                },
            ]
        }
        preds_by_run = {
            ("coedit-large", "en", "light"): {
                "en_1": "Cost of goods sold decreased by 8% in 2024",
                "en_2": "Net revenue reached VND 1,25O billion",
            },
            ("qwen3.5-2b-basic", "en", "light"): {
                "en_1": "Cost of goods sold decreased by 8% in 2023",  # làm hỏng năm 2024 -> 2023
                "en_2": "Net revenue reached VND 1,250 billion",
            },
            ("qwen3.5-2b-strict", "en", "light"): {
                "en_1": "Cost of goods sold decreased by 8% in 2024",  # giữ nguyên đúng 2024
                "en_2": "Net revenue reached VND 1,250 billion",
            },
        }
        examples = analyze.extract_typical_examples(gold_by_group, preds_by_run)
        self.assertGreaterEqual(len(examples["improved_exact"]), 1)
        self.assertGreaterEqual(len(examples["number_corrupted"]), 1)
        self.assertGreaterEqual(len(examples["clean_line_corrupted"]), 1)
        self.assertGreaterEqual(len(examples["rq4_strict_saves_number"]), 1)
        self.assertGreaterEqual(len(examples["en_coedit_vs_llm"]), 1)
        self.assertGreaterEqual(len(examples["by_error_label"]), 1)


@unittest.skipUnless(analyze.HAS_DATA_STACK, "Cần numpy, pandas, matplotlib để test xuất biểu đồ")
class TestAnalyzeChartsEndToEnd(unittest.TestCase):
    def test_percentage_vs_ratio_scale_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)

            # Bảng đã ở thang phần trăm (0..100), có clean_corrupted_rate = 0.5 (%) -> KHÔNG được nhân thành 50%
            pct_csv = tmp_dir / "summary_pct.csv"
            pct_csv.write_text(
                "model,lang,noise,cer,wer,number_preservation,clean_corrupted_rate\n"
                "coedit-large,en,light,4.2,11.5,94.0,0.5\n",
                encoding="utf-8",
            )
            df_pct = analyze.load_and_standardize_summary(pct_csv)
            self.assertAlmostEqual(float(df_pct["clean_corrupted_rate"].iloc[0]), 0.5)
            self.assertAlmostEqual(float(df_pct["number_preservation"].iloc[0]), 94.0)

            # Bảng ở thang tỷ lệ thập phân (0..1) -> phải tự nhân 100 đồng bộ
            ratio_csv = tmp_dir / "summary_ratio.csv"
            ratio_csv.write_text(
                "model,lang,noise,cer,wer,number_preservation,clean_corrupted_rate\n"
                "coedit-large,en,light,0.042,0.115,0.94,0.005\n",
                encoding="utf-8",
            )
            df_ratio = analyze.load_and_standardize_summary(ratio_csv)
            self.assertAlmostEqual(float(df_ratio["clean_corrupted_rate"].iloc[0]), 0.5)
            self.assertAlmostEqual(float(df_ratio["number_preservation"].iloc[0]), 94.0)

    def test_full_analyze_pipeline_generates_all_rq_figures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            data_dir = tmp_dir / "data"
            pred_dir = tmp_dir / "predictions"
            out_dir = tmp_dir / "figures"
            summary_csv = tmp_dir / "summary.csv"
            terms_csv = data_dir / "terms.csv"
            data_dir.mkdir(parents=True)
            pred_dir.mkdir(parents=True)

            terms_csv.write_text(
                "term,lang,category\ndoanh thu thuần,vi,metric\nnet revenue,en,metric\n",
                encoding="utf-8",
            )

            vi_gold = [
                {
                    "id": "vi_1_l",
                    "lang": "vi",
                    "split": "test",
                    "noise": "light",
                    "has_number": True,
                    "source": "Doanh thu thuan nam 2025 dat 1.25O ty",
                    "target": "Doanh thu thuần năm 2025 đạt 1.250 tỷ",
                    "error_types": ["DAU", "SO"],
                },
                {
                    "id": "vi_1_h",
                    "lang": "vi",
                    "split": "test",
                    "noise": "heavy",
                    "has_number": True,
                    "source": "Doanhthu thuan nam 2O25 dat 1.25O ty",
                    "target": "Doanh thu thuần năm 2025 đạt 1.250 tỷ",
                    "error_types": ["DAU", "SO", "TACH"],
                },
            ]
            en_gold = [
                {
                    "id": "en_1_l",
                    "lang": "en",
                    "split": "test",
                    "noise": "light",
                    "has_number": True,
                    "source": "Net revenue in 2025 reached 5OO million",
                    "target": "Net revenue in 2025 reached 500 million",
                    "error_types": ["SO"],
                },
                {
                    "id": "en_1_h",
                    "lang": "en",
                    "split": "test",
                    "noise": "heavy",
                    "has_number": False,
                    "source": "The Cornpany Audit Cornmittee met",
                    "target": "The Company Audit Committee met",
                    "error_types": ["KT", "MAT"],
                },
            ]
            (data_dir / "vi_test.jsonl").write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in vi_gold) + "\n",
                encoding="utf-8",
            )
            (data_dir / "en_test.jsonl").write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in en_gold) + "\n",
                encoding="utf-8",
            )

            pred_files = {
                "bartpho-v2_vi_light.jsonl": [
                    {"id": "vi_1_l", "source": vi_gold[0]["source"], "prediction": vi_gold[0]["target"]}
                ],
                "bartpho-v2_vi_heavy.jsonl": [
                    {"id": "vi_1_h", "source": vi_gold[1]["source"], "prediction": "Doanh thu thuần năm 2025 đạt 1.25O tỷ"}
                ],
                "qwen3.5-2b-basic_vi_light.jsonl": [
                    {"id": "vi_1_l", "source": vi_gold[0]["source"], "prediction": "Doanh thu thuần năm 2024 đạt 1.250 tỷ"}
                ],
                "qwen3.5-2b-strict_vi_light.jsonl": [
                    {"id": "vi_1_l", "source": vi_gold[0]["source"], "prediction": vi_gold[0]["target"]}
                ],
                "qwen3.5-2b-basic_vi_heavy.jsonl": [
                    {"id": "vi_1_h", "source": vi_gold[1]["source"], "prediction": vi_gold[1]["target"]}
                ],
                "qwen3.5-2b-strict_vi_heavy.jsonl": [
                    {"id": "vi_1_h", "source": vi_gold[1]["source"], "prediction": vi_gold[1]["target"]}
                ],
                "coedit-large_en_light.jsonl": [
                    {"id": "en_1_l", "source": en_gold[0]["source"], "prediction": en_gold[0]["source"]}
                ],
                "coedit-large_en_heavy.jsonl": [
                    {"id": "en_1_h", "source": en_gold[1]["source"], "prediction": en_gold[1]["target"]}
                ],
                "qwen3.5-2b-basic_en_light.jsonl": [
                    {"id": "en_1_l", "source": en_gold[0]["source"], "prediction": en_gold[0]["target"]}
                ],
                "qwen3.5-2b-strict_en_light.jsonl": [
                    {"id": "en_1_l", "source": en_gold[0]["source"], "prediction": en_gold[0]["target"]}
                ],
                "qwen3.5-2b-basic_en_heavy.jsonl": [
                    {"id": "en_1_h", "source": en_gold[1]["source"], "prediction": en_gold[1]["target"]}
                ],
                "qwen3.5-2b-strict_en_heavy.jsonl": [
                    {"id": "en_1_h", "source": en_gold[1]["source"], "prediction": en_gold[1]["target"]}
                ],
            }
            for fname, rows in pred_files.items():
                (pred_dir / fname).write_text(
                    "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                    encoding="utf-8",
                )

            with summary_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "model",
                        "lang",
                        "noise",
                        "cer",
                        "wer",
                        "worse_rate",
                        "clean_corrupted_rate",
                        "number_preservation",
                        "term_preservation",
                        "diacritic_error_rate",
                    ]
                )
                writer.writerows(
                    [
                        ["baseline", "vi", "light", 8.5, 28.0, 0.0, 0.0, 40.0, 65.0, 22.0],
                        ["bartpho-v2", "vi", "light", 2.4, 8.0, 4.0, 3.0, 82.0, 92.0, 3.5],
                        ["qwen3.5-2b-basic", "vi", "light", 2.9, 9.5, 9.0, 8.0, 78.0, 90.0, 4.1],
                        ["qwen3.5-2b-strict", "vi", "light", 2.6, 8.8, 3.0, 2.0, 95.0, 94.0, 4.3],
                        ["baseline", "vi", "heavy", 16.2, 46.0, 0.0, 0.0, 22.0, 45.0, 38.0],
                        ["bartpho-v2", "vi", "heavy", 6.8, 20.0, 7.0, 5.0, 68.0, 84.0, 9.2],
                        ["qwen3.5-2b-basic", "vi", "heavy", 6.1, 18.0, 11.0, 9.0, 72.0, 86.0, 8.5],
                        ["qwen3.5-2b-strict", "vi", "heavy", 6.3, 18.5, 5.0, 3.0, 91.0, 90.0, 8.9],
                        ["baseline", "en", "light", 4.8, 15.0, 0.0, 0.0, 55.0, 80.0, ""],
                        ["coedit-large", "en", "light", 3.1, 10.2, 5.0, 4.0, 85.0, 92.0, ""],
                        ["qwen3.5-2b-basic", "en", "light", 2.1, 7.0, 8.0, 6.0, 84.0, 94.0, ""],
                        ["qwen3.5-2b-strict", "en", "light", 2.2, 7.4, 2.5, 1.5, 96.0, 96.0, ""],
                        ["baseline", "en", "heavy", 11.4, 32.0, 0.0, 0.0, 30.0, 62.0, ""],
                        ["coedit-large", "en", "heavy", 7.9, 22.5, 6.5, 5.0, 74.0, 85.0, ""],
                        ["qwen3.5-2b-basic", "en", "heavy", 5.4, 15.5, 10.0, 7.5, 79.0, 90.0, ""],
                        ["qwen3.5-2b-strict", "en", "heavy", 5.6, 16.0, 4.0, 2.0, 93.0, 93.0, ""],
                    ]
                )

            analyze.main(
                [
                    "--summary",
                    str(summary_csv),
                    "--pred_dir",
                    str(pred_dir),
                    "--data_dir",
                    str(data_dir),
                    "--terms",
                    str(terms_csv),
                    "--out",
                    str(out_dir),
                ]
            )

            expected_outputs = [
                "rq1_cer_wer_comparison.png",
                "rq1_cer_wer_comparison.pdf",
                "rq2_safety_and_preservation.png",
                "rq2_safety_and_preservation.pdf",
                "rq2_cer_scatter_before_after.png",
                "rq3_vi_diacritic_and_error_types.png",
                "error_types_distribution.png",
                "rq4_basic_vs_strict_tradeoff.png",
                "rq4_basic_vs_strict_tradeoff.pdf",
                "error_types_breakdown.csv",
                "typical_examples.md",
                "typical_examples.json",
            ]
            for name in expected_outputs:
                p = out_dir / name
                self.assertTrue(p.exists(), f"Thiếu file đầu ra: {name}")
                self.assertGreater(p.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
