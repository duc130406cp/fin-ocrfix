"""Unit tests (chạy hoàn toàn trên CPU) cho src/run_seq2seq_en.py (R3)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import run_seq2seq_en  # noqa: E402


class TestRunSeq2SeqEn(unittest.TestCase):
    def test_model_resolution_and_stem_naming(self) -> None:
        self.assertEqual(
            run_seq2seq_en.resolve_model_id("coedit-large"),
            "grammarly/coedit-large",
        )
        self.assertEqual(
            run_seq2seq_en.resolve_model_id("grammarly/coedit-large"),
            "grammarly/coedit-large",
        )
        self.assertEqual(
            run_seq2seq_en.build_output_stem("grammarly/coedit-large", "en", "light"),
            "coedit-large_en_light",
        )
        self.assertEqual(
            run_seq2seq_en.build_output_stem("baseline", "vi", "heavy", baseline=True),
            "baseline_vi_heavy",
        )

    def test_format_input_and_postprocess(self) -> None:
        prefix = "Fix grammatical errors in this sentence: "
        inp = run_seq2seq_en.format_seq2seq_input("  Net revenue increased l5%  ", prefix)
        self.assertEqual(
            inp,
            "Fix grammatical errors in this sentence: Net revenue increased l5%",
        )

        # Output sạch bình thường
        pred, echoed = run_seq2seq_en.postprocess_seq2seq_output(
            "Net revenue increased 15%", prefix=prefix
        )
        self.assertEqual(pred, "Net revenue increased 15%")
        self.assertFalse(echoed)

        # Output lặp lại prefix và có khoảng trắng/dòng thừa
        pred_echo, echoed_flag = run_seq2seq_en.postprocess_seq2seq_output(
            "Fix grammatical errors in this sentence:   Net revenue\n  increased 15% ",
            prefix=prefix,
        )
        self.assertEqual(pred_echo, "Net revenue increased 15%")
        self.assertTrue(echoed_flag)

    def test_baseline_generation_vi_and_en(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            vi_file = tmp_dir / "vi_test.jsonl"
            en_file = tmp_dir / "en_test.jsonl"
            out_dir = tmp_dir / "predictions"
            log_dir = tmp_dir / "logs"

            # Gồm cả ký tự NFD để kiểm tra chuẩn hoá NFC trong baseline
            vi_rows = [
                {
                    "id": "vi_0001_light",
                    "lang": "vi",
                    "split": "test",
                    "noise": "light",
                    "source": "Doanh thu thua\u0302\u0300n nam 2025",
                    "target": "Doanh thu thuần năm 2025",
                },
                {
                    "id": "vi_0001_heavy",
                    "lang": "vi",
                    "split": "test",
                    "noise": "heavy",
                    "source": "Doanh thu thuan nam 2O25",
                    "target": "Doanh thu thuần năm 2025",
                },
            ]
            en_rows = [
                {
                    "id": "en_0001_light",
                    "lang": "en",
                    "split": "test",
                    "noise": "light",
                    "source": "Profit after tax reached l5%",
                    "target": "Profit after tax reached 15%",
                },
                {
                    "id": "en_0001_heavy",
                    "lang": "en",
                    "split": "test",
                    "noise": "heavy",
                    "source": "Proflt after tax reached l5%",
                    "target": "Profit after tax reached 15%",
                },
            ]
            vi_file.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in vi_rows) + "\n",
                encoding="utf-8",
            )
            en_file.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in en_rows) + "\n",
                encoding="utf-8",
            )

            written = run_seq2seq_en.main(
                [
                    "--baseline",
                    "--data",
                    str(vi_file),
                    str(en_file),
                    "--out_dir",
                    str(out_dir),
                    "--log_dir",
                    str(log_dir),
                ]
            )
            self.assertEqual(len(written), 4)

            expected_files = [
                "baseline_vi_light.jsonl",
                "baseline_vi_heavy.jsonl",
                "baseline_en_light.jsonl",
                "baseline_en_heavy.jsonl",
            ]
            for fname in expected_files:
                self.assertTrue((out_dir / fname).exists(), f"Thiếu {fname}")

            vi_light_preds = run_seq2seq_en.read_jsonl(out_dir / "baseline_vi_light.jsonl")
            self.assertEqual(len(vi_light_preds), 1)
            self.assertEqual(set(vi_light_preds[0].keys()), {"id", "source", "prediction"})
            self.assertEqual(vi_light_preds[0]["prediction"], "Doanh thu thuần nam 2025")

    def test_coedit_dry_run_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            en_file = tmp_dir / "en_test.jsonl"
            out_dir = tmp_dir / "predictions"
            log_dir = tmp_dir / "logs"

            en_rows = [
                {
                    "id": "en_0001_light",
                    "lang": "en",
                    "split": "test",
                    "noise": "light",
                    "doc_id": "VNM_2025_EN",
                    "page": 5,
                    "has_number": True,
                    "source": "Net revenue reached VND 1,25O billion",
                    "target": "Net revenue reached VND 1,250 billion",
                    "error_types": ["SO"],
                },
                {
                    "id": "en_0002_light",
                    "lang": "en",
                    "split": "test",
                    "noise": "light",
                    "doc_id": "VNM_2025_EN",
                    "page": 6,
                    "has_number": False,
                    "source": "The Cornpany maintained stable operations",
                    "target": "The Company maintained stable operations",
                    "error_types": ["KT"],
                },
                {
                    "id": "en_0001_heavy",
                    "lang": "en",
                    "split": "test",
                    "noise": "heavy",
                    "doc_id": "VNM_2025_EN",
                    "page": 5,
                    "has_number": True,
                    "source": "Net revenue reached VND l,25O blllion",
                    "target": "Net revenue reached VND 1,250 billion",
                    "error_types": ["SO", "KT"],
                },
            ]
            en_file.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in en_rows) + "\n",
                encoding="utf-8",
            )

            argv = [
                "--model",
                "grammarly/coedit-large",
                "--data",
                str(en_file),
                "--prefix",
                "Fix grammatical errors in this sentence: ",
                "--out_dir",
                str(out_dir),
                "--log_dir",
                str(log_dir),
                "--batch_size",
                "2",
                "--dry_run",
            ]
            run_seq2seq_en.main(argv)

            light_path = out_dir / "coedit-large_en_light.jsonl"
            heavy_path = out_dir / "coedit-large_en_heavy.jsonl"
            self.assertTrue(light_path.exists())
            self.assertTrue(heavy_path.exists())

            light_preds = run_seq2seq_en.read_jsonl(light_path)
            self.assertEqual([r["id"] for r in light_preds], ["en_0001_light", "en_0002_light"])
            self.assertEqual(set(light_preds[0].keys()), {"id", "source", "prediction"})

            # Chạy lại với resume -> không xử lý lại dòng nào
            run_seq2seq_en.main(argv)
            meta = json.loads(
                (log_dir / "coedit-large_en_light.meta.json").read_text(encoding="utf-8")
            )
            self.assertEqual(meta["processed_this_run"], 0)
            self.assertEqual(meta["resumed_lines"], 2)


if __name__ == "__main__":
    unittest.main()
