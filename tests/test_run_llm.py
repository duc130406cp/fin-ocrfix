"""Unit tests (chạy hoàn toàn trên CPU) cho src/run_llm.py."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import run_llm  # noqa: E402


class TestRunLLM(unittest.TestCase):
    def test_postprocess_clean(self) -> None:
        pred, err = run_llm.postprocess_output("Doanh thu thuần năm 2025 đạt 1.250 tỷ đồng")
        self.assertEqual(pred, "Doanh thu thuần năm 2025 đạt 1.250 tỷ đồng")
        self.assertFalse(err)

    def test_postprocess_prefix_quotes_multiline(self) -> None:
        raw = '\n\nDòng đã sửa: "Doanh thu thuần"\nGiải thích: đã thêm dấu.'
        pred, err = run_llm.postprocess_output(raw)
        self.assertEqual(pred, "Doanh thu thuần")
        self.assertTrue(err)

    def test_postprocess_english_prefix_and_think(self) -> None:
        raw = "<think>let me check...</think>\nCorrected line: “Net revenue increased 15%”"
        pred, err = run_llm.postprocess_output(raw)
        self.assertEqual(pred, "Net revenue increased 15%")
        self.assertTrue(err)

    def test_postprocess_empty(self) -> None:
        pred, err = run_llm.postprocess_output("   \n\t  ")
        self.assertEqual(pred, "")
        self.assertTrue(err)

    def test_postprocess_nfc_normalization(self) -> None:
        # "thuần" dạng NFD (chưa dựng sẵn) -> NFC
        nfd = "Doanh thu thua\u0302\u0300n"
        pred, err = run_llm.postprocess_output(nfd)
        self.assertEqual(pred, "Doanh thu thuần")
        self.assertFalse(err)

    def test_fill_prompt_with_braces_in_source(self) -> None:
        tmpl = "OCR: {source}\nFixed:"
        out = run_llm.fill_prompt(tmpl, "Chi phí {khác} là 100")
        self.assertEqual(out, "OCR: Chi phí {khác} là 100\nFixed:")

    def test_output_stem_naming(self) -> None:
        stem = run_llm.build_output_stem(
            "Qwen/Qwen3.5-2B", Path("prompts/vi_strict.txt"), lang="vi", noise="heavy"
        )
        self.assertEqual(stem, "qwen3.5-2b-strict_vi_heavy")

        stem_short = run_llm.build_output_stem(
            run_llm.resolve_model_id("Qwen3.5-2B"),
            Path("prompts/en_basic.txt"),
            lang="en",
            noise="light",
        )
        self.assertEqual(stem_short, "qwen3.5-2b-basic_en_light")

    def test_dry_run_end_to_end_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            data_file = tmp_dir / "vi_test.jsonl"
            out_dir = tmp_dir / "predictions"
            log_dir = tmp_dir / "logs"

            rows = [
                {
                    "id": "vi_0001_light",
                    "lang": "vi",
                    "split": "test",
                    "noise": "light",
                    "doc_id": "VNM_2025",
                    "page": 1,
                    "has_number": True,
                    "source": "Doanh thu 1.25O ty",
                    "target": "Doanh thu 1.250 tỷ",
                    "error_types": [],
                },
                {
                    "id": "vi_0001_heavy",
                    "lang": "vi",
                    "split": "test",
                    "noise": "heavy",
                    "doc_id": "VNM_2025",
                    "page": 1,
                    "has_number": True,
                    "source": "Doanh thu 1.25O ty",
                    "target": "Doanh thu 1.250 tỷ",
                    "error_types": [],
                },
            ]
            data_file.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                encoding="utf-8",
            )

            argv = [
                "--model",
                "Qwen/Qwen3.5-2B",
                "--data",
                str(data_file),
                "--prompt",
                str(ROOT / "prompts/vi_basic.txt"),
                "--out_dir",
                str(out_dir),
                "--log_dir",
                str(log_dir),
                "--batch_size",
                "2",
                "--dry_run",
            ]
            run_llm.main(argv)

            light_out = out_dir / "qwen3.5-2b-basic_vi_light.jsonl"
            heavy_out = out_dir / "qwen3.5-2b-basic_vi_heavy.jsonl"
            self.assertTrue(light_out.exists())
            self.assertTrue(heavy_out.exists())

            pred_rows = run_llm.read_jsonl(light_out)
            self.assertEqual(len(pred_rows), 1)
            self.assertEqual(set(pred_rows[0].keys()), {"id", "source", "prediction"})
            self.assertEqual(pred_rows[0]["id"], "vi_0001_light")

            # Chạy lại với resume -> processed_this_run phải bằng 0
            run_llm.main(argv)
            meta = json.loads((log_dir / "qwen3.5-2b-basic_vi_light.meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["processed_this_run"], 0)
            self.assertEqual(meta["resumed_lines"], 1)


if __name__ == "__main__":
    unittest.main()
