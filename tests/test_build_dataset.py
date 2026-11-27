"""R1: kiểm tra bộ lọc PDF, chống rò rỉ, tái lập nhiễu và hợp đồng JSONL."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import build_dataset as builder  # noqa: E402
import pymupdf  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402


def make_pdf(path: Path) -> None:
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((50, 25), "Repeated financial report header should disappear")
        for i in range(12):
            page.insert_text((25, 90 + i * 30), " ")
            page.insert_text((50, 90 + i * 30), f"The company reported net revenue of {2020 + i} billion.")
        # Bảng không kẻ đường: một nhãn dài cùng hàng với hai cột số.
        page.insert_text((50, 500), "Long term cash and cash equivalents")
        page.insert_text((360, 500), "15000")
        page.insert_text((450, 500), "12000")
        page.insert_text((50, 800), "Financial report footer must also disappear")
        doc.save(path)


class TestBuildDataset(unittest.TestCase):
    def test_normalization_and_invalid_text(self) -> None:
        self.assertEqual(builder.normalize_text("  Doanh thu thua\u0302\u0300n\n tăng  "), "Doanh thu thuần tăng")
        self.assertFalse(builder.valid_target("One two three four \ufffd"))
        self.assertFalse(builder.valid_target("123 456 789 123 456"))
        self.assertFalse(builder.valid_target("one two three"))

    def test_extraction_excludes_margins_and_table_cells(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            make_pdf(directory / "report.pdf")
            candidates, stats = builder.extract_candidates(
                {"doc_id": "ACME_EN", "lang": "en", "filename": "report.pdf"}, directory)
            self.assertEqual(len(candidates), 12)
            self.assertTrue(all(c.page == 1 and c.has_number for c in candidates))
            self.assertGreater(stats["table_or_columns"], 0)
            self.assertTrue(all("header" not in c.target and "footer" not in c.target for c in candidates))

    def test_scan_pdf_rejected_even_with_text_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            image = Image.new("RGB", (600, 850), "white")
            image.save(directory / "scan.png")
            with pymupdf.open() as doc:
                page = doc.new_page()
                page.insert_image(page.rect, filename=str(directory / "scan.png"))
                page.insert_text((50, 100), "An unreliable OCR layer should never become gold")
                doc.save(directory / "scan.pdf")
            with self.assertRaisesRegex(ValueError, "PDF số"):
                builder.extract_candidates({"doc_id": "SCAN", "lang": "en", "filename": "scan.pdf"}, directory)

    def test_sampling_deduplicates_before_split_and_enforces_quota(self) -> None:
        pool = [builder.Candidate("A", "vi", "a.pdf", 1, (0, i, 100, i + 1),
                                  f"Doanh thu của công ty đạt {i}" if i < 4 else
                                  "Báo cáo tài chính của công ty " + chr(65 + i)) for i in range(20)]
        duplicates = [builder.Candidate("B", c.lang, "b.pdf", c.page, c.bbox, c.target) for c in pool]
        selected = builder.sample_and_split(pool + duplicates, "vi", 10, 3, 42)
        self.assertEqual(selected, builder.sample_and_split(list(reversed(pool + duplicates)), "vi", 10, 3, 42))
        self.assertEqual(sum(c.has_number for c, _ in selected), 2)
        self.assertEqual(sum(split == "dev" for _, split in selected), 3)
        dev = {c.target for c, split in selected if split == "dev"}
        test = {c.target for c, split in selected if split == "test"}
        self.assertFalse(dev & test)
        with self.assertRaisesRegex(ValueError, "chỉ có"):
            builder.sample_and_split(pool + duplicates, "vi", 21, 3, 42)

    def test_noise_reproducible_and_changes_with_seed(self) -> None:
        image = Image.new("RGB", (700, 80), "white")
        ImageDraw.Draw(image).text((10, 20), "Net revenue reached 15000 billion", fill="black")
        light = builder.add_noise(image, "light", 42)
        self.assertEqual(light.size, (350, 40))
        first = builder.add_noise(image, "heavy", 42)
        self.assertEqual(first.tobytes(), builder.add_noise(image, "heavy", 42).tobytes())
        self.assertNotEqual(first.tobytes(), builder.add_noise(image, "heavy", 43).tobytes())

    def test_pipeline_schema_pairs_reproducibility_and_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            make_pdf(directory / "report.pdf")
            sources = directory / "sources.csv"
            with sources.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["doc_id", "company", "year", "report_type",
                                                            "lang", "filename", "url", "sha256"])
                writer.writeheader()
                writer.writerow(dict(doc_id="ACME_EN", company="ACME", year="2024", report_type="annual",
                                     lang="en", filename="report.pdf", url="https://example.com/report.pdf",
                                     sha256=builder.file_sha256(directory / "report.pdf")))
            args = ["--sources", str(sources), "--pdf_dir", str(directory), "--vi_lines", "0",
                    "--en_lines", "4", "--dev_lines", "1", "--out", str(directory / "out")]
            # Mock chỉ OCR; trích PDF, render, nhiễu và ghi dữ liệu đều chạy thật.
            with patch.object(builder, "check_ocr", return_value="test"), patch.object(
                builder, "ocr_line", return_value="Net revenue remained unchanged"):
                files = builder.main(args)
                before = {p.name: p.read_bytes() for p in files}
                with self.assertRaises(FileExistsError):
                    builder.main(args)
                builder.main(args + ["--overwrite"])
                self.assertEqual(before, {p.name: p.read_bytes() for p in files})
            dev = [json.loads(line) for line in (directory / "out/en_dev.jsonl").read_text().splitlines()]
            test = [json.loads(line) for line in (directory / "out/en_test.jsonl").read_text().splitlines()]
            self.assertEqual((len(dev), len(test)), (2, 6))
            self.assertFalse({r["target"] for r in dev} & {r["target"] for r in test})
            self.assertEqual(len({r["id"] for r in dev + test}), 8)
            for row in dev + test:
                self.assertEqual(set(row), {"id", "lang", "split", "noise", "doc_id", "page",
                                            "has_number", "source", "target", "error_types"})
                self.assertEqual(row["error_types"], [])
            with patch.object(builder, "check_ocr", return_value="test"), patch.object(
                builder, "ocr_line", side_effect=RuntimeError("OCR timed out")):
                with self.assertRaisesRegex(RuntimeError, "timed out"):
                    builder.main(args + ["--overwrite"])
            self.assertEqual(before, {p.name: p.read_bytes() for p in files})
            with (directory / "report.pdf").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                builder.load_sources(sources, directory)


if __name__ == "__main__":
    unittest.main()
