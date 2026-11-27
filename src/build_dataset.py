#!/usr/bin/env python3
"""R1 · PDF số → ảnh dòng → nhiễu light/heavy → Tesseract → JSONL (C3).

Chạy từ gốc repo, xem README.md về sources.csv và bộ mẫu ngày 1–2.
Không dùng OCR làm ground truth; không tự gắn error_types.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import random
import re
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import PIL
import pymupdf
from PIL import Image, ImageFilter, ImageOps


LANGUAGES = {"vi": "vie", "en": "eng"}
NOISE_SETTINGS = {
    "light": {"dpi": 150, "blur": 0.3, "jpeg_quality": 60},
    "heavy": {"dpi": 100, "blur": 0.6, "jpeg_quality": 30,
              "rotation_degrees": 1.0, "salt_pepper_fraction": 0.01},
}


@dataclass(frozen=True)
class Candidate:
    doc_id: str
    lang: str
    filename: str
    page: int  # Số trang PDF, bắt đầu từ 1 (không phải số in trên trang).
    bbox: tuple[float, float, float, float]
    target: str

    @property
    def has_number(self) -> bool:
        return bool(re.search(r"\d", self.target))


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", " ".join(text.split()))


def stable_seed(seed: int, *parts: str) -> int:
    payload = json.dumps([seed, *parts], ensure_ascii=False).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def load_sources(path: Path, pdf_dir: Path) -> list[dict[str, str]]:
    required = {"doc_id", "company", "year", "report_type", "lang", "filename", "url"}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: cần các cột {', '.join(sorted(required))}")
        sources = list(reader)
    seen = set()
    for row in sources:
        if any(not row.get(key, "").strip() for key in required):
            raise ValueError(f"Nguồn thiếu trường bắt buộc: {row.get('doc_id')}")
        if row["doc_id"] in seen or not re.fullmatch(r"[A-Za-z0-9_-]+", row["doc_id"]):
            raise ValueError(f"doc_id trùng hoặc không hợp lệ: {row['doc_id']}")
        seen.add(row["doc_id"])
        if row["lang"] not in LANGUAGES:
            raise ValueError(f"Ngôn ngữ không hỗ trợ: {row['lang']}")
        pdf = (pdf_dir / row["filename"]).resolve()
        if not pdf.is_relative_to(pdf_dir.resolve()) or not pdf.is_file():
            raise ValueError(f"PDF phải tồn tại bên trong {pdf_dir}: {row['filename']}")
        actual_hash = file_sha256(pdf)
        if row.get("sha256") and row["sha256"] != actual_hash:
            raise ValueError(f"PDF khác checksum trong sources.csv: {pdf.name}")
        row["sha256"] = actual_hash
    if not sources:
        raise ValueError("sources.csv không có nguồn PDF.")
    return sorted(sources, key=lambda row: row["doc_id"])


def valid_target(text: str) -> bool:
    return (
        5 <= len(text.split()) <= 40
        and any(char.isalpha() for char in text)
        and not any(unicodedata.category(char) in {"Cc", "Co", "Cs", "Cn"} for char in text)
        and "\ufffd" not in text
    )


def intersects(a: tuple, b: tuple) -> bool:
    return bool(pymupdf.Rect(a).intersects(pymupdf.Rect(b)))


def extract_candidates(source: dict[str, str], pdf_dir: Path) -> tuple[list[Candidate], dict]:
    """Lọc bảo thủ bảng có đường kẻ/các ô cùng hàng; QA thủ công vẫn cần theo B2."""
    stats: Counter[str] = Counter()
    candidates: list[Candidate] = []
    with pymupdf.open(pdf_dir / source["filename"]) as doc:
        start = int(source.get("page_start") or 1)
        end = int(source.get("page_end") or len(doc))
        if not 1 <= start <= end <= len(doc):
            raise ValueError(f"Khoảng trang không hợp lệ: {source['doc_id']} {start}–{end}")
        for page_no in range(start, end + 1):
            page = doc[page_no - 1]
            stats["pages"] += 1
            if page.rotation:
                stats["rotated_pages_skipped"] += 1
                continue
            # Không nhận lớp OCR ẩn trên ảnh scan làm đáp án đúng.
            if any(pymupdf.Rect(info["bbox"]).get_area() > page.rect.get_area() * 0.8
                   for info in page.get_image_info()):
                stats["scan_pages_skipped"] += 1
                continue
            lines = [line for block in page.get_text("dict")["blocks"]
                     if block["type"] == 0 for line in block["lines"]
                     if normalize_text("".join(span["text"] for span in line["spans"]))]
            if not lines:
                stats["no_text_pages"] += 1
                continue
            tables = [table.bbox for table in page.find_tables(strategy="lines_strict").tables
                      if table.col_count >= 2]
            for line in lines:
                stats["raw_lines"] += 1
                bbox = tuple(line["bbox"])
                text = normalize_text("".join(span["text"] for span in line["spans"]))
                if not valid_target(text):
                    stats["invalid_text"] += 1
                    continue
                if bbox[1] < page.rect.height * 0.06 or bbox[3] > page.rect.height * 0.94:
                    stats["page_margin"] += 1
                    continue
                if tuple(line["dir"]) != (1.0, 0.0):
                    stats["non_horizontal"] += 1
                    continue
                # Các ô độc lập cùng baseline thường là bảng không kẻ đường.
                center_y = (bbox[1] + bbox[3]) / 2
                same_row = any(
                    other is not line
                    and abs((other["bbox"][1] + other["bbox"][3]) / 2 - center_y)
                    < min(bbox[3] - bbox[1], other["bbox"][3] - other["bbox"][1]) * 0.35
                    and (other["bbox"][2] < bbox[0] or other["bbox"][0] > bbox[2])
                    for other in lines
                )
                spans = line["spans"]
                large_gap = any(right["bbox"][0] - left["bbox"][2] > max(16, left["size"] * 1.8)
                                for left, right in zip(spans, spans[1:]))
                if same_row or large_gap or any(intersects(bbox, table) for table in tables):
                    stats["table_or_columns"] += 1
                    continue
                candidates.append(Candidate(source["doc_id"], source["lang"], source["filename"],
                                            page_no, bbox, text))
    if not candidates:
        raise ValueError(f"{source['doc_id']}: không có dòng PDF số hợp lệ; kiểm tra lớp text/khoảng trang.")
    # Bỏ header/footer lặp còn sót; giữ một bản của dòng nội dung lặp.
    counts = Counter(candidate.target for candidate in candidates)
    seen = set()
    unique = []
    for candidate in candidates:
        if counts[candidate.target] >= 3 or candidate.target in seen:
            stats["repeated_lines"] += 1
            continue
        seen.add(candidate.target)
        unique.append(candidate)
    stats["candidates"] = len(unique)
    return unique, dict(stats)


def sample_and_split(candidates: list[Candidate], lang: str, count: int,
                     dev_count: int, seed: int) -> list[tuple[Candidate, str]]:
    # Loại trùng target xuyên tài liệu trước khi chia để không rò rỉ dev/test.
    unique = {}
    for candidate in sorted(candidates, key=lambda c: (c.doc_id, c.page, c.bbox)):
        if candidate.lang == lang:
            unique.setdefault(candidate.target, candidate)
    pool = list(unique.values())
    rng = random.Random(stable_seed(seed, lang, "sampling"))
    if len(pool) < count:
        raise ValueError(f"{lang}: cần {count} dòng gốc, chỉ có {len(pool)}; bổ sung PDF hoặc giảm --{lang}_lines.")
    if lang == "vi":
        numeric = [c for c in pool if c.has_number]
        text = [c for c in pool if not c.has_number]
        number_count = round(count * 0.2)
        if len(numeric) < number_count or len(text) < count - number_count:
            raise ValueError(f"vi: không đủ quota 20% dòng có số ({len(numeric)} số, {len(text)} văn bản).")
        selected = rng.sample(numeric, number_count) + rng.sample(text, count - number_count)
        rng.shuffle(selected)
    else:
        selected = rng.sample(pool, count)
    return [(candidate, "dev" if i < dev_count else "test") for i, candidate in enumerate(selected)]


def render_line(page: pymupdf.Page, bbox: tuple) -> Image.Image:
    pix = page.get_pixmap(dpi=300, clip=pymupdf.Rect(bbox), colorspace=pymupdf.csRGB, alpha=False)
    return ImageOps.expand(Image.frombytes("RGB", (pix.width, pix.height), pix.samples), border=4, fill="white")


def add_noise(image: Image.Image, level: str, seed: int) -> Image.Image:
    settings = NOISE_SETTINGS[level]
    rng = random.Random(seed)
    ratio = settings["dpi"] / 300
    image = image.convert("RGB").resize(
        (max(1, round(image.width * ratio)), max(1, round(image.height * ratio))), Image.Resampling.LANCZOS)
    if level == "heavy":
        image = image.rotate(rng.uniform(-1, 1), resample=Image.Resampling.BICUBIC,
                             expand=True, fillcolor="white")
    image = image.filter(ImageFilter.GaussianBlur(settings["blur"]))
    if level == "heavy":
        pixels = image.load()
        for pos in rng.sample(range(image.width * image.height),
                              round(image.width * image.height * settings["salt_pepper_fraction"])):
            value = rng.choice((0, 255))
            pixels[pos % image.width, pos // image.width] = (value, value, value)
    compressed = io.BytesIO()
    image.save(compressed, format="JPEG", quality=settings["jpeg_quality"])
    compressed.seek(0)
    with Image.open(compressed) as decoded:
        return decoded.copy()


def check_ocr(langs: set[str]) -> str:
    import pytesseract

    try:
        version = str(pytesseract.get_tesseract_version())
        available = set(pytesseract.get_languages(config=""))
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError("Thiếu Tesseract: cài tesseract-ocr và tesseract-ocr-vie (xem README).") from exc
    missing = {LANGUAGES[lang] for lang in langs} - available
    if missing:
        raise RuntimeError(f"Tesseract thiếu ngôn ngữ: {', '.join(sorted(missing))}")
    return version


def ocr_line(image: Image.Image, lang: str, timeout: float) -> str:
    import pytesseract

    return normalize_text(pytesseract.image_to_string(
        image, lang=LANGUAGES[lang], config="--psm 7", timeout=timeout))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf_dir", type=Path, default=Path("data/pdf"))
    parser.add_argument("--sources", type=Path, default=Path("data/sources.csv"))
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vi_lines", type=int, default=1050, help="Số dòng gốc VI, gồm dev và test.")
    parser.add_argument("--en_lines", type=int, default=550, help="Số dòng gốc EN, gồm dev và test.")
    parser.add_argument("--dev_lines", type=int, default=50, help="Số dòng gốc dev mỗi ngôn ngữ.")
    parser.add_argument("--ocr_timeout", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true", help="Cho phép thay bộ dữ liệu đã có.")
    return parser.parse_args(argv)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main(argv: list[str] | None = None) -> list[Path]:
    args = parse_args(argv)
    counts = {"vi": args.vi_lines, "en": args.en_lines}
    if args.dev_lines < 0 or args.ocr_timeout <= 0 or any(n < 0 for n in counts.values()):
        raise ValueError("Số dòng phải không âm và ocr_timeout phải > 0.")
    active = {lang for lang, count in counts.items() if count}
    if not active or any(args.dev_lines > counts[lang] for lang in active):
        raise ValueError("Cần ít nhất một ngôn ngữ; dev_lines không được vượt số dòng của ngôn ngữ đó.")
    output_names = [f"{lang}_{split}.jsonl" for lang in LANGUAGES for split in ("dev", "test")]
    output_names += ["dataset.meta.json", "line_manifest.jsonl"]
    existing = [name for name in output_names + ["lines/r1"] if (args.out / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Đã có {', '.join(existing)}; chọn --out khác hoặc dùng --overwrite.")
    sources = [row for row in load_sources(args.sources, args.pdf_dir) if row["lang"] in active]
    if active - {row["lang"] for row in sources}:
        raise ValueError("sources.csv thiếu PDF cho ngôn ngữ được yêu cầu.")
    version = check_ocr(active)
    candidates = []
    extraction = {}
    for source in sources:
        rows, stats = extract_candidates(source, args.pdf_dir)
        candidates.extend(rows)
        extraction[source["doc_id"]] = stats
        print(f"[R1] {source['doc_id']}: {len(rows)} dòng hợp lệ", flush=True)
    selected = {lang: sample_and_split(candidates, lang, counts[lang], args.dev_lines, args.seed)
                for lang in sorted(active)}
    args.out.mkdir(parents=True, exist_ok=True)
    # Chỉ thay các JSONL đích sau khi tất cả OCR hoàn tất thành công.
    with tempfile.TemporaryDirectory(prefix=".r1-build-", dir=args.out) as tmp:
        stage = Path(tmp)
        images = stage / "lines/r1"
        images.mkdir(parents=True)
        datasets: dict[str, list[dict]] = {name: [] for name in output_names if name.endswith(".jsonl")}
        manifest = []
        empty_count = unchanged_count = 0
        for lang, items in selected.items():
            for i, (candidate, split) in enumerate(items, 1):
                # ID dựa trên vị trí PDF, không thay khi đổi quota hay thứ tự lấy mẫu.
                key = json.dumps([candidate.doc_id, candidate.page, candidate.bbox, candidate.target],
                                 ensure_ascii=False)
                base_id = f"{lang}_{hashlib.sha256(key.encode()).hexdigest()[:16]}"
                with pymupdf.open(args.pdf_dir / candidate.filename) as doc:
                    clean = render_line(doc[candidate.page - 1], candidate.bbox)
                clean_path = f"lines/r1/{base_id}_clean.png"
                clean.save(stage / clean_path)
                for noise in NOISE_SETTINGS:
                    rec_id = f"{base_id}_{noise}"
                    noisy = add_noise(clean, noise, stable_seed(args.seed, base_id, noise))
                    image_path = f"lines/r1/{rec_id}.png"
                    # PNG giữ đúng pixel sau giải mã JPEG, không nén mất dữ liệu lần hai.
                    noisy.save(stage / image_path)
                    prediction = ocr_line(noisy, lang, args.ocr_timeout)
                    empty_count += not prediction
                    unchanged_count += prediction == candidate.target
                    row = {"id": rec_id, "lang": lang, "split": split, "noise": noise,
                           "doc_id": candidate.doc_id, "page": candidate.page,
                           "has_number": candidate.has_number, "source": prediction,
                           "target": candidate.target, "error_types": []}
                    datasets[f"{lang}_{split}.jsonl"].append(row)
                    manifest.append({"id": rec_id, "filename": candidate.filename,
                                     "page": candidate.page, "bbox": list(candidate.bbox),
                                     "clean_image": clean_path, "ocr_image": image_path,
                                     "ocr_image_sha256": file_sha256(stage / image_path)})
                print(f"[R1] {lang}: {i}/{len(items)} dòng gốc (2 mức nhiễu)", flush=True)
        datasets["line_manifest.jsonl"] = manifest
        for name, rows in datasets.items():
            write_jsonl(stage / name, rows)
        import pytesseract

        metadata = {
            "seed": args.seed, "base_line_counts": counts, "dev_lines_per_language": args.dev_lines,
            "record_counts": {name: len(datasets[name]) for name in output_names[:4]},
            "empty_source_count": empty_count, "unchanged_source_count": unchanged_count,
            "sources": sources, "extraction": extraction,
            "render_dpi": 300, "margin_px": 4, "noise_settings": NOISE_SETTINGS,
            "ocr_config": "--psm 7", "ocr_timeout": args.ocr_timeout,
            "versions": {"python": platform.python_version(), "pymupdf": pymupdf.VersionBind,
                         "pillow": PIL.__version__, "pytesseract": pytesseract.__version__,
                         "tesseract": version},
            "split_policy": "deduplicate target per language; sample with seed; dev first; paired noise stays together",
            "quality_status": "sample_unreviewed; error_types require manual annotation under B2",
            "output_sha256": {name: file_sha256(stage / name) for name in datasets},
        }
        (stage / "dataset.meta.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                target = args.out / path.relative_to(stage)
                target.parent.mkdir(parents=True, exist_ok=True)
                path.replace(target)
    print(f"[R1] Xong {sum(counts.values()) * 2} cặp OCR/target tại {args.out}; "
          f"source rỗng={empty_count}, đã đúng={unchanged_count}.")
    return [args.out / name for name in output_names]


if __name__ == "__main__":
    main()
