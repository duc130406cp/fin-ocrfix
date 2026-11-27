#!/usr/bin/env python3
"""R3 · Chạy inference Seq2Seq tiếng Anh (CoEdIT) & xuất baseline "không sửa" (VI & EN).

Tuân thủ đặc tả Plan4project Phần A4, A5, C1, C3, C4, C6, C8:
- Mô hình chính: grammarly/coedit-large (Flan-T5-large ~770M, license CC-BY-NC-4.0).
- Tiền tố mặc định: "Fix grammatical errors in this sentence: ".
- Chỉ inference, không fine-tune; seed cố định 42, greedy decoding (do_sample=False, num_beams=1),
  max_new_tokens=256.
- An toàn số học cho họ T5: tự động chọn bfloat16 trên GPU Ampere+ hoặc float32 trên GPU T4/CPU
  (tránh lỗi tràn số NaN của T5 khi ép chạy float16 trên T4).
- Chế độ --baseline (hoặc --model baseline): xuất các file baseline "không sửa"
  (prediction = source, chuẩn hoá NFC) cho cả tiếng Việt và tiếng Anh.
- Xuất đúng schema: results/predictions/{model}_{lang}_{noise}.jsonl với các trường:
  id, source, prediction; kèm file thống kê chạy tại results/logs/{stem}.meta.json.

Ví dụ chạy (C6):
    python src/run_seq2seq_en.py --model grammarly/coedit-large --data data/en_test.jsonl \
        --prefix "Fix grammatical errors in this sentence: "

Xuất toàn bộ baseline "không sửa" cho VI & EN:
    python src/run_seq2seq_en.py --baseline --data data/vi_test.jsonl data/en_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_COEDIT_MODEL = "grammarly/coedit-large"
DEFAULT_COEDIT_PREFIX = "Fix grammatical errors in this sentence: "
DEFAULT_BASELINE_DATASETS = (Path("data/vi_test.jsonl"), Path("data/en_test.jsonl"))

MODEL_SHORTHANDS = {
    "coedit": "grammarly/coedit-large",
    "coedit-large": "grammarly/coedit-large",
    "flan-t5-large": "google/flan-t5-large",
    "baseline": "baseline",
    "none": "baseline",
    "identity": "baseline",
}


def resolve_model_id(model_arg: str) -> str:
    """Cho phép truyền tên ngắn (vd: coedit-large, baseline) hoặc full HF repo id."""
    cleaned = model_arg.strip()
    if "/" in cleaned:
        return cleaned
    return MODEL_SHORTHANDS.get(cleaned.lower(), cleaned)


def is_baseline_mode(baseline_flag: bool, model_id: str) -> bool:
    return baseline_flag or model_id.strip().lower() == "baseline"


def slugify_model(model_id: str) -> str:
    """Chuyển HF repo id thành tên ngắn gọn, an toàn cho tên file (vd: coedit-large)."""
    base = model_id.strip().rstrip("/").split("/")[-1].lower()
    return re.sub(r"[^a-z0-9._-]+", "-", base)


def build_output_stem(model_id: str, lang: str, noise: str, baseline: bool = False) -> str:
    """Tạo tên file chuẩn {model}_{lang}_{noise} theo C3, C4 và A5."""
    slug = "baseline" if baseline else slugify_model(model_id)
    return f"{slug}_{lang}_{noise}"


def format_seq2seq_input(source: str, prefix: str) -> str:
    """Ghép tiền tố điều khiển (instruction prefix) với dòng văn bản OCR."""
    clean_source = source.strip()
    if not prefix:
        return clean_source
    return f"{prefix}{clean_source}"


def postprocess_seq2seq_output(raw_output: str, prefix: str = "") -> tuple[str, bool]:
    """Hậu xử lý output của mô hình Seq2Seq.

    Trả về (prediction, prefix_echoed):
    - Gộp khoảng trắng / xuống dòng thừa thành một dòng đơn.
    - Nếu mô hình vô tình lặp lại prefix ở đầu câu thì gỡ bỏ và đánh dấu prefix_echoed=True.
    - Chuẩn hoá Unicode NFC theo C3/C5.
    """
    prefix_echoed = False
    text = re.sub(r"\s+", " ", raw_output).strip()

    stripped_prefix = prefix.strip()
    if stripped_prefix and text.lower().startswith(stripped_prefix.lower()):
        text = text[len(stripped_prefix) :].strip()
        prefix_echoed = True

    text = unicodedata.normalize("NFC", text)
    return text, prefix_echoed


def normalize_baseline_prediction(source: str) -> str:
    """Baseline 'không sửa' theo C4: giữ nguyên source (đảm bảo chuẩn Unicode NFC)."""
    return unicodedata.normalize("NFC", str(source))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                records.append(json.loads(raw_line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Lỗi JSON ở {path}:{line_no}: {exc}") from exc
    return records


def load_existing_predictions(path: Path) -> dict[str, dict[str, Any]]:
    """Đọc các dòng đã dự đoán trước đó để hỗ trợ resume."""
    existing: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return existing
    for rec in read_jsonl(path):
        rec_id = str(rec.get("id", ""))
        if rec_id:
            existing[rec_id] = rec
    return existing


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_seq2seq_dtype(dtype_arg: str):
    """Chọn dtype an toàn cho kiến trúc T5 / Flan-T5 / CoEdIT.

    Lưu ý quan trọng: T5-large / Flan-T5-large dễ bị tràn số (NaN -> chuỗi rỗng) khi chạy
    ở float16 trên GPU Tesla T4 (không có bfloat16 phần cứng). Ở float32, mô hình 770M
    chỉ chiếm ~3.1 GB VRAM, vừa thoải mái trên T4 16 GB.
    """
    import torch

    if dtype_arg == "fp32":
        return torch.float32
    if dtype_arg == "bf16":
        return torch.bfloat16
    if dtype_arg == "fp16":
        print(
            "[R3][Cảnh báo] Họ mô hình T5/CoEdIT dễ bị tràn số NaN ở float16 trên GPU T4. "
            "Nếu thấy output rỗng, hãy đổi sang --dtype auto hoặc --dtype fp32.",
            file=sys.stderr,
        )
        return torch.float16

    # auto: dùng bfloat16 trên GPU Ampere+ (compute capability >= 8), còn lại dùng float32
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability()[0] >= 8:
            return torch.bfloat16
    return torch.float32


def load_seq2seq_model_and_tokenizer(model_id: str, dtype_arg: str):
    """Load AutoModelForSeq2SeqLM + AutoTokenizer cho CoEdIT (T5ForConditionalGeneration)."""
    import importlib.util
    import torch
    import transformers
    from transformers import AutoTokenizer

    dtype = resolve_seq2seq_dtype(dtype_arg)
    has_cuda = torch.cuda.is_available()
    has_accelerate = importlib.util.find_spec("accelerate") is not None

    model = None
    loaded_class = None
    errors: dict[str, str] = {}

    for cls_name in ("AutoModelForSeq2SeqLM", "T5ForConditionalGeneration"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            errors[cls_name] = "không tồn tại trong phiên bản transformers hiện tại"
            continue

        # Chỉ dùng device_map='auto' khi đã cài accelerate; nếu không thì load chuẩn rồi .to('cuda')
        load_attempts: list[dict[str, Any]] = []
        if has_cuda and has_accelerate:
            load_attempts.append({"device_map": "auto"})
        load_attempts.append({})

        for base_kwargs in load_attempts:
            try:
                kwargs = dict(base_kwargs)
                kwargs["dtype"] = dtype
                try:
                    model = cls.from_pretrained(model_id, **kwargs)
                except TypeError:
                    kwargs.pop("dtype", None)
                    kwargs["torch_dtype"] = dtype
                    model = cls.from_pretrained(model_id, **kwargs)

                if has_cuda and "device_map" not in base_kwargs:
                    model = model.to("cuda")
                loaded_class = cls_name
                break
            except Exception as exc:  # noqa: BLE001
                errors[f"{cls_name}({','.join(base_kwargs.keys()) or 'cpu/manual'})"] = (
                    f"{type(exc).__name__}: {exc}"
                )
        if model is not None:
            break

    if model is None:
        detail = "\n".join(f"  - {k}: {v}" for k, v in errors.items())
        raise RuntimeError(
            f"Không load được checkpoint Seq2Seq '{model_id}'. Chi tiết lỗi:\n{detail}"
        )

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Encoder-decoder T5 dùng right padding chuẩn kết hợp attention_mask
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, loaded_class, dtype


def generate_seq2seq_batch(
    model,
    tokenizer,
    sources: list[str],
    prefix: str,
    max_new_tokens: int,
    max_source_length: int = 512,
) -> list[str]:
    """Sinh greedy (do_sample=False, num_beams=1) cho một batch văn bản."""
    import torch

    formatted_inputs = [format_seq2seq_input(s, prefix) for s in sources]
    encoded = tokenizer(
        formatted_inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_source_length,
    )
    target_device = getattr(model, "device", None)
    if target_device is not None:
        encoded = {k: v.to(target_device) for k, v in encoded.items()}

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
    }
    gen_config = getattr(model, "generation_config", None)
    if gen_config is not None:
        for attr in ("temperature", "top_p", "top_k", "min_p"):
            if hasattr(gen_config, attr):
                gen_kwargs[attr] = None

    with torch.inference_mode():
        outputs = model.generate(**encoded, **gen_kwargs)

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def infer_default_lang(data_path: Path) -> str:
    stem = data_path.stem.lower()
    if stem.startswith("vi_") or "_vi_" in stem or stem == "vi":
        return "vi"
    if stem.startswith("en_") or "_en_" in stem or stem == "en":
        return "en"
    return "en"


def group_records(
    records: list[dict[str, Any]],
    noise_filter: str,
    default_lang: str,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Gom nhóm bản ghi theo (lang, noise) để xuất đúng file {model}_{lang}_{noise}.jsonl."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        lang = str(rec.get("lang") or default_lang).lower()
        noise = str(rec.get("noise") or "unknown").lower()
        if noise_filter != "all" and noise != noise_filter:
            continue
        grouped[(lang, noise)].append(rec)
    return dict(grouped)


def resolve_data_paths(data_args: list[Path] | None, baseline: bool) -> list[Path]:
    """Xác định danh sách file .jsonl đầu vào."""
    if data_args:
        missing = [str(p) for p in data_args if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu đầu vào: {', '.join(missing)}")
        return data_args

    if baseline:
        existing = [p for p in DEFAULT_BASELINE_DATASETS if p.exists()]
        if not existing:
            expected = ", ".join(str(p) for p in DEFAULT_BASELINE_DATASETS)
            raise FileNotFoundError(
                f"Chế độ --baseline không truyền --data và chưa tìm thấy file mặc định ({expected})."
            )
        return existing

    raise ValueError("Cần truyền --data <path_to_jsonl> khi chạy mô hình Seq2Seq.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chạy inference Seq2Seq tiếng Anh (CoEdIT) & tạo baseline không sửa (R3)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_COEDIT_MODEL,
        help=f"Hugging Face checkpoint ID hoặc 'baseline' (mặc định: {DEFAULT_COEDIT_MODEL}).",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Chạy chế độ baseline 'không sửa' (prediction = source) cho các file ở --data, không cần GPU.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        nargs="+",
        default=None,
        help="Một hoặc nhiều file dữ liệu .jsonl đầu vào (vd: data/en_test.jsonl).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=DEFAULT_COEDIT_PREFIX,
        help=f"Tiền tố đầu vào cho CoEdIT (mặc định: {DEFAULT_COEDIT_PREFIX!r}).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("results/predictions"),
        help="Thư mục lưu file predictions .jsonl (mặc định: results/predictions).",
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=Path("results/logs"),
        help="Thư mục lưu file thống kê chạy .meta.json (mặc định: results/logs).",
    )
    parser.add_argument(
        "--noise",
        type=str,
        choices=("all", "light", "heavy"),
        default="all",
        help="Lọc mức nhiễu cần chạy (mặc định: all — tự tách file cho từng mức nhiễu).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size khi inference Seq2Seq (mặc định: 8).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Số token sinh tối đa theo C1 (mặc định: 256).",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=512,
        help="Độ dài tối đa của chuỗi đầu vào encoder (mặc định: 512).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed cố định theo C1 (mặc định: 42).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=("auto", "bf16", "fp32", "fp16"),
        default="auto",
        help="Độ chính xác trọng số (mặc định: auto -> bf16 trên Ampere+, fp32 trên T4/CPU để tránh NaN).",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Chạy lại từ đầu và ghi đè file predictions nếu đã tồn tại.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Chỉ chạy N dòng đầu tiên của mỗi nhóm (dùng để test nhanh).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Chạy thử toàn bộ pipeline (đọc dữ liệu, ghép prefix, hậu xử lý, ghi file) KHÔNG load model GPU.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> list[Path]:
    args = parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("--batch_size phải >= 1")

    set_seed(args.seed)
    model_id = resolve_model_id(args.model)
    baseline = is_baseline_mode(args.baseline, model_id)
    data_paths = resolve_data_paths(args.data, baseline=baseline)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    model = tokenizer = loaded_class = dtype = None
    gpu_name = "baseline_cpu" if baseline else ("dry_run" if args.dry_run else "CPU")
    torch_version = transformers_version = "n/a"

    if not baseline and not args.dry_run:
        import torch
        import transformers

        torch_version = torch.__version__
        transformers_version = transformers.__version__
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        print(f"[R3] Loading {model_id} (dtype={args.dtype}, device={gpu_name})...")
        t_load = time.time()
        model, tokenizer, loaded_class, dtype = load_seq2seq_model_and_tokenizer(
            model_id, dtype_arg=args.dtype
        )
        print(
            f"[R3] Loaded via {loaded_class} ({type(model).__name__}), "
            f"dtype={dtype} in {time.time() - t_load:.1f}s"
        )

    try:
        from tqdm import tqdm  # type: ignore
    except ImportError:
        tqdm = None

    written_files: list[Path] = []

    for data_path in data_paths:
        records = read_jsonl(data_path)
        default_lang = infer_default_lang(data_path)
        grouped = group_records(records, noise_filter=args.noise, default_lang=default_lang)
        if not grouped:
            raise ValueError(f"Không có dòng nào phù hợp trong {data_path} (noise={args.noise}).")

        for (lang, noise), group in sorted(grouped.items()):
            if args.limit is not None:
                group = group[: args.limit]

            stem = build_output_stem(model_id, lang=lang, noise=noise, baseline=baseline)
            out_path = args.out_dir / f"{stem}.jsonl"
            meta_path = args.log_dir / f"{stem}.meta.json"

            if baseline:
                t_start = time.time()
                ordered_rows = []
                empty_count = 0
                for rec in group:
                    pred = normalize_baseline_prediction(str(rec["source"]))
                    if not pred:
                        empty_count += 1
                    ordered_rows.append(
                        {
                            "id": rec["id"],
                            "source": rec["source"],
                            "prediction": pred,
                        }
                    )
                with out_path.open("w", encoding="utf-8") as fout:
                    for row in ordered_rows:
                        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                elapsed = time.time() - t_start
                meta = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "model_id": "baseline",
                    "loaded_class": "IdentityBaseline",
                    "dtype": "n/a",
                    "gpu": gpu_name,
                    "torch_version": torch_version,
                    "transformers_version": transformers_version,
                    "data_path": str(data_path),
                    "prefix": "",
                    "lang": lang,
                    "noise": noise,
                    "split": str(group[0].get("split", "")),
                    "seed": args.seed,
                    "decoding": "identity_copy",
                    "max_new_tokens": 0,
                    "batch_size": len(group),
                    "baseline": True,
                    "dry_run": args.dry_run,
                    "total_lines": len(group),
                    "processed_this_run": len(group),
                    "resumed_lines": 0,
                    "prefix_echo_count": 0,
                    "empty_prediction_count": empty_count,
                    "unchanged_count": len(group),
                    "elapsed_sec": round(elapsed, 4),
                    "sec_per_line": round(elapsed / len(group), 6) if group else 0.0,
                    "output_path": str(out_path),
                }
                meta_path.write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                print(f"[R3][Baseline] Xong {out_path} ({len(ordered_rows)} dòng)")
                written_files.append(out_path)
                continue

            existing = {} if args.no_resume else load_existing_predictions(out_path)
            pending = [rec for rec in group if str(rec["id"]) not in existing]

            print(
                f"[R3] {stem}: tổng={len(group)}, đã có={len(existing)}, cần chạy={len(pending)}"
            )

            if args.no_resume and out_path.exists():
                out_path.unlink()

            t_start = time.time()
            prefix_echo_count = 0
            empty_count = 0
            unchanged_count = 0
            processed_now = 0

            batches = [
                pending[i : i + args.batch_size]
                for i in range(0, len(pending), args.batch_size)
            ]
            iterator = tqdm(batches, desc=stem, unit="batch") if tqdm is not None else batches

            with out_path.open("a", encoding="utf-8") as fout:
                for batch in iterator:
                    sources = [str(rec["source"]) for rec in batch]

                    if args.dry_run:
                        # Mô phỏng toàn bộ bước ghép prefix và gỡ/hậu xử lý trên CPU
                        raw_outputs = [
                            format_seq2seq_input(s, args.prefix)[len(args.prefix) :]
                            if args.prefix
                            else s
                            for s in sources
                        ]
                    else:
                        raw_outputs = generate_seq2seq_batch(
                            model=model,
                            tokenizer=tokenizer,
                            sources=sources,
                            prefix=args.prefix,
                            max_new_tokens=args.max_new_tokens,
                            max_source_length=args.max_source_length,
                        )

                    for rec, raw_out in zip(batch, raw_outputs):
                        pred, echoed = postprocess_seq2seq_output(raw_out, prefix=args.prefix)
                        if echoed:
                            prefix_echo_count += 1
                        if not pred:
                            empty_count += 1
                        if pred == unicodedata.normalize("NFC", str(rec["source"]).strip()):
                            unchanged_count += 1

                        out_rec = {
                            "id": rec["id"],
                            "source": rec["source"],
                            "prediction": pred,
                        }
                        fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                        processed_now += 1
                    fout.flush()

            elapsed = time.time() - t_start

            if processed_now > 0 and empty_count == processed_now and not args.dry_run:
                raise RuntimeError(
                    f"Toàn bộ {processed_now} dòng dự đoán ở {out_path} đều rỗng! "
                    "Nhiều khả năng mô hình T5 bị tràn số NaN do dtype không tương thích. "
                    "Hãy chạy lại với --dtype fp32 --no_resume."
                )

            # Đảm bảo thứ tự dòng trong file output khớp 100% với tập input
            final_map = load_existing_predictions(out_path)
            ordered_rows = [
                final_map[str(rec["id"])] for rec in group if str(rec["id"]) in final_map
            ]
            if len(ordered_rows) != len(group):
                raise RuntimeError(
                    f"Số dòng output ({len(ordered_rows)}) != input ({len(group)}) cho {out_path}"
                )
            with out_path.open("w", encoding="utf-8") as fout:
                for row in ordered_rows:
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")

            meta = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "model_id": model_id,
                "loaded_class": loaded_class,
                "dtype": str(dtype),
                "gpu": gpu_name,
                "torch_version": torch_version,
                "transformers_version": transformers_version,
                "data_path": str(data_path),
                "prefix": args.prefix,
                "lang": lang,
                "noise": noise,
                "split": str(group[0].get("split", "")),
                "seed": args.seed,
                "decoding": "greedy",
                "max_new_tokens": args.max_new_tokens,
                "max_source_length": args.max_source_length,
                "batch_size": args.batch_size,
                "baseline": False,
                "dry_run": args.dry_run,
                "total_lines": len(group),
                "processed_this_run": processed_now,
                "resumed_lines": len(group) - processed_now,
                "prefix_echo_count": prefix_echo_count,
                "empty_prediction_count": empty_count,
                "unchanged_count": unchanged_count,
                "elapsed_sec": round(elapsed, 2),
                "sec_per_line": round(elapsed / processed_now, 4) if processed_now else 0.0,
                "output_path": str(out_path),
            }
            meta_path.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

            print(
                f"[R3] Xong {out_path} ({len(ordered_rows)} dòng) | "
                f"empty={empty_count}/{processed_now}, "
                f"prefix_echo={prefix_echo_count}/{processed_now}, "
                f"tốc độ={meta['sec_per_line']:.2f}s/dòng"
            )
            written_files.append(out_path)

    return written_files


if __name__ == "__main__":
    main()
