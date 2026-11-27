#!/usr/bin/env python3
"""R4 · Chạy inference LLM (Qwen3.5-2B / 4B / dự phòng) để hậu xử lý lỗi OCR.

Tuân thủ đặc tả Plan4project Phần C1, C4, C6, C7:
- Chỉ inference, không fine-tune, không gọi API trả phí.
- Seed cố định 42, greedy decoding (do_sample=False), max_new_tokens=256.
- Chỉ truyền văn bản (không ảnh), tắt chế độ thinking tường minh (enable_thinking=False).
- Hậu xử lý output (bỏ <think>, lấy dòng đầu không rỗng, bỏ tiền tố & ngoặc kép, chuẩn hoá NFC)
  và ghi nhận số lượng format_error / has_think vào results/logs/.
- Xuất file predictions đúng schema: results/predictions/{model}_{lang}_{noise}.jsonl
  với các trường: id, source, prediction.

Ví dụ chạy (C6):
    python src/run_llm.py --model Qwen/Qwen3.5-2B --data data/vi_test.jsonl --prompt prompts/vi_basic.txt
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import re
import time
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

THINK_MARKERS = ("<think>", "</think>", "Thinking Process", "<|im_start|>think")
PREFIX_RE = re.compile(
    r"^\s*(Dòng đã sửa|Dong da sua|Corrected line|Corrected|Output)\s*:\s*",
    re.IGNORECASE,
)
QUOTE_PAIRS = (('"', '"'), ("“", "”"), ("'", "'"), ("«", "»"), ("`", "`"))

MODEL_SHORTHANDS = {
    "qwen3.5-2b": "Qwen/Qwen3.5-2B",
    "qwen3.5-4b": "Qwen/Qwen3.5-4B",
    "qwen2.5-1.5b-instruct": "Qwen/Qwen2.5-1.5B-Instruct",
}


def resolve_model_id(model_arg: str) -> str:
    """Cho phép truyền tên ngắn (vd: Qwen3.5-2B) hoặc full HF repo id."""
    cleaned = model_arg.strip()
    if "/" in cleaned:
        return cleaned
    return MODEL_SHORTHANDS.get(cleaned.lower(), cleaned)


def slugify_model(model_id: str) -> str:
    """Chuyển HF repo id thành tên ngắn gọn, an toàn cho tên file."""
    base = model_id.strip().rstrip("/").split("/")[-1].lower()
    return re.sub(r"[^a-z0-9._-]+", "-", base)


def infer_prompt_variant(prompt_path: Path) -> str:
    """Trích tên biến thể prompt từ tên file (vd: vi_basic.txt -> basic)."""
    stem = prompt_path.stem.lower()
    for variant in ("strict", "basic"):
        if stem.endswith(f"_{variant}") or stem == variant:
            return variant
    return re.sub(r"[^a-z0-9-]+", "-", stem)


def build_output_stem(model_id: str, prompt_path: Path, lang: str, noise: str) -> str:
    """Tạo tên file chuẩn {model}_{lang}_{noise} kèm biến thể prompt cho RQ4."""
    model_slug = slugify_model(model_id)
    variant = infer_prompt_variant(prompt_path)
    return f"{model_slug}-{variant}_{lang}_{noise}"


def fill_prompt(template: str, source: str) -> str:
    """Thay {source} bằng str.replace (tránh lỗi .format khi dòng OCR chứa dấu ngoặc nhọn)."""
    if "{source}" not in template:
        raise ValueError("File prompt thiếu placeholder '{source}'.")
    return template.replace("{source}", source)


def has_think_marker(raw_output: str) -> bool:
    return any(marker in raw_output for marker in THINK_MARKERS)


def postprocess_output(output: str) -> tuple[str, bool]:
    """Hậu xử lý output theo C7.

    Trả về (prediction, format_error).
    format_error=True khi phải can thiệp (có <think>, nhiều dòng, có tiền tố, có ngoặc bao ngoài, hoặc rỗng).
    """
    fmt_err = False
    text = output

    if "<think>" in text or "</think>" in text:
        fmt_err = True
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # Trường hợp bị cắt giữa chừng chưa kịp đóng </think>
        text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "", True

    if len(lines) > 1:
        fmt_err = True

    line = lines[0]

    match = PREFIX_RE.match(line)
    if match:
        line = line[match.end() :].strip()
        fmt_err = True

    for left_q, right_q in QUOTE_PAIRS:
        if len(line) >= 2 and line.startswith(left_q) and line.endswith(right_q):
            line = line[1:-1].strip()
            fmt_err = True
            break

    line = unicodedata.normalize("NFC", line)
    if not line:
        fmt_err = True
    return line, fmt_err


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


def resolve_dtype(dtype_arg: str):
    import torch

    if dtype_arg == "fp32":
        return torch.float32
    if dtype_arg == "fp16":
        return torch.float16
    if dtype_arg == "bf16":
        return torch.bfloat16
    # auto
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


def load_model_and_tokenizer(model_id: str, dtype_arg: str):
    """Load model theo thứ tự khuyến nghị từ model card Qwen3.5 (và tương thích dự phòng Qwen2.5)."""
    import torch
    import transformers
    from transformers import AutoProcessor, AutoTokenizer

    dtype = resolve_dtype(dtype_arg)
    device_map = "auto" if torch.cuda.is_available() else None

    model = None
    loaded_class = None
    errors: dict[str, str] = {}

    for cls_name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText", "AutoModelForCausalLM"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            errors[cls_name] = "không tồn tại trong phiên bản transformers hiện tại"
            continue
        try:
            kwargs: dict[str, Any] = {"dtype": dtype}
            if device_map is not None:
                kwargs["device_map"] = device_map
            try:
                model = cls.from_pretrained(model_id, **kwargs)
            except TypeError:
                # Tương thích transformers cũ hơn dùng torch_dtype
                kwargs.pop("dtype", None)
                kwargs["torch_dtype"] = dtype
                model = cls.from_pretrained(model_id, **kwargs)
            loaded_class = cls_name
            break
        except Exception as exc:  # noqa: BLE001
            errors[cls_name] = f"{type(exc).__name__}: {exc}"

    if model is None:
        detail = "\n".join(f"  - {k}: {v}" for k, v in errors.items())
        raise RuntimeError(
            f"Không load được checkpoint '{model_id}'. Chi tiết lỗi:\n{detail}\n"
            "Gợi ý: nâng cấp transformers lên bản mới nhất (hoặc git+main) hoặc kiểm tra phương án dự phòng ở C4."
        )

    model.eval()
    is_multimodal = loaded_class in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText")

    tokenizer = None
    if is_multimodal:
        try:
            processor = AutoProcessor.from_pretrained(model_id)
            tokenizer = getattr(processor, "tokenizer", processor)
        except Exception:  # noqa: BLE001
            tokenizer = None

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Bắt buộc padding bên trái cho decoder-only LLM khi chạy batch > 1
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, loaded_class, dtype, is_multimodal


def render_chat_prompt(tokenizer, user_text: str, is_multimodal: bool) -> str:
    """Áp chat template, luôn chỉ dùng text và tắt thinking tường minh nếu template hỗ trợ."""
    if is_multimodal:
        messages = [{"role": "user", "content": [{"type": "text", "text": user_text}]}]
    else:
        messages = [{"role": "user", "content": user_text}]

    base_kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }

    # Truyền enable_thinking=False tường minh theo quyết định R4
    sig = inspect.signature(tokenizer.apply_chat_template)
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    try:
        if accepts_kwargs or "enable_thinking" in sig.parameters:
            return tokenizer.apply_chat_template(messages, enable_thinking=False, **base_kwargs)
        return tokenizer.apply_chat_template(messages, **base_kwargs)
    except TypeError:
        # Fallback nếu template chỉ nhận content dạng chuỗi hoặc không hỗ trợ enable_thinking
        plain_messages = [{"role": "user", "content": user_text}]
        return tokenizer.apply_chat_template(plain_messages, **base_kwargs)


def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    is_multimodal: bool,
    max_new_tokens: int,
) -> list[tuple[str, str]]:
    """Sinh greedy cho một batch, trả về danh sách (raw_with_special, clean_without_special)."""
    import torch

    rendered = [render_chat_prompt(tokenizer, p, is_multimodal=is_multimodal) for p in prompts]
    inputs = tokenizer(rendered, return_tensors="pt", padding=True).to(model.device)
    input_len = inputs["input_ids"].shape[1]

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    # Xoá các tham số sampling mặc định trong generation_config để tránh warning khi do_sample=False
    gen_config = getattr(model, "generation_config", None)
    if gen_config is not None:
        for attr in ("temperature", "top_p", "top_k", "min_p"):
            if hasattr(gen_config, attr):
                gen_kwargs[attr] = None

    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)

    results: list[tuple[str, str]] = []
    for row_idx in range(outputs.shape[0]):
        new_tokens = outputs[row_idx, input_len:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=False)
        clean = tokenizer.decode(new_tokens, skip_special_tokens=True)
        results.append((raw, clean))
    return results


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


def infer_default_lang(data_path: Path, prompt_path: Path) -> str:
    for name in (data_path.stem.lower(), prompt_path.stem.lower()):
        if name.startswith("vi_") or "_vi_" in name:
            return "vi"
        if name.startswith("en_") or "_en_" in name:
            return "en"
    return "vi"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chạy inference LLM (Qwen3.5) hậu xử lý lỗi OCR cho báo cáo tài chính (R4)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3.5-2B",
        help="Hugging Face checkpoint ID (mặc định: Qwen/Qwen3.5-2B).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="File dữ liệu đầu vào .jsonl (vd: data/vi_test.jsonl).",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        required=True,
        help="File prompt .txt (vd: prompts/vi_basic.txt).",
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
        default=4,
        help="Batch size khi inference (mặc định: 4; giảm xuống 1 nếu thiếu VRAM).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Số token sinh tối đa theo C1 (mặc định: 256).",
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
        choices=("auto", "fp16", "bf16", "fp32"),
        default="auto",
        help="Độ chính xác trọng số (mặc định: auto -> bf16 trên Ampere+, fp16 trên T4).",
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
        help="Chạy thử toàn bộ pipeline (đọc dữ liệu, ghép prompt, hậu xử lý, ghi file) KHÔNG load model GPU.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("--batch_size phải >= 1")

    set_seed(args.seed)
    model_id = resolve_model_id(args.model)
    prompt_template = args.prompt.read_text(encoding="utf-8")
    if "{source}" not in prompt_template:
        raise ValueError(f"Prompt {args.prompt} thiếu placeholder '{{source}}'.")

    records = read_jsonl(args.data)
    default_lang = infer_default_lang(args.data, args.prompt)
    grouped = group_records(records, noise_filter=args.noise, default_lang=default_lang)
    if not grouped:
        raise ValueError(f"Không có dòng nào phù hợp trong {args.data} (noise={args.noise}).")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    model = tokenizer = loaded_class = dtype = None
    is_multimodal = False
    gpu_name = "dry_run"
    torch_version = transformers_version = "n/a"

    if not args.dry_run:
        import torch
        import transformers

        torch_version = torch.__version__
        transformers_version = transformers.__version__
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        print(f"[R4] Loading {model_id} (dtype={args.dtype}, device={gpu_name})...")
        t_load = time.time()
        model, tokenizer, loaded_class, dtype, is_multimodal = load_model_and_tokenizer(
            model_id, dtype_arg=args.dtype
        )
        print(
            f"[R4] Loaded via {loaded_class} ({type(model).__name__}), "
            f"dtype={dtype} in {time.time() - t_load:.1f}s"
        )

    try:
        from tqdm import tqdm  # type: ignore
    except ImportError:
        tqdm = None

    for (lang, noise), group in sorted(grouped.items()):
        if args.limit is not None:
            group = group[: args.limit]

        stem = build_output_stem(model_id, args.prompt, lang=lang, noise=noise)
        out_path = args.out_dir / f"{stem}.jsonl"
        meta_path = args.log_dir / f"{stem}.meta.json"

        existing = {} if args.no_resume else load_existing_predictions(out_path)
        pending = [rec for rec in group if str(rec["id"]) not in existing]

        print(
            f"[R4] {stem}: tổng={len(group)}, đã có={len(existing)}, cần chạy={len(pending)}"
        )

        # Nếu ghi đè thì xoá file cũ trước khi append
        if args.no_resume and out_path.exists():
            out_path.unlink()

        t_start = time.time()
        format_error_count = 0
        think_count = 0
        empty_count = 0
        processed_now = 0

        batches = [
            pending[i : i + args.batch_size]
            for i in range(0, len(pending), args.batch_size)
        ]
        iterator = tqdm(batches, desc=stem, unit="batch") if tqdm is not None else batches

        with out_path.open("a", encoding="utf-8") as fout:
            for batch in iterator:
                sources = [str(rec["source"]) for rec in batch]
                prompts = [fill_prompt(prompt_template, s) for s in sources]

                if args.dry_run:
                    outputs = [(s, s) for s in sources]
                else:
                    outputs = generate_batch(
                        model=model,
                        tokenizer=tokenizer,
                        prompts=prompts,
                        is_multimodal=is_multimodal,
                        max_new_tokens=args.max_new_tokens,
                    )

                for rec, (raw_out, clean_out) in zip(batch, outputs):
                    if has_think_marker(raw_out):
                        think_count += 1
                    pred, fmt_err = postprocess_output(clean_out)
                    if fmt_err:
                        format_error_count += 1
                    if not pred:
                        empty_count += 1

                    out_rec = {
                        "id": rec["id"],
                        "source": rec["source"],
                        "prediction": pred,
                    }
                    fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                    processed_now += 1
                fout.flush()

        elapsed = time.time() - t_start

        # Đảm bảo thứ tự dòng trong file output khớp 100% với tập input
        final_map = load_existing_predictions(out_path)
        ordered_rows = [final_map[str(rec["id"])] for rec in group if str(rec["id"]) in final_map]
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
            "data_path": str(args.data),
            "prompt_path": str(args.prompt),
            "prompt_variant": infer_prompt_variant(args.prompt),
            "lang": lang,
            "noise": noise,
            "split": str(group[0].get("split", "")),
            "seed": args.seed,
            "decoding": "greedy",
            "max_new_tokens": args.max_new_tokens,
            "enable_thinking": False,
            "batch_size": args.batch_size,
            "dry_run": args.dry_run,
            "total_lines": len(group),
            "processed_this_run": processed_now,
            "resumed_lines": len(group) - processed_now,
            "format_error_count": format_error_count,
            "format_error_rate": round(format_error_count / processed_now, 4) if processed_now else 0.0,
            "think_marker_count": think_count,
            "empty_prediction_count": empty_count,
            "elapsed_sec": round(elapsed, 2),
            "sec_per_line": round(elapsed / processed_now, 4) if processed_now else 0.0,
            "output_path": str(out_path),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        print(
            f"[R4] Xong {out_path} ({len(ordered_rows)} dòng) | "
            f"format_error={format_error_count}/{processed_now}, "
            f"think={think_count}/{processed_now}, "
            f"tốc độ={meta['sec_per_line']:.2f}s/dòng"
        )


if __name__ == "__main__":
    main()
