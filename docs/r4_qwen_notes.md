# R4 · Ghi chú Qwen3.5-2B & prompt (Tuần 1, ngày 1–2)

Người phụ trách: R4 (Đồng Minh Đức). Tham chiếu: Plan4project C1, C4, C7.

## 1. Thông tin từ model card (đọc 27/11/2026)

| Mục | Giá trị |
|---|---|
| Checkpoint | `Qwen/Qwen3.5-2B` (bản post-trained/instruct). **Không** dùng `Qwen/Qwen3.5-2B-Base` |
| Bản thêm (nếu kịp) | `Qwen/Qwen3.5-4B`, cần mở model card riêng để xác nhận trước khi chạy |
| Kiến trúc | Causal LM + vision encoder; 24 layer lai Gated DeltaNet / Gated Attention; vocab 248.320; context 262K |
| Class load | `AutoModelForMultimodalLM` + `AutoProcessor` (theo code mẫu trên model card) |
| transformers | Cần bản mới nhất; model card gợi ý cài từ nhánh `main` |
| License | Apache-2.0 |
| Thinking | **Mặc định TẮT ở bản 2B** ("operates in non-thinking mode by default"). Không hỗ trợ soft switch `/think`, `/nothink` của Qwen3 |
| Sampling khuyến nghị (non-thinking, text) | `temperature=1.0, top_p=1.0, top_k=20, min_p=0, presence_penalty=2.0` |

Model card **không** nói gì về dtype trên T4 hay `attn_implementation`.

## 2. Quyết định cho `run_llm.py`

1. **Tắt thinking:** vẫn truyền `enable_thinking=False` tường minh vào `apply_chat_template`, dù đó là mặc định, để không phụ thuộc vào thay đổi template sau này. Không dùng `/no_think` vì model card nói không hỗ trợ. Sau khi sinh, kiểm tra output có `<think>` hay không và đếm vào số lỗi định dạng.
2. **Chỉ văn bản:** message dạng `[{"role":"user","content":[{"type":"text","text": prompt}]}]`, không đưa ảnh.
3. **Decoding:** greedy (`do_sample=False`), `max_new_tokens=256` theo C1, **khác** sampling khuyến nghị của model card. Lý do: tái lập kết quả. Cần ghi rõ trong `method.tex`.
4. **dtype:** T4 không có bf16, nên dùng `float16`. Nếu output hỏng (rỗng, NaN, ký tự rác) thì chuyển sang `float32` (khoảng 9 GB, vẫn vừa T4).
5. **Prompt:** toàn bộ nội dung file `prompts/*.txt` (đã gồm few-shot) là một user message. Thay `{source}` bằng `str.replace`, không dùng `.format`.
6. **Hậu xử lý (C7):** bỏ khối `<think>` nếu có, lấy dòng đầu không rỗng, bỏ tiền tố `Dòng đã sửa:` / `Corrected line:`, bỏ ngoặc kép bao ngoài, chuẩn hoá NFC. Gắn cờ `format_error` khi phải can thiệp. Code mẫu có trong notebook, bước 5.
7. **Batch:** khi batch > 1 phải đặt `tokenizer.padding_side = "left"`.

## 3. Prompt nháp (`prompts/`)

- `vi_basic.txt`, `en_basic.txt`: nguyên văn khung ở C7 (bản EN dịch tương đương), kèm 3 ví dụ few-shot: một dòng lỗi dấu/ký tự, một dòng lỗi số, một dòng đã đúng giữ nguyên.
- `vi_strict.txt`, `en_strict.txt`: giống bản basic, thêm câu ràng buộc số liệu của C7. Few-shot giống hệt bản basic, để hai bản chỉ khác đúng câu ràng buộc (RQ4).
- ⚠️ **Few-shot hiện là ví dụ tự soạn.** Tuần 1, ngày 3–7 phải thay bằng dòng thật từ `vi_dev.jsonl` / `en_dev.jsonl`. Tuyệt đối không lấy từ tập test.

## 4. Smoke test: `notebooks/r4_qwen35_smoke_test.ipynb`

Mở trên Colab/Kaggle (GPU T4), chạy từ trên xuống. Notebook làm các việc sau:
1. Cài transformers, load model với lần lượt 3 class và ghi lại class chạy được.
2. So sánh chat template ở 3 chế độ: mặc định, `enable_thinking=False`, `enable_thinking=True`.
3. Sinh thử, phát hiện `<think>` trong output. Có thêm một lần chạy đối chứng bật thinking.
4. Chạy 4 prompt trên 6 dòng tự soạn, kiểm tra hậu xử lý, đo tốc độ.
5. In ra tổng kết để dán vào mục 5 bên dưới.

Phương án dự phòng (C4): nếu sau nửa ngày vẫn không load được hoặc không tắt được thinking, đổi `MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"`, báo nhóm và ghi lý do vào README.

## 5. Kết quả chạy thử (điền sau khi chạy notebook)

- Môi trường (GPU / torch / transformers):
- Class load được:
- dtype dùng:
- VRAM:
- Template mặc định == `enable_thinking=False`:
- Output còn thinking:
- Lỗi định dạng:
- Tốc độ (s/dòng) và ước tính thời gian chạy full:
- Ghi chú / lỗi gặp phải (dán nguyên thông báo lỗi):
