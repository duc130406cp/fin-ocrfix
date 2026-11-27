# R3 · Ghi chú CoEdIT (`grammarly/coedit-large`), Baseline & Phân tích biểu đồ

Người phụ trách: **R3 (Seq2seq tiếng Anh & Phân tích/Biểu đồ)**.  
Tham chiếu: `Plan4project.pdf` Phần A4, A5, A6, B4, C1–C6, C8.

---

## 1. Thông tin từ Model Card `grammarly/coedit-large` (xác nhận 11/2026)

| Mục | Giá trị |
|---|---|
| Checkpoint | `grammarly/coedit-large` |
| Mô hình gốc (Base) | `google/flan-t5-large` (~770M tham số; 24 encoder layers + 24 decoder layers, `d_model=1024`, `d_ff=2816`, `vocab_size=32.128`) |
| Kiến trúc & Class load | Encoder–Decoder Seq2Seq (`T5ForConditionalGeneration`), load qua `AutoModelForSeq2SeqLM` + `AutoTokenizer` |
| Bài báo gốc | Raheja et al. (2023), *CoEdIT: Text Editing by Task-Specific Instruction Tuning* (`arXiv:2305.09857`) |
| Tiền tố đầu vào (GEC) | `"Fix grammatical errors in this sentence: "` + `source` (khớp chính xác C4 và C6) |
| `torch_dtype` gốc | `float32` (trong `config.json`) |
| License | **`CC-BY-NC-4.0`** — chỉ sử dụng cho mục đích học tập và nghiên cứu phi thương mại (ghi rõ trong báo cáo theo C4) |
| Giới hạn miền | Mô hình được instruction-tune cho sửa lỗi ngữ pháp/biên tập văn bản (GEC), **không** được huấn luyện chuyên biệt trên nhiễu ký tự OCR (`0/O`, `1/l`, `rn/m`) hay báo cáo tài chính (cần nêu rõ ở mục Giới hạn trong `method.tex` / `conclusion.tex`) |

---

## 2. Quyết định kỹ thuật cho `src/run_seq2seq_en.py`

1. **Tránh lỗi tràn số `NaN` của họ T5 ở `float16` trên GPU Tesla T4:**
   - Khác với Qwen3.5 ở R4, kiến trúc `T5 / Flan-T5` (`is_gated_act=True`, `gelu_new`) được tiền huấn luyện ở `bfloat16`/`float32`. Khi ép sang `torch.float16` trên GPU Tesla T4 (kiến trúc Turing, không có `bfloat16` phần cứng), giá trị kích hoạt dễ vượt ngưỡng `65.504` gây `NaN` và sinh output rỗng hoặc toàn `<pad>`.
   - Với ~770M tham số, trọng số `float32` chỉ chiếm **~3,1 GB VRAM** (vừa thoải mái trên T4 16 GB lẫn GPU cá nhân 6 GB; trên GPU Ampere/Ada như RTX 30xx/40xx, L4, A100 thì `bfloat16` chỉ chiếm **~1,6 GB VRAM**).
   - Quy tắc `dtype="auto"` trong `run_seq2seq_en.py`: dùng `torch.bfloat16` nếu GPU hỗ trợ `bf16` phần cứng (compute capability $\ge 8.0$), ngược lại dùng **`torch.float32`** (trên T4 và CPU). Nếu người dùng truyền tay `--dtype fp16`, script sẽ in cảnh báo rõ ràng và kiểm tra output rỗng.
2. **Định dạng input & Batching:**
   - Ghép trực tiếp `prefix + source` (mặc định `--prefix "Fix grammatical errors in this sentence: "`).
   - Encoder–decoder T5 dùng `tokenizer.padding_side = "right"` và bắt buộc truyền cả `input_ids` lẫn `attention_mask` vào `model.generate(...)` khi `batch_size > 1`.
   - Đặt `truncation=True`, `max_length=512` ở encoder để an toàn với dòng dài (dữ liệu thực tế ở C3 đã lọc 5–40 từ).
3. **Decoding & Tái lập kết quả (C1):**
   - `seed = 42`, greedy decoding (`do_sample=False`, `num_beams=1`), `max_new_tokens=256`.
4. **Hậu xử lý output:**
   - `skip_special_tokens=True`, gộp dòng/khoảng trắng thừa trên một dòng, gỡ tiền tố bị lặp lại nếu mô hình vô tình chép lại `"Fix grammatical errors in this sentence:"`, chuẩn hoá Unicode `NFC`.
   - Lưu thống kê chạy (`empty_prediction_count`, `prefix_echo_count`, `unchanged_count`, tốc độ `sec_per_line`) vào `results/logs/{stem}.meta.json` đồng bộ với quy ước của R4.
5. **Xuất file predictions & Baseline "không sửa" (C3, C4, A5):**
   - Chạy CoEdIT trên `data/en_test.jsonl` tự động tách theo mức nhiễu thành:
     - `results/predictions/coedit-large_en_light.jsonl`
     - `results/predictions/coedit-large_en_heavy.jsonl`
   - Chế độ `--baseline` (hoặc `--model baseline`) tạo đầy đủ 4 file baseline "không sửa" (`prediction = source`, chuẩn hoá NFC) cho cả tiếng Việt và tiếng Anh mà không cần GPU:
     - `results/predictions/baseline_vi_light.jsonl`
     - `results/predictions/baseline_vi_heavy.jsonl`
     - `results/predictions/baseline_en_light.jsonl`
     - `results/predictions/baseline_en_heavy.jsonl`
   - Hỗ trợ **resume** tự động (giống `src/run_llm.py`), `--no_resume`, `--limit`, và `--dry_run` để test toàn bộ pipeline trên CPU.

---

## 3. Quyết định thiết kế cho `src/analyze.py` (Biểu đồ RQ1–RQ4 & Phân tích lỗi)

Tuân thủ lệnh chạy chuẩn ở C6:
```bash
python src/analyze.py --summary results/summary.csv --pred_dir results/predictions/ --out results/figures/
```

1. **Đồng bộ tuyệt đối với `results/summary.csv` (R5) kèm cơ chế tự tương thích:**
   - Bộ ánh xạ tên cột linh hoạt (nhận diện tự động cả thang tỷ lệ `[0, 1]` và phần trăm `[0, 100]`, chuẩn hoá về `%` trên biểu đồ) để không bị lệch tên cột khi tích hợp với `evaluate.py` của R5.
   - Có sẵn bộ tính metric dự phòng đúng đặc tả C5 (dựa trên `data/*_test.jsonl` + `results/predictions/*.jsonl` + `data/terms.csv` + `results/metrics/*.json`) trong trường hợp cần bổ sung thống kê chi tiết theo từng dòng (scatter plot) hoặc theo 5 nhãn lỗi OCR (`DAU`, `KT`, `SO`, `TACH`, `MAT`) khi `summary.csv` chỉ lưu số tổng hợp.
2. **Danh mục biểu đồ xuất ra `results/figures/` (xuất kép `.png` 300 DPI cho slide/README và `.pdf` vector cho báo cáo LaTeX Overleaf):**
   - `rq1_cer_wer_comparison.{png,pdf}`: So sánh CER (%) và WER (%) giữa Baseline, Seq2seq (BARTpho / CoEdIT) và LLM (Qwen3.5) ở 2 ngôn ngữ × 2 mức nhiễu (`light`, `heavy`), kèm mức cải thiện CER tương đối.
   - `rq2_safety_and_preservation.{png,pdf}`: So sánh 4 chỉ số an toàn tài chính — Bảo toàn số liệu (%), Bảo toàn thuật ngữ (%), Tỷ lệ dòng bị làm tệ hơn / sửa quá tay (%), và Tỷ lệ làm hỏng dòng vốn đã đúng (%).
   - `rq2_cer_scatter_before_after.{png,pdf}`: Biểu đồ phân tán CER từng dòng trước và sau hậu xử lý so với đường chéo $y = x$ (theo phong cách bài *No Free Lunches*, 2025), trực quan hoá vùng cải thiện và vùng sửa quá tay.
   - `rq3_vi_diacritic_and_error_types.{png,pdf}`: Tỷ lệ lỗi dấu tiếng Việt (%) giữa các mô hình và tỷ lệ khôi phục hoàn toàn (`exact match`) theo 5 nhãn lỗi OCR (`DAU`, `KT`, `SO`, `TACH`, `MAT`) trên 100 dòng gắn nhãn.
   - `rq4_basic_vs_strict_tradeoff.{png,pdf}`: Đối chứng trực tiếp prompt `basic` vs `strict` trên LLM (độ chính xác CER vs Bảo toàn số liệu, Hỏng dòng đúng, Sửa quá tay).
3. **Bảng thống kê 5 nhãn lỗi & Trích xuất ví dụ lỗi điển hình:**
   - `error_types_breakdown.csv`: Thống kê số lượng dòng, tần suất và tỷ lệ sửa đúng hoàn toàn của từng mô hình trên 5 nhãn `DAU`, `KT`, `SO`, `TACH`, `MAT`.
   - `typical_examples.md` & `typical_examples.json`: Tự động lọc các cặp ví dụ đối chiếu đa mô hình theo 6 nhóm hiện tượng (sửa thành công tốt, làm hỏng số liệu tài chính, làm hỏng dòng đúng, đối chứng `basic` vs `strict` cứu số liệu ở RQ4, CoEdIT trên lỗi ký tự OCR tiếng Anh, và ví dụ đại diện cho từng nhãn trong 5 loại lỗi) để đưa thẳng vào `results.tex` và slide.

---

## 4. Lộ trình thao tác cho R3 theo tiến độ nhóm

1. **Đã hoàn thành (Tuần 1):**
   - Qua toàn bộ unit test CPU + render biểu đồ (`tests/test_run_seq2seq_en.py`, `tests/test_analyze.py`).
   - Đã tải và chạy thử thật `grammarly/coedit-large` lẫn kiểm thử end-to-end `src/run_seq2seq_en.py` trên GPU RTX 4050 (xem mục 5 bên dưới).
2. **Khi R1 đẩy `data/en_dev.jsonl`, `data/vi_test.jsonl`, `data/en_test.jsonl`:**
   - Mở `notebooks/r3_run_all.ipynb` (hoặc chạy CLI trực tiếp trên RTX 4050 / Colab T4) để xuất ngay 4 file baseline `baseline_{vi,en}_{light,heavy}.jsonl` và 2 file `coedit-large_en_{light,heavy}.jsonl`, rồi commit đẩy lên GitHub cho R5 chấm `evaluate.py`.
3. **Khi R5 cập nhật `results/summary.csv` (và R1 cập nhật 100 dòng `error_types`):**
   - Chạy `python src/analyze.py --summary results/summary.csv --pred_dir results/predictions/ --out results/figures/` để sinh toàn bộ biểu đồ và bảng ví dụ cho Tuần 3 viết `method.tex` & `results.tex`.

---

## 5. Kết quả chạy thử thực tế (27/11/2026 trên GPU RTX 4050 Laptop)

- **Môi trường:** `NVIDIA GeForce RTX 4050 Laptop GPU` (6.141 MiB), Python `3.14.6`, `torch 2.11.0+cu128`, `transformers 5.12.1`, `sentencepiece 0.2.1`.
- **Class load được:** `AutoModelForSeq2SeqLM` (`T5ForConditionalGeneration`) + `AutoTokenizer` (chạy ổn định kể cả khi không cài `accelerate`).
- **dtype dùng:** `torch.bfloat16` (tự chọn bởi `--dtype auto` trên GPU Ada/Ampere; trên GPU T4 Colab sẽ tự chọn `torch.float32`).
- **Thời gian load từ cache cục bộ:** `3.8s`.
- **VRAM:** sau khi load `1.50 GB`, đỉnh `1.54 GB` (dư nhiều trên cả RTX 4050 6 GB lẫn T4 16 GB).
- **Tính nhất quán batch & CLI:**
  - Output giữa `batch=1` và `batch=6` **khớp 100%** (`batch_consistent=True`), xác nhận padding bên phải + `attention_mask` chính xác.
  - Chạy end-to-end thật `src/run_seq2seq_en.py` (`--batch_size 4`): `empty_prediction_count = 0`, `prefix_echo_count = 0`, tách đúng 2 file `coedit-large_en_light.jsonl` & `coedit-large_en_heavy.jsonl` kèm `.meta.json`.
- **Tốc độ & ước tính chạy full:**
  - `batch=1`: `1.177 s/dòng`
  - `batch=6`: **`0.248 s/dòng`** → ước tính chạy full **1.000 dòng `en_test`** (500 light + 500 heavy) trên RTX 4050 chỉ mất **~4,1 phút**.
- **Kết quả định tính trên 6 câu báo cáo tài chính tiếng Anh tự soạn (khớp hoàn toàn target: `3/6 = 50%`):**
  - ✅ **`TACH` (gộp từ):** `cashequivalents` → `cash equivalents` (sửa đúng 100%).
  - ✅ **`MAT` (thiếu ký tự):** `financia statements` → `financial statements` (sửa đúng 100%).
  - ✅ **`CLEAN` (thuật ngữ):** `Retained earnings and EBITDA remained positive in Q4 2025.` → giữ nguyên chính xác.
  - ⚠️ **`KT` (`rn → m`):** sửa được `cornpared` → `compared`, nhưng bỏ sót từ viết hoa `Cornpany's` (coi là danh từ riêng).
  - ⚠️ **`SO` (`O → 0`, `l → 1`):** giữ nguyên lỗi `1,25O` và `l5%` (không nhận diện lỗi ký tự OCR trong số), đồng thời **chèn thêm mạo từ `"The"`** ở đầu câu (`Profit after tax` → `The profit after tax`).
  - ⚠️ **`CLEAN` (số liệu đúng):** `Cost of goods sold decreased by 8% compared to 2024.` → giữ nguyên số `8%` và `2024`, nhưng **làm hỏng dòng vốn đã đúng** do tự thêm `"The"` ở đầu tên chỉ tiêu (`The cost of goods sold...`).
- **Tín hiệu sớm cho `method.tex` / `results.tex`:** Thực nghiệm xác nhận đúng giả thuyết ở C4 — vì `grammarly/coedit-large` được instruction-tune cho GEC (ngữ pháp) chứ không phải nhiễu OCR tài chính, mô hình xử lý tốt lỗi chính tả/tách-gộp từ vựng tiếng Anh nhưng yếu ở lỗi nhầm chữ–số (`O/0`, `l/1`) và có xu hướng **sửa quá tay phong cách báo cáo tài chính** (thêm mạo từ `"The"` vào đầu câu chỉ tiêu kế toán).
- **Ghi chú kỹ thuật:** Nếu tải checkpoint trên mạng Việt Nam bị treo ở `hf-xet`, đặt biến môi trường `HF_HUB_DISABLE_XET=1` để tải bằng HTTP tiêu chuẩn. Warning `tie_word_embeddings` của `transformers 5.x` là cảnh báo mặc định từ cấu hình gốc của checkpoint và không ảnh hưởng output.
