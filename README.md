# fin-ocrfix

Hậu xử lý lỗi OCR cho báo cáo tài chính Việt–Anh bằng Transformer, so sánh seq2seq
và LLM nhỏ về độ chính xác, bảo toàn số liệu và thuật ngữ. Đặc tả gốc:
[`references/Plan4project.pdf`](references/Plan4project.pdf).

## R1 — dữ liệu mẫu tuần 1, ngày 1–2

Đã có `src/build_dataset.py` và **50 cặp OCR/target thật** từ hai PDF số báo cáo
thường niên Hòa Phát 2024 (VI + EN), phần thuyết minh BCTC ở trang PDF 115–128.
Nguồn, khoảng trang và SHA-256 nằm trong [`data/sources.csv`](data/sources.csv).
Đây là dữ liệu thử tích hợp, **chưa phải tập đánh giá đã chốt** của tuần 1 ngày 3–7.

| File | Dòng gốc | Cặp OCR/target, gồm light + heavy |
|---|---:|---:|
| `data/vi_dev.jsonl` | 3 | 6 |
| `data/vi_test.jsonl` | 10 | 20 |
| `data/en_dev.jsonl` | 3 | 6 |
| `data/en_test.jsonl` | 9 | 18 |

### Môi trường

Python 3.10+. Môi trường dùng để kiểm chứng trên máy hiện tại:
`/home/armordkay/venvs/ai/bin/python` (Python 3.12.3).

```bash
source /home/armordkay/venvs/ai/bin/activate
python -m pip install -r requirements-r1.txt
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-vie
tesseract --list-langs
```

Tesseract cần có `vie` và `eng`; phiên bản đã chạy là **5.3.4**. File dependency
trên chỉ dành cho R1; chưa thay thế môi trường inference và phân tích của R2–R5.
Phiên bản Python, PyMuPDF, Pillow, pytesseract và Tesseract được ghi lại trong
`data/dataset.meta.json`.

### Tải nguồn và tái tạo bộ mẫu

Chạy từ thư mục gốc repo. PDF gốc và ảnh dòng nằm trong `.gitignore`.

```bash
mkdir -p data/pdf
curl -fL 'https://file.hoaphat.com.vn/hoaphat-com-vn/2025/04/bao-cao-thuong-nien-hpg-2024.pdf' -o data/pdf/HPG_2024_BCTN_VI.pdf
curl -fL 'https://file.hoaphat.com.vn/hoaphat-com-vn/2025/04/hpg-annual-report-2024.pdf' -o data/pdf/HPG_2024_BCTN_EN.pdf

python src/build_dataset.py --pdf_dir data/pdf --sources data/sources.csv \
  --out /tmp/fin-ocrfix-r1-sample --seed 42 \
  --vi_lines 13 --en_lines 12 --dev_lines 3
```

`--vi_lines` và `--en_lines` đếm **dòng gốc**, bao gồm dev + test; mỗi dòng sinh
hai bản ghi nhiễu. `--dev_lines` là số dòng gốc dev **mỗi ngôn ngữ**. Chọn `--out`
mới khi chạy lại; script từ chối ghi đè nếu không có `--overwrite`.

Để chạy riêng một ngôn ngữ, đặt số dòng của ngôn ngữ còn lại bằng `0`.
Khi có đủ nguồn cho tuần 1 ngày 3–7, lệnh mặc định dùng 1.050 dòng VI và 550 dòng
EN, tách 50 dòng dev mỗi ngôn ngữ. Hai PDF mẫu hiện tại chưa đủ quota này; cần
bổ sung nguồn của 8–10 doanh nghiệp theo B1 và cập nhật `sources.csv`.

### Schema và truy vết

Mỗi dòng trong bốn file JSONL có đúng các trường C3:

```json
{"id":"vi_<hash>_light","lang":"vi","split":"dev","noise":"light","doc_id":"HPG_2024_BCTN_VI","page":119,"has_number":false,"source":"...","target":"...","error_types":[]}
```

- `page` là số trang PDF, bắt đầu từ 1. `target` lấy từ lớp text, chuẩn hóa NFC và khoảng trắng.
- `id` có hậu tố mức nhiễu, duy nhất trên cả bộ; phần hash dựa trên tài liệu, trang, bbox và target.
- Hai mức nhiễu của cùng dòng luôn ở cùng split. Loại target trùng theo ngôn ngữ trước khi chia dev/test.
- `error_types` để trống đến khi có người gắn nhãn theo B2/B4. Giữ cả source rỗng hoặc đã đúng để thống kê trung thực.
- `line_manifest.jsonl` lưu bbox, đường dẫn ảnh sạch/ảnh OCR và checksum ảnh.
- `dataset.meta.json` lưu thông số, nguồn, thống kê lọc, phiên bản và checksum đầu ra.

Chi tiết quyết định, giới hạn và kiểm chứng: [`docs/r1_dataset_notes.md`](docs/r1_dataset_notes.md).

### Kiểm tra

```bash
python -m unittest discover -s tests -p test_build_dataset.py -v
python -m unittest discover -s tests -v
```

Test R1 dùng PDF tạo tại thư mục tạm, chạy thật bước trích dòng/render/nhiễu và
mock lời gọi OCR trong test pipeline; bộ mẫu trong `data/` dùng Tesseract thật.
Test xuất biểu đồ cần `numpy`, `pandas`, `matplotlib` của phần R3.

### Các thành phần hiện có

- `src/run_seq2seq_en.py`: CoEdIT tiếng Anh và baseline không sửa VI/EN.
- `src/run_llm.py`: Qwen với các prompt `basic`/`strict` trong `prompts/`.
- `src/analyze.py`: metric tham chiếu, biểu đồ RQ1–RQ4 và phân tích lỗi.
- `notebooks/`, `docs/`: hướng dẫn và ghi chú thực nghiệm R3/R4.
