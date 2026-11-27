# fin-ocrfix

Hậu xử lý lỗi OCR cho báo cáo tài chính tiếng Việt và tiếng Anh bằng Transformer.

## Cấu trúc thư mục

```text
fin-ocrfix/
├── data/
│   ├── sources.csv              # Nguồn PDF và checksum SHA-256
│   ├── dataset.meta.json        # Metadata và cấu hình tạo dữ liệu
│   ├── line_manifest.jsonl      # Truy vết bbox và ảnh dòng
│   ├── vi_dev.jsonl
│   ├── vi_test.jsonl
│   ├── en_dev.jsonl
│   └── en_test.jsonl
├── docs/
│   ├── r1_dataset_notes.md      # Ghi chú pipeline dữ liệu R1
│   ├── r3_coedit_notes.md       # Ghi chú CoEdIT, baseline và phân tích R3
│   └── r4_qwen_notes.md         # Ghi chú Qwen3.5 và prompt R4
├── notebooks/
│   ├── r3_coedit_smoke_test.ipynb
│   ├── r3_run_all.ipynb
│   ├── r4_qwen35_smoke_test.ipynb
│   └── r4_run_all.ipynb
├── prompts/
│   ├── vi_basic.txt
│   ├── vi_strict.txt
│   ├── en_basic.txt
│   └── en_strict.txt
├── references/                  # Đặc tả dự án và tài liệu tham khảo
├── src/
│   ├── build_dataset.py         # Pipeline tạo dữ liệu OCR từ PDF số (R1)
│   ├── run_seq2seq_en.py        # Inference CoEdIT & baseline không sửa (R3)
│   ├── run_llm.py               # Inference Qwen3.5 (R4)
│   └── analyze.py               # Vẽ biểu đồ RQ1–RQ4 & phân tích lỗi (R3)
├── tests/
│   ├── test_build_dataset.py
│   ├── test_run_seq2seq_en.py
│   ├── test_run_llm.py
│   └── test_analyze.py
└── requirements-r1.txt          # Thư viện cho pipeline R1
```
