# R1 · Pipeline và bộ mẫu ngày 1–2

Phạm vi: mốc R1 tuần 1 ngày 1–2 trong Plan4project A4; triển khai theo B1, B2,
C1, C3. Chưa mở rộng thành bộ dữ liệu chính thức của 8–10 doanh nghiệp.

## Nguồn

Hai bản báo cáo thường niên 2024 từ website quan hệ cổ đông Hòa Phát:

- [Bản tiếng Việt](https://file.hoaphat.com.vn/hoaphat-com-vn/2025/04/bao-cao-thuong-nien-hpg-2024.pdf).
- [Bản tiếng Anh](https://file.hoaphat.com.vn/hoaphat-com-vn/2025/04/hpg-annual-report-2024.pdf).

Chỉ dùng trang PDF 115–128, thuộc thuyết minh báo cáo tài chính/chính sách kế toán.
Mỗi file có 156 trang và lớp text đọc được bằng PyMuPDF. `sources.csv` cố định
checksum để phát hiện website thay nội dung tài liệu. `filename` phải nằm trong
`--pdf_dir`; `doc_id` phải duy nhất. `page_start`, `page_end`, `sha256` là các cột
tùy chọn, còn company/year/report_type/lang/filename/url/doc_id là bắt buộc.

Hai PDF BCTC kiểm toán riêng tải thử ban đầu (67 trang mỗi bản) không có lớp text,
đã bị loại khỏi manifest nguồn. Không lấy OCR của PDF scan làm ground truth.

## Quyết định triển khai

1. Lấy dòng và bbox từ `page.get_text("dict")`, nối spans theo thứ tự PDF;
   chuẩn hóa Unicode NFC và gộp khoảng trắng, lọc 5–40 từ.
2. Bỏ mép trên/dưới 6%, trang xoay, dòng không nằm ngang, glyph không hợp lệ,
   trang có ảnh phủ trên 80% diện tích. Bỏ bảng có từ hai cột được
   `find_tables(strategy="lines_strict")` nhận diện; bỏ các đoạn cùng hàng có
   ô rời bên cạnh và span cách nhau quá xa. Bỏ dòng chỉ gồm khoảng trắng trước
   khi xét ô cùng hàng, vì PDF thực có tab thụt đầu dòng thành một text line riêng.
3. Bỏ dòng lặp từ ba lần trong một tài liệu; các dòng khác trùng target chỉ giữ
   một bản, kể cả xuyên tài liệu cùng ngôn ngữ.
4. Render đúng bbox ở 300 DPI, thêm viền trắng 4 px. Light: 150 DPI, Gaussian
   radius 0.3, JPEG quality 60. Heavy: 100 DPI, xoay ngẫu nhiên trong ±1° với
   expand để không cắt chữ, Gaussian radius 0.6, 1% pixel muối tiêu, JPEG quality 30.
   C3 không quy định radius và tỷ lệ muối tiêu; các giá trị này là lựa chọn triển
   khai ban đầu, đã lưu trong metadata. Không điều chỉnh dựa trên metric mô hình.
5. Ảnh sau giải mã JPEG lưu lại dưới PNG để bảo toàn đúng pixel đưa vào OCR.
   Tesseract dùng `vie` hoặc `eng`, `--psm 7`, timeout mặc định 30 giây/ảnh.
6. Lấy mẫu seed 42, VI ưu tiên đúng quota xấp xỉ 20% dòng có số (làm tròn theo
   số dòng gốc), EN lấy mẫu ngẫu nhiên. Không âm thầm giảm quota khi thiếu dữ liệu.
   Chia dev trước rồi test, sau đó mới sinh hai mức nhiễu. Seed nhiễu riêng được
   dẫn xuất từ seed chung, ID dòng và mức nhiễu để không phụ thuộc thứ tự chạy.
7. Ghi vào thư mục tạm, chỉ đưa file đầu ra đến đích sau khi toàn bộ OCR thành
   công. Mặc định từ chối ghi đè. `--overwrite` cần được dùng có chủ ý; không dùng
   cho tập test đã chốt. Khi ghi đè, ảnh cũ không còn được manifest tham chiếu có
   thể vẫn nằm trong `lines/r1`; manifest là danh sách ảnh của lần chạy hiện tại.

## Bộ mẫu

25 dòng gốc (13 VI + 12 EN) × 2 mức nhiễu = **50 cặp OCR/target**.
Dev có 3 dòng gốc mỗi ngôn ngữ; test có 10 VI và 9 EN. Trong 13 dòng VI có
3 dòng chứa số. Các con số này phục vụ thử tích hợp, chưa đạt quota B1 chính thức.

Lọc được 139 ứng viên VI và 115 EN từ hai khoảng trang đã chọn. OCR thật với
Tesseract 5.3.4 tạo 0 source rỗng và 23 source khớp target. Đây là thống kê dữ liệu
đầu vào, không phải kết quả đánh giá mô hình sửa lỗi.

## Kiểm chứng ngày 28/11/2026

- Chạy `python -m unittest discover -s tests -v` trong `venvs/ai`: **24/24 test
  đạt**, gồm 6 test mới của R1 và các test biểu đồ R3. Môi trường sandbox không
  ghi được cache mặc định của matplotlib nên thư viện tự dùng thư mục tạm;
  việc render biểu đồ vẫn thành công.
- Chạy pipeline OCR thật hai lần với seed 42, so sánh byte: bốn JSONL,
  `line_manifest.jsonl`, `dataset.meta.json`, 25 ảnh sạch và 50 ảnh nhiễu đều giống nhau.
- Chạy baseline R3 trên bộ test mẫu và dry-run CoEdIT/Qwen: xuất lần lượt 4, 2,
  4 file predictions, kiểm tra đủ ID, đúng thứ tự và giữ source. Không chạy
  inference CoEdIT/Qwen thật trong tác vụ R1.
- `git diff --check` sạch; PDF/ảnh không xuất hiện trong danh sách file Git mới.

## Giới hạn cần giữ khi dùng dữ liệu

- Chia ở cấp dòng trong cùng doanh nghiệp; không phải phép chia theo doanh nghiệp.
  Dòng VI và EN có thể là bản dịch của nhau. Bộ mẫu dùng thử tích hợp, không đủ
  đại diện để kết luận về chất lượng mô hình hoặc khả năng tổng quát hóa.
- Bộ lọc bảng/header là heuristic bảo thủ; có thể bỏ nhầm dòng hợp lệ hoặc lọt
  một số bố cục đặc biệt. Target là dòng vật lý nên có thể là một phần của câu.
- Chưa làm quy trình 100 dòng, hai người gắn nhãn độc lập của B2; `error_types`
  vẫn rỗng. Đã xem một số ảnh crop để kiểm tra kỹ thuật, không coi đó là QA đầy đủ.
- Tái lập cần cùng PDF, seed, tham số, phiên bản thư viện/Tesseract và dữ liệu
  ngôn ngữ. Đổi phiên bản OCR hoặc traineddata có thể đổi source dù seed giữ nguyên.
- PDF và ảnh được giữ cục bộ và đã bị `.gitignore` loại trừ; chỉ JSONL, metadata,
  manifest nguồn và code được chuẩn bị để chia sẻ. Chưa commit hoặc push.
