# Báo cáo Day 09 — Trần Tuấn Anh, 2A202603230

> Cấu hình nộp: `duplicates.phash_dist = 16`, `second_opinion.conf = 0.6` (các mục khác giữ nguyên). Số liệu lấy từ
> `submission/reports/` và `submission/calibration/`. Các thí nghiệm phụ chạy vào `work/`, không nộp.

## 1. Kết luận trước
**HOLD** (mã thoát 10). Quét **224** ảnh, loại **31** ở cửa vào, **193** ảnh vào QC; sau sửa tự động và người duyệt (80 ảnh)
**không còn lỗi bắt buộc**, nhưng **34/193 ảnh (17,62%)** còn cảnh báo chưa được người xem ≥ ngưỡng 10%: hàng đợi có **121** ảnh,
ngân sách chỉ 80. Dataset `submission/export/clean_v1` (193 ảnh) chỉ là **bản nháp để thử nghiệm**, chưa được phát hành.

## 2. Lỗi và phân việc
Từ `02_error_list_v1` (trước sửa). Lỗi cửa vào tính trên 224 ảnh quét; lỗi nhãn/cảnh báo tính trên 193 ảnh vào QC. Số là **số issue**.

| Mã lỗi | Số ảnh/dòng | Ai sửa (máy/người) | Vì sao |
|---|---:|---|---|
| LEAK_TEST | 11 / 224 | máy (loại ảnh) | ảnh trùng kho test/val: giữ lại thì điểm test bị thổi phồng |
| DUP_IMAGE_EXACT | 8 / 224 | máy (loại ảnh) | md5 trùng tuyệt đối, chắc chắn |
| DUP_IMAGE_NEAR | 7 / 224 | máy (loại ảnh) | phash ≤ 16; ở mép ngưỡng có xoá nhầm (xem §3, §4) |
| BAD_SIZE / CORRUPT_IMAGE | 3 / 2 (trên 224) | máy (loại ảnh) | quy tắc cứng: cạnh < 250 px / không `load()` được |
| BOX_OUT_OF_IMAGE | 8 | máy (`clip_box` ×8) | cắt về [0,1], giữ lớp — sửa cú pháp, không đổi nghĩa |
| DUP_BOX · BAD_CLASS_ID · BOX_ZERO · BAD_FORMAT | 7 · 6 · 4 · 3 | máy (`drop_line` ×20) | dòng hỏng/lặp không cứu được; xoá và ghi `03_fixes.csv` (trước → sau) |
| BAD_CLASS_NAME | 2 | máy (`rename_classes` ×1) | ghi lại `names` đúng danh sách dự án |
| MISSING_LABEL_FILE | 6 | **người** (`needs_label` ×6) | máy không bịa nhãn, không tạo file rỗng (rỗng = "ảnh nền") |
| POSSIBLE_MISSING | 53 | **người** | ý nghĩa: model lớn thấy vật mà nhãn không có — model cũng có thể sai |
| POSSIBLE_FALSE_POS | 28 | **người** | box nhãn không khớp vật nào model thấy |
| POSSIBLE_LOOSE_BOX | 27 | **người** | đúng vật, box lỏng/lệch |
| POSSIBLE_WRONG_CLASS | 17 | **người** | nhầm lớp hợp lệ (vd. truck ↔ bus): quy tắc cú pháp không bắt được |
| EMPTY_LABEL · BOX_TINY · CLASS_RARE · BOX_HUGE | 11 · 6 · 4 · 3 | **người** | ảnh nền hay thiếu nhãn? box nhỏ/to bất thường: cần mắt người |

Tổng: 36 lỗi bắt buộc + 149 cảnh báo (vòng 1); máy tạo **35 thay đổi** (`03_fixes.csv`); sau sửa còn 6 lỗi bắt buộc (đều
MISSING_LABEL_FILE) → chuyển người.

**Hàng đợi `05_review_queue.csv`** xếp theo điểm ưu tiên = tổng trọng số mã lỗi của ảnh (`review.weights`: NEEDS_LABEL 10,
WRONG_CLASS 3, MISSING/FALSE_POS/FIXED_DROP_LINE 2, LOOSE 1,5…). Đầu hàng là img_0020 (11,5: dòng bị xoá + MISSING + WRONG_CLASS)
và **6 ảnh chưa có nhãn** (10,0) — để chúng luôn nằm trong ngân sách vì nếu không gán thì cả lô Rework. Với ngân sách 80/121,
người xem được 80 ảnh ưu tiên cao nhất; 41 ảnh còn lại (34 ảnh còn cảnh báo) không được xem → Hold.
Người xem được **94/120 cờ ngữ nghĩa** (POSSIBLE_*, theo `04_error_list_v2`) = **78%**, dù chỉ duyệt 80/121 ảnh (66%) — nhờ hàng đợi
dồn ảnh nhiều cờ lên trước. 26 cờ còn lại nằm trên ảnh chưa duyệt.

3 lỗi tự nhìn thấy ở Bước 1 (`viz_labels.py`): img_0001 (box người lỏng, thiếu box người lính), img_0030 (thiếu 3 người).
Cả hai ảnh vào hàng đợi (hạng 26 và 21) và **đã được duyệt** (`06_review_log`: "thay bằng nhãn người").

## 3. Calibration (2 ngưỡng)
| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do (đánh đổi) |
|---|---|---|---|---|
| `duplicates.phash_dist` | 2 | **16** | NEAR: 1,00 / 0,50 → 1,00 / 0,83 (EXACT 1,00 / 1,00) | Ưu tiên **recall** trùng/rò rỉ: ảnh test lọt vào train làm sai chính phép đo, hại hơn mất vài ảnh. Trên raw, so với ≤ 15, 16 bắt thêm 1 cặp trùng thật + 6 ứng viên rò rỉ test; giá: xoá nhầm 2 ảnh (~1%). Không lên 17+: trên raw 17 = 16, calib sụp ở 20 (EXACT recall 0,33). |
| `second_opinion.conf` | 0,25 | **0,6** | WRONG_CLASS 0,56/0,83 → 0,62/0,83 · MISSING 0,50/1,00 → 0,57/1,00 · LOOSE 0,29/0,40 → 0,20/0,20 | Giá trị cao nhất giữ recall MISSING = 1,00 (thiếu nhãn là lỗi chính tôi tự thấy); precision tăng. 0,8 cho WRONG_CLASS P = 1,00 nhưng MISSING recall 0,75. Chấp nhận LOOSE recall giảm. |

**Calibration có thể đánh lừa — bằng chứng:**
1. Calib (39 ảnh) cho NEAR precision 1,00 ở 16. Trên raw, quét 10–17 (`work/`): 10–15 cho cùng kết quả, 16 = 17; mọi khác biệt
   nằm đúng ở dist 16. Ba cặp raw↔raw ở dist 16, tôi mở xem: chỉ img_0084↔img_0074 trùng thật; img_0011↔img_0008 và
   img_0177↔img_0113 là ảnh khác → precision thật ở mép ngưỡng **1/3**. Calib quá nhỏ, thiếu cặp "khác nhưng giống" (cảnh phố COCO).
2. POSSIBLE_FALSE_POS 0,31 / 1,00 ở **mọi** `conf` (dùng `fp_conf` cố định) → calibrate `conf` không giảm nguồn nhiễu lớn nhất.
3. Thử `conf` = 0,8 trên raw: cổng ra **Release** (13/193 = 6,74%) — nhưng vì MISSING giảm 40% (53 → 32), không phải vì dữ liệu
   sạch hơn; img_0030 (thiếu 3 người) mất cờ. Ngưỡng "hợp với cổng" ≠ ngưỡng đúng → giữ 0,6.

## 4. Cửa chặn và điều pipeline KHÔNG bắt được
**(a) Cổng chặn thật:** `python -m pipeline.run --budget 0 --reports work/rep_nb --export-dir work/exp_nb` → kết luận
**REWORK** ("Còn 6 lỗi bắt buộc (MISSING_LABEL_FILE=6)"), **mã thoát 20**, log "KHÔNG xuất: còn lỗi bắt buộc"; `Test-Path
work/exp_nb/clean_v1` = **False**. Không người gán nhãn → không có dataset; lỗi bắt buộc không bao giờ bị bỏ qua kể cả khi ảnh đã duyệt.

**(b) Điểm mù (có bằng chứng):**
1. **Box lỏng nhẹ** — img_0001: box người chưa chạm chân (tự thấy) nhưng **không** có POSSIBLE_LOOSE_BOX: IoU với box model vẫn
   ≥ `iou_loose` 0,5. Calib: LOOSE recall chỉ 0,20.
2. **Thiếu nhãn khi model không đủ tự tin** — img_0030 thiếu 3 người, máy chỉ cờ 1 (POSSIBLE_MISSING, conf 0,686); 2 người còn lại
   dưới conf 0,6 hoặc < 600 px² → chỉ người bắt được.
3. **phash ở mép ngưỡng xoá nhầm** — img_0011, img_0177 bị loại là "trùng" dù là ảnh khác; máy tự xoá, không qua người.
4. **Rò rỉ test chưa xác minh** — 6 ảnh (img_0013, 0017, 0021, 0033, 0038, 0115) bị loại vì khớp ảnh test ở dist 16; theo tỉ lệ 1/3
   ở trên, có thể một phần bị xoá nhầm. Tôi không mở ảnh test để kiểm (RULES). Hướng sửa: ca ở vùng mép ngưỡng nên chuyển người.

## 5. Train: raw vs clean (tùy chọn, chấm riêng)
**Cấu hình (rút gọn, máy không có GPU NVIDIA):** `python scripts/train_compare.py --variants raw clean --seeds 0 1 --epochs 15
--imgsz 416 --device cpu` — yolo26n.pt, batch 16, **15 epoch, imgsz 416, CPU** (đề gợi ý cho máy không GPU; chuẩn là 20 epoch, 640).
Công bằng: cùng model khởi tạo, siêu tham số, seed {0, 1}, val `gold_val` (40 ảnh), test (100 ảnh); chỉ khác dữ liệu train.

| biến thể | #ảnh train | số seed | mAP50 | mAP50-95 | precision | recall |
|---|---|---|---|---|---|---|
| raw | 224 | 2 | 0.621 ± 0.008 | 0.418 ± 0.020 | 0.640 ± 0.003 | 0.585 ± 0.009 |
| clean | 193 | 2 | 0.644 ± 0.027 | 0.428 ± 0.027 | 0.735 ± 0.036 | 0.517 ± 0.006 |

*(từ `submission/results/comparison.md`, TB ± độ lệch giữa 2 seed)*. AP50 theo lớp (TB 2 seed, raw → clean): car 0,681 → 0,649 ·
person 0,744 → 0,741 · truck 0,534 → 0,580 · bus 0,525 → 0,605.

**So với nhiễu:**
- **mAP50 +0,023**, **mAP50-95 +0,010**: **nhỏ hơn độ lệch giữa seed** của clean (0,027) → **chưa kết luận được** clean tốt hơn.
  Hai khoảng chồng nhau: clean seed 1 (0,625) ≈ raw seed 0 (0,626).
- **Precision +0,095, recall −0,068**: cả hai lớn hơn nhiều lần độ lệch seed (≤ 0,036) → khác biệt có vẻ thật, nhưng là
  **đánh đổi** (model clean "dè dặt" hơn), không phải cải thiện toàn diện.
- Theo lớp, bus (+0,08) và truck (+0,05) tăng — khớp với loại lỗi pipeline sửa/cờ (BAD_CLASS_ID, WRONG_CLASS truck↔bus, CLASS_RARE);
  car giảm (−0,03). Với 2 seed và 100 ảnh test, AP từng lớp rất nhiễu (bus raw dao động 0,49–0,56) → chỉ là gợi ý.

**Nguyên nhân có thể và cách kiểm:**
1. **Rò rỉ test trong raw**: raw còn **11 ảnh** pipeline gắn LEAK_TEST (trùng ảnh test) → model raw đã "thấy" một phần test, điểm raw
   (nhất là recall) có thể bị thổi phồng. *Kiểm:* đánh giá lại trên test bỏ các ảnh test bị khớp, hoặc train raw chỉ bỏ 11 ảnh rò rỉ.
2. **Clean ít dữ liệu hơn** (193 vs 224, −14% ảnh) → ít box hơn, recall giảm. *Kiểm:* biến thể "raw bỏ đúng 31 ảnh nhưng giữ nhãn gốc"
   để tách tác động "bớt ảnh" khỏi tác động "sửa nhãn".
3. **Clean vẫn còn lỗi nhãn** (Hold: 34/193 ảnh còn cảnh báo chưa duyệt; 53 cờ POSSIBLE_MISSING ở vòng 1) → thiếu box dạy model bỏ
   sót vật → recall thấp. *Kiểm:* chạy biến thể `oracle` (nhãn người, chặn trên) bằng `--variants oracle`.
4. **Nhiễu thí nghiệm**: chỉ 2 seed, test 100 ảnh, cấu hình rút gọn trên CPU.

**Kết luận trung thực:** với dữ liệu hiện có, **không thể nói clean tốt hơn raw** về mAP — chênh lệch nằm trong nhiễu giữa seed.
Thứ đo được rõ là clean cho precision cao hơn và recall thấp hơn. **Việc tiếp theo:** ≥ 3–5 seed với cấu hình chuẩn (20 epoch, 640,
GPU Colab/Kaggle); thêm biến thể `oracle`; đo lại raw khi bỏ ảnh rò rỉ test; duyệt nốt hàng đợi (≥ 15 ảnh) rồi train lại clean.

## 6. Quyết định và ghi chú AI
**(a) Quyết định: HOLD.** Điều kiện để thành Release (thí nghiệm trong `work/`, cấu hình nộp không đổi):
1. **Duyệt thêm ≥ 15 ảnh** theo thứ tự hàng đợi: ngân sách 95 → 19/193 = 9,84% < 10% → Release (90 → 24/193, vẫn Hold).
   Đây là cách đúng: cảnh báo giảm vì **người đã xem**.
2. Nâng `conf` lên 0,8 cũng Release (13/193) nhưng **không chọn**: cảnh báo giảm vì máy im lặng (recall MISSING calib 1,00 → 0,75).
   Chỉ hợp lệ nếu guideline chấp nhận sót ~25% thiếu nhãn — quyết định của trưởng dự án.
3. Lâu dài: giảm nhiễu POSSIBLE_FALSE_POS (28 cờ, precision 0,31) để rút ngắn hàng đợi; đưa ca phash mép ngưỡng sang người.

**(b) Dùng AI (Claude):**
- *Prompt chính:* đưa docstring + test của từng module (đầu vào, đầu ra, ràng buộc như số dòng gốc, không sửa `data/raw`, ngưỡng
  đọc từ `cfg`, bẫy dòng 2 lỗi/dòng trống) và yêu cầu code + lệnh để tôi tự chạy.
- *Cách kiểm:* đọc từng dòng; chạy `pytest` từng file rồi toàn bộ (79 xanh); chạy pipeline, đối chiếu với 3 lỗi tôi tự thấy và
  mở ảnh các cặp trùng để kiểm calibration.
- *AI sai:* (1) AI khẳng định `phash_dist` 10–15 cho cùng kết quả trên raw — sai, vì quên ngưỡng này cũng dùng cho LEAK_TEST
  (16 bắt thêm 6 ảnh rò rỉ); (2) AI từng gợi ý mở ảnh `data/test` để quyết ngưỡng — trái RULES, tôi không dùng; (3) vòng lặp
  PowerShell đầu tiên làm hỏng UTF-8 của file config tạm (`#x0090`), phải sửa cách đọc/ghi.
- *Kiểm bằng `check_submission`:* lần đầu báo `manifest_sha1` không khớp — trên Windows `manifest.json` có xuống dòng CRLF,
  `train_compare.py` băm byte thô còn `check_submission` chuẩn hoá về LF. Tôi lấy lại manifest bản LF từ git (nội dung không đổi) và
  train lại `clean` cùng cấu hình; không sửa tay file kết quả.

*Ghi chú:* trên Windows, `check_submission.py` còn báo 2 lỗi ở `pipeline\calibrate.py` (file đề, không sửa — sha1 khớp bản gốc): đường
dẫn `\` không khớp `pipeline/calibrate.py` trong `scripts/_protected.json` nên file đề bị quét như code học viên; trên Linux không gặp.