"""CHECK 2 — Dòng nhãn YOLO: sai định dạng, ID lớp lạ, box tràn ảnh, box rỗng, box lặp.   (Script 2 trong slide)   ★★

Test:  pytest tests/test_label_format.py -q

Mỗi dòng nhãn: `id cx cy w h` (5 số, toạ độ 0–1). Trả về list[Issue]; `Issue.line` = số dòng (đếm từ 1).

| code             | severity | khi nào                                                                  |
|------------------|----------|--------------------------------------------------------------------------|
| BAD_FORMAT       | ERROR    | không đúng 5 cột số (dùng `parse_yolo_line`, bắt `ValueError`)           |
| BAD_CLASS_ID     | ERROR    | id lớp ngoài 0..len(ds.classes)-1                                        |
| BOX_ZERO         | ERROR    | w <= 0 hoặc h <= 0                                                       |
| BOX_OUT_OF_IMAGE | ERROR    | mép box vượt [0,1] quá `cfg["label_format"]["overflow_tol"]`; extra={"overflow": mức tràn} |
| DUP_BOX          | ERROR    | cùng lớp với một dòng TRƯỚC ĐÓ và IoU > `dup_box_iou`; báo ở dòng SAU    |
| BOX_TINY         | WARN     | cạnh ngắn của box (px) < `min_box_px`  (cần `ds.image_size(stem)`)       |
| BOX_HUGE         | WARN     | w*h > `max_box_frac`                                                     |

Gợi ý
- Mép trái = x - w/2, mép phải = x + w/2 (xem câu hỏi "dòng YOLO nào lỗi?" trong slide).
- Một dòng hỏng (BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO) thì `continue`, đừng kiểm tiếp hình học của dòng đó.
- Dòng thiếu cột làm script của slide dừng giữa chừng: đây là lý do phải `try/except`.
- `iou_xywh(a, b)` có sẵn trong common.py.
- Ngưỡng nằm trong `cfg["label_format"]` — đừng viết cứng số vào code.
"""
from ..common import ERROR, WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401


def run(ds, cfg):
    lf = cfg["label_format"]
    tol, min_px = lf["overflow_tol"], lf["min_box_px"]
    max_frac, dup_iou = lf["max_box_frac"], lf["dup_box_iou"]
    n_cls = len(ds.classes)
    issues = []

    for lp in ds.label_files():
        stem = lp.stem
        size = ds.image_size(stem)            # (W, H) hoặc None
        seen = []                             # [(lớp, box, số dòng)] các dòng hợp lệ phía trước

        for no, text in ds.read_label_lines(stem) or []:
            def add(sev, code, msg, **extra):
                issues.append(Issue("label_format", sev, code, stem, msg, no, extra))

            # 1) đúng 5 cột số?
            try:
                c, x, y, w, h = parse_yolo_line(text)
            except ValueError as e:
                add(ERROR, "BAD_FORMAT", f"dòng sai định dạng: {e}")
                continue
            # 2) id lớp hợp lệ?
            if not 0 <= c < n_cls:
                add(ERROR, "BAD_CLASS_ID", f"id lớp {c} ngoài 0..{n_cls - 1}", id=c)
                continue
            # 3) box có diện tích?
            if w <= 0 or h <= 0:
                add(ERROR, "BOX_ZERO", f"box rỗng w={w}, h={h}")
                continue
            # 4) tràn ảnh (so với dung sai)
            overflow = max(0.0, -(x - w / 2), (x + w / 2) - 1.0, -(y - h / 2), (y + h / 2) - 1.0)
            if overflow > tol:
                add(ERROR, "BOX_OUT_OF_IMAGE", f"box tràn ảnh {overflow:.4f} > {tol}", overflow=round(overflow, 6))
            # 5) box lặp: cùng lớp với một dòng TRƯỚC ĐÓ, IoU > ngưỡng -> báo ở dòng sau
            box = (x, y, w, h)
            for c0, b0, no0 in seen:
                if c0 == c:
                    v = iou_xywh(box, b0)
                    if v > dup_iou:
                        add(ERROR, "DUP_BOX", f"trùng box dòng {no0} (IoU={v:.3f})", iou=round(v, 4), dup_of=no0)
                        break
            seen.append((c, box, no))
            # 6) cảnh báo kích thước
            if size:
                W, H = size
                short = min(w * W, h * H)
                if short < min_px:
                    add(WARN, "BOX_TINY", f"cạnh ngắn {short:.1f}px < {min_px}px", short_px=round(short, 2))
            if w * h > max_frac:
                add(WARN, "BOX_HUGE", f"box chiếm {w * h:.2%} ảnh > {max_frac:.0%}", frac=round(w * h, 4))
    return issues

