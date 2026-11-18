"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict  # noqa: F401

import yaml  # noqa: F401

from ..common import ERROR, parse_yolo_line  # noqa: F401

DROP_CODES = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
REMOVE_CODES = {"LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}


def _is_remove(code):
    return code in REMOVE_CODES or code.startswith("DUP_IMAGE_")


def _rec(image, line, action, before, after):
    return {"image": image, "line": line, "action": action, "before": before, "after": after}


def fix(ds, issues, cfg):
    errs = [i for i in issues if i.severity == ERROR]      # KHÔNG động tới WARN
    fixes = []

    # 1) Xoá ảnh + nhãn (trùng, rò rỉ test, hỏng, quá nhỏ)
    removed = set()
    for i in errs:
        if not _is_remove(i.code) or i.image in removed:
            continue
        img, lab = ds.image_path(i.image), ds.label_path(i.image)
        if img is None and not lab.exists():                 # đã xoá từ lần trước -> idempotent
            continue
        before = img.name if img else lab.name
        if img is not None:
            img.unlink()
        if lab.exists():
            lab.unlink()
        removed.add(i.image)
        fixes.append(_rec(i.image, None, "remove_image", before, f"(đã xoá: {i.code})"))

    # 2) Sửa tên lớp trong data.yaml
    if any(i.code == "BAD_CLASS_NAME" for i in errs):
        want = {k: n for k, n in enumerate(cfg["classes"])}
        if ds.yaml_names() != want:
            data = {}
            if ds.yaml_path.exists():
                data = yaml.safe_load(ds.yaml_path.read_text(encoding="utf-8")) or {}
            before = data.get("names")
            data["names"] = want
            with open(ds.yaml_path, "w", encoding="utf-8", newline="\n") as f:
                yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
            fixes.append(_rec("data.yaml", None, "rename_classes", str(before), str(cfg["classes"])))

    # 3) Thiếu file nhãn: KHÔNG bịa nhãn, KHÔNG tạo file rỗng -> chuyển người gán
    flagged = set()
    for i in errs:
        if i.code == "MISSING_LABEL_FILE" and i.image not in removed and i.image not in flagged:
            if not ds.label_path(i.image).exists():
                flagged.add(i.image)
                fixes.append(_rec(i.image, None, "needs_label", "(không có file)", "chuyển người gán"))

    # 4) Lỗi theo dòng: gom theo file, dùng số dòng GỐC, ghi lại MỘT lần
    by_file = defaultdict(lambda: defaultdict(set))          # stem -> {số dòng: {code}}
    for i in errs:
        if i.line is None or i.image in removed:
            continue
        if i.code in DROP_CODES or i.code == "BOX_OUT_OF_IMAGE":
            by_file[i.image][int(i.line)].add(i.code)

    for stem in sorted(by_file):
        p = ds.label_path(stem)
        if not p.exists():                                   # ảnh/nhãn đã bị xoá -> bỏ qua êm
            continue
        raw = p.read_text(encoding="utf-8", errors="replace").splitlines()
        out, changed = [], False
        for no, text in enumerate(raw, 1):                   # đếm cả dòng trống -> khớp số dòng gốc
            codes = by_file[stem].get(no)
            if not codes:
                out.append(text)
                continue
            if codes & DROP_CODES:                           # xoá dòng thắng, không clip thêm
                fixes.append(_rec(stem, no, "drop_line", text, "(xoá dòng)"))
                changed = True
                continue
            # chỉ còn BOX_OUT_OF_IMAGE -> cắt về [0,1]
            try:
                c, x, y, w, h = parse_yolo_line(text)
            except ValueError:
                out.append(text)                             # không parse được: để check khác báo
                continue
            x1, y1 = max(0.0, x - w / 2), max(0.0, y - h / 2)
            x2, y2 = min(1.0, x + w / 2), min(1.0, y + h / 2)
            if x2 <= x1 or y2 <= y1:                         # nằm hẳn ngoài ảnh -> xoá, không ghi box rỗng
                fixes.append(_rec(stem, no, "drop_line", text, "(box ngoài ảnh, xoá dòng)"))
                changed = True
                continue
            new = f"{c} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"
            out.append(new)
            if new != text:
                fixes.append(_rec(stem, no, "clip_box", text, new))
                changed = True
        if changed:
            with open(p, "w", encoding="utf-8", newline="\n") as f:   # chỉ sửa trong ds, không phải data/raw
                f.write("".join(l + "\n" for l in out))
    return fixes

