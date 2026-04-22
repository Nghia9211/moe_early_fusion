import json

path = 'review.json'
errors = []

with open(path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        stripped = line.strip()
        if not stripped:
            errors.append((i, 'EMPTY LINE', ''))
            continue
        try:
            json.loads(stripped)
        except json.JSONDecodeError as e:
            errors.append((i, str(e), stripped[:100]))

print(f"Total errors: {len(errors)}")
for line_no, err, preview in errors[:20]:  # chỉ show 20 lỗi đầu
    print(f"  Line {line_no:>8}: {err}")
    if preview:
        print(f"             Preview: {preview}")