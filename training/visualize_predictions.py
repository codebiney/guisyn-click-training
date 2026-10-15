"""Render saved grounding predictions with commands and predicted click points."""
import argparse
import csv
import html
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.predictions.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    split = data.get('split', 'validation')
    records, cards = [], []
    for index, row in enumerate(data['predictions'], 1):
        original = Image.open(row['image']).convert('RGB')
        prediction = row['finetuned']
        width = max(760, original.width)
        command = 'Command: ' + row['instruction']
        lines, current = [], ''
        for word in command.split():
            candidate = (current + ' ' + word).strip()
            if font.getlength(candidate) > width - 40 and current:
                lines.append(current)
                current = word
            else:
                current = candidate
        lines.append(current)
        xy = prediction.get('predicted_pixels')
        normalized = prediction.get('predicted_normalized')
        hit = prediction['bbox_hit']
        status = 'HIT' if hit else 'MISS'
        lines += [f'Prediction (0..1000): {normalized} | {status}',
                  'Prediction (pixels): ' + (f'({xy[0]:.2f}, {xy[1]:.2f})' if xy else 'invalid'),
                  'Red cross = model prediction; green box = ground truth',
                  f'Saved {split} inference | step {data["summary"]["step"]} | case {index:02d}']
        header = len(lines) * 27 + 24
        canvas = Image.new('RGB', (width, original.height + header + 16), '#111827')
        offset_x = (width - original.width) // 2
        canvas.paste(original, (offset_x, header))
        draw = ImageDraw.Draw(canvas)
        for line_index, line in enumerate(lines):
            draw.text((20, 12 + line_index * 27), line, font=font, fill='white')
        x0, y0, x1, y1 = row['bbox_pixels']
        draw.rectangle((x0 + offset_x, y0 + header, x1 + offset_x, y1 + header), outline='#00b050', width=2)
        distance = None
        if xy:
            px, py = xy
            x, y = px + offset_x, py + header
            for color, thickness in [('white', 5), ('#ef233c', 3)]:
                draw.ellipse((x-9, y-9, x+9, y+9), outline=color, width=thickness)
                draw.line((x-15, y, x+15, y), fill=color, width=thickness)
                draw.line((x, y-15, x, y+15), fill=color, width=thickness)
            distance = math.hypot(max(x0-px, 0, px-x1), max(y0-py, 0, py-y1))
        filename = f'{index:02d}_{status.lower()}_{Path(row["image"]).stem}.png'
        canvas.save(args.output / filename)
        record = {'case': index, 'image': filename, 'source_image': row['image'],
                  'instruction': row['instruction'], 'target_name': row['target_name'],
                  'model_response': prediction['response'], 'predicted_normalized': normalized,
                  'predicted_pixels': xy, 'target_normalized': row['target'],
                  'bbox_pixels': row['bbox_pixels'], 'bbox_hit': hit,
                  'distance_outside_bbox_pixels': distance}
        records.append(record)
        cards.append(f'<article><a href="{filename}"><img loading="lazy" src="{filename}"></a>'
                     f'<p>{index:02d} · {status} · {html.escape(row["target_name"])}</p>'
                     f'<p>{html.escape(row["instruction"])}</p></article>')
    (args.output / 'predictions.json').write_text(json.dumps({
        'source_predictions': str(args.predictions.resolve()), 'summary': data['summary'],
        'note': 'Visualization of saved model inference; not a new inference run.',
        'predictions': records}, ensure_ascii=False, indent=2))
    with (args.output / 'predictions.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (args.output / 'index.html').write_text('<!doctype html><meta charset="utf-8">'
        '<title>Model prediction gallery</title><style>body{background:#111827;color:#fff;'
        'font-family:sans-serif;margin:24px}.grid{display:grid;grid-template-columns:repeat('
        'auto-fit,minmax(340px,1fr));gap:24px}img{width:100%}article{background:#1f2937;'
        f'padding:12px}}p{{line-height:1.5}}</style><h1>Model {html.escape(split)} predictions</h1>'
        f'<p>Step {data["summary"]["step"]} · {len(records)} cases · '
        f'{sum(r["bbox_hit"] for r in records)} hits. Red: prediction; green: ground-truth box.</p>'
        '<div class="grid">' + ''.join(cards) + '</div>')
    print(f'Saved {len(records)} annotated images and JSON/CSV/HTML to {args.output}')


if __name__ == '__main__':
    main()
