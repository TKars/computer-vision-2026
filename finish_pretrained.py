import json
import argparse
import re
from pathlib import Path

import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--output', default='output/pretrained-vit-20261015')
output = Path(parser.parse_args().output)
verification = json.loads((output / 'verification.json').read_text())
summary = json.loads((output / 'summary.json').read_text())
assert verification['val_examples'] == 103500
assert verification['submission_rows'] == 172500
assert verification['submission_differences'] == 0
assert verification['accuracy'] == summary['selected']['accuracy']
scores = pd.read_csv(output / 'scores.csv')
config = json.loads((output / 'config.json').read_text())
name = 'EfficientNet-B3' if config['model'].startswith('efficientnet') else 'pretrained ViT'
heading = f'\n## Результат {name}\n'
section = heading + '\n| Модель | Верных из 103500 | Accuracy |\n| --- | ---: | ---: |\n'
for row in scores.itertuples():
    section += f'| {row.model} | {row.correct} | {row.accuracy:.8f} |\n'
section += f'\nПорог 0.76102 на val: {summary["val_above_6_threshold"]}; порог 0.77489: {summary["val_above_8_threshold"]}. Это проверка на val, баллы Kaggle пока неизвестны.\n'
section += '\nОтдельный процесс повторил полный val и test: метрика совпала, все 172500 меток submission совпали.\n'
readme = Path('README.md')
text = readme.read_text()
pattern = re.escape(heading) + r'.*?(?=\n## |\Z)'
readme.write_text(re.sub(pattern, lambda _: section, text, count=1, flags=re.S)
                  if re.search(pattern, text) else text.rstrip() + '\n' + section)
print(json.dumps(verification))
