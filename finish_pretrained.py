import json
from pathlib import Path

import pandas as pd


output = Path('output/pretrained-vit-20261015')
verification = json.loads((output / 'verification.json').read_text())
summary = json.loads((output / 'summary.json').read_text())
assert verification['val_examples'] == 103500
assert verification['submission_rows'] == 172500
assert verification['submission_differences'] == 0
assert verification['accuracy'] == summary['selected']['accuracy']
scores = pd.read_csv(output / 'scores.csv')
section = '\n## Результат pretrained ViT\n\n| Модель | Верных из 103500 | Accuracy |\n| --- | ---: | ---: |\n'
for row in scores.itertuples():
    section += f'| {row.model} | {row.correct} | {row.accuracy:.8f} |\n'
section += f'\nПорог 0.76102 на val: {summary["val_above_6_threshold"]}; порог 0.77489: {summary["val_above_8_threshold"]}. Это проверка на val, баллы Kaggle пока неизвестны.\n'
section += '\nОтдельный процесс повторил полный val и test: метрика совпала, все 172500 меток submission совпали.\n'
readme = Path('README.md')
readme.write_text(readme.read_text().split('\n## Результат pretrained ViT\n')[0].rstrip() + '\n' + section)
print(json.dumps(verification))
