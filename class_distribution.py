import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from split95 import sha256


def run(output):
    source = Path('output/two-vit-efficientnet-95-20261015')
    labels_path = Path('input/split95-seed45/val_labels.npy')
    labels = np.load(labels_path)
    names = np.load('input/classes.npy')
    predictions = np.load(source / 'val_predictions.npy')
    previous = json.loads((source / 'summary.json').read_text())
    checked = json.loads((source / 'verification.json').read_text())
    assert sha256(labels_path) == previous['sha256'][str(labels_path)]
    assert len(names) == 345 and predictions.shape == labels.shape == (34500,)
    counts = np.bincount(labels, minlength=len(names))
    correct = np.bincount(labels[predictions == labels], minlength=len(names))
    assert (counts == 100).all() and correct.sum() == checked['correct'] == 26808
    assert checked['recipe_prediction_differences'] == 0
    assert correct.sum() / counts.sum() == previous['selected']['accuracy']
    accuracy = correct / counts
    z = 1.959963984540054
    center = (accuracy + z * z / (2 * counts)) / (1 + z * z / counts)
    half = z * np.sqrt(accuracy * (1 - accuracy) / counts + z * z / (4 * counts * counts)) / (1 + z * z / counts)
    table = pd.DataFrame({'class_id': np.arange(len(names)), 'class': names,
                          'count': counts, 'correct': correct, 'accuracy': accuracy,
                          'errors': counts - correct, 'lower95': center - half, 'upper95': center + half})
    table = table.sort_values(['accuracy', 'class']).reset_index(drop=True)
    table['rank'] = np.arange(1, len(table) + 1)
    histogram, edges = np.histogram(correct, bins=np.arange(0, 105, 5))
    assert histogram.sum() == len(names)
    sorted_scores = np.sort(correct)
    gaps = np.diff(sorted_scores)
    central = sorted_scores[17:-17]
    errors = counts - correct
    worst_errors = np.sort(errors)[::-1]
    summary = {'source': str(source), 'val_examples': len(labels), 'classes': len(names),
               'examples_per_class': 100, 'correct': int(correct.sum()), 'accuracy': float(correct.sum() / counts.sum()),
               'quantiles_pct': {str(q): float(np.quantile(correct, q)) for q in (0, .1, .25, .5, .75, .9, 1)},
               'std_pp': float(correct.std()),
               'equal_accuracy_binomial_std_pp': float(100 * np.sqrt(accuracy.mean() * (1 - accuracy.mean()) / 100)),
               'below_50': int((correct < 50).sum()), 'below_60': int((correct < 60).sum()),
               'from_60_to_80': int(((correct >= 60) & (correct < 80)).sum()),
               'at_least_80': int((correct >= 80).sum()), 'at_least_90': int((correct >= 90).sum()),
               'unique_accuracy_values': len(np.unique(correct)),
               'largest_adjacent_gap_pp': int(gaps.max()),
               'largest_gap_bounds_pct': sorted_scores[int(gaps.argmax()):int(gaps.argmax()) + 2].tolist(),
               'central_90pct_largest_gap_pp': int(np.diff(central).max()),
               'total_errors': int(errors.sum()),
               'worst_class_error_share': {str(k): float(worst_errors[:k].sum() / errors.sum()) for k in (10, 20, 50, 69)},
               'histogram': [{'lower': int(a), 'upper': int(b), 'classes': int(n)}
                             for a, b, n in zip(edges[:-1], edges[1:], histogram)],
               'sha256': {str(path): sha256(path) for path in
                          (labels_path, Path('input/classes.npy'), source / 'val_predictions.npy', source / 'summary.json')}}
    output.mkdir(parents=True, exist_ok=False)
    table.to_csv(output / 'classes.csv', index=False)
    (output / 'classes.json').write_text(table.to_json(orient='records', force_ascii=False, double_precision=6))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    (output / 'source.py').write_text(Path(__file__).read_text())
    print(json.dumps(summary), flush=True)
    print(table.head(10).to_string(index=False))
    print(table.tail(10).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args().output)
