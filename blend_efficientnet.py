import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from blend_val import temper


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(parent, efficientnet, output):
    parent, efficientnet, output = Path(parent), Path(efficientnet), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    previous = json.loads((parent / 'summary.json').read_text())
    scores = json.loads((efficientnet / 'summary.json').read_text())
    checked = json.loads((efficientnet / 'verification.json').read_text())
    assert checked['val_examples'] == 103500 and checked['submission_differences'] == 0
    assert checked['accuracy'] == scores['selected']['accuracy']
    labels = np.load('input/val_labels.npy')
    assert len(labels) == 103500
    groups = {name: previous['groups'][name] for name in ('vit', 'resnets', 'scratch_vit')}
    groups['efficientnet'] = [str(efficientnet / 'val_flip.npy')]
    raw = {p: np.load(p, mmap_mode='r') for paths in groups.values() for p in paths}
    assert all(p.shape == (103500, 345) for p in raw.values())
    hashes = {p: digest(p) for p in raw}
    assert all(hashes[p] == sha for p, sha in previous['probability_sha256'].items())
    transformed = {}

    def group(name, temperature):
        key = name, temperature
        if key not in transformed:
            transformed[key] = sum(temper(raw[p], temperature) for p in groups[name]) / len(groups[name])
        return transformed[key]

    def predict(recipe):
        return sum(item['weight'] * group(item['group'], item['temperature']) for item in recipe).argmax(1)

    rows = []

    def score(family, recipe):
        correct = int((predict(recipe) == labels).sum())
        row = {'family': family, 'correct': correct, 'accuracy': correct / len(labels), 'recipe': recipe}
        rows.append(row)
        return row

    baseline = score('previous', previous['recipe'])
    assert baseline['correct'] == previous['selected']['correct']
    single = score('efficientnet', [{'group': 'efficientnet', 'weight': 1.0, 'temperature': 1}])
    assert single['correct'] == next(row['correct'] for row in pd.read_csv(efficientnet / 'scores.csv').to_dict('records')
                                     if row['model'] == 'efficientnet_b3_flip')
    bases = {'ensemble': previous['recipe'], 'vit_resnets': json.loads(previous['best_pair']['recipe']),
             'resnets': [{'group': 'resnets', 'weight': 1.0, 'temperature': 4}]}
    bases.update({f'vit_t{t}': [{'group': 'vit', 'weight': 1.0, 'temperature': t}] for t in (1, 2, 4)})
    for name, base in bases.items():
        def candidate(weight, temperature):
            return [{**item, 'weight': item['weight'] * (1 - weight)} for item in base] + [
                {'group': 'efficientnet', 'weight': weight, 'temperature': temperature}]

        for temperature in (1, 2, 4):
            for weight in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
                score(name, candidate(weight, temperature))
        best = max((r for r in rows if r['family'] == name), key=lambda r: r['correct'])
        weight = best['recipe'][-1]['weight']
        temperature = best['recipe'][-1]['temperature']
        for w in np.arange(max(0, weight - .1), min(1, weight + .1) + .001, .025):
            score(name, candidate(float(w), temperature))
        print(json.dumps(max((r for r in rows if r['family'] == name), key=lambda r: r['correct'])), flush=True)
    center = max((r for r in rows if r['family'] == 'vit_resnets'), key=lambda r: r['correct'])['recipe']
    for dv in (-.05, -.025, 0, .025, .05):
        for de in (-.05, -.025, 0, .025, .05):
            wv, we = center[0]['weight'] + dv, center[2]['weight'] + de
            wr = 1 - wv - we
            if min(wv, wr, we) >= 0:
                score('vit_resnets_refined', [{**item, 'weight': weight}
                                             for item, weight in zip(center, (wv, wr, we))])
    selected = max(rows, key=lambda r: r['correct'])
    np.save(output / 'val_predictions.npy', predict(selected['recipe']))
    table = pd.DataFrame([{**row, 'recipe': json.dumps(row['recipe'])} for row in rows])
    table.sort_values('correct', ascending=False).to_csv(output / 'scores.csv', index=False)
    summary = {'selected': selected, 'recipe': selected['recipe'], 'groups': groups,
               'parent': str(parent), 'efficientnet': str(efficientnet),
               'efficientnet_snapshot': json.loads((efficientnet / 'best.json').read_text()),
               'efficientnet_weights_sha256': digest(efficientnet / 'model.pth'),
               'baselines': [baseline, single], 'val_examples': len(labels), 'candidates': len(rows),
               'above_6_threshold': selected['accuracy'] > .76102,
               'above_8_threshold': selected['accuracy'] > .77489,
               'above_10_threshold': selected['accuracy'] > .78013,
               'probability_sha256': hashes, 'labels_sha256': digest('input/val_labels.npy'),
               'flip_tta': True, 'validation_only': True}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    (output / 'source.py').write_text(Path(__file__).read_text())
    print(json.dumps(selected), flush=True)


def verify(output):
    output = Path(output)
    summary = json.loads((output / 'summary.json').read_text())
    for path, sha in summary['probability_sha256'].items():
        assert digest(path) == sha
    assert digest('input/val_labels.npy') == summary['labels_sha256']
    probabilities = 0
    for item in summary['recipe']:
        arrays = [np.load(path, mmap_mode='r') for path in summary['groups'][item['group']]]
        probs = sum(temper(p, item['temperature']) for p in arrays) / len(arrays)
        probabilities = probabilities + item['weight'] * probs
    predictions = probabilities.argmax(1)
    differences = int((predictions != np.load(output / 'val_predictions.npy')).sum())
    correct = int((predictions == np.load('input/val_labels.npy')).sum())
    assert differences == 0 and correct == summary['selected']['correct']
    result = {'val_examples': len(predictions), 'correct': correct, 'accuracy': correct / len(predictions),
              'recipe_prediction_differences': differences, 'serialized_recipe_reloaded': True,
              'probability_hashes_verified': True, 'validation_only': True}
    (output / 'verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', default='output/pretrained-blend87-20261015/results')
    parser.add_argument('--efficientnet', default='output/efficientnet-b3-resume-20261015')
    parser.add_argument('--output', required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        verify(args.output)
    else:
        run(args.parent, args.efficientnet, args.output)
