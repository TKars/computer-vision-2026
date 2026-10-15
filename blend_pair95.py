import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from blend_val import temper
from split95 import sha256


def inputs(vit, efficientnet, similar_vit=None):
    split = Path('input/split95-seed45')
    manifest = json.loads((split / 'manifest.json').read_text())
    for name in ('val_indices.npy', 'val_labels.npy'):
        assert sha256(split / name) == manifest['files_sha256'][name]
    indices = np.load(split / 'val_indices.npy') - len(np.load('input/train_labels.npy'))
    labels = np.load(split / 'val_labels.npy')
    assert len(labels) == 34500 and len(np.unique(indices)) == len(indices)
    assert np.array_equal(labels, np.load('input/val_labels.npy')[indices])
    assert np.array_equal(np.load(vit / 'val_indices.npy'), np.load(split / 'val_indices.npy'))
    assert sha256(Path('input/val_labels.npy')) == manifest['sources_sha256']['val_labels.npy']
    folders = [(vit, False), (efficientnet, True)]
    if similar_vit is not None:
        assert np.array_equal(np.load(similar_vit / 'val_indices.npy'), np.load(split / 'val_indices.npy'))
        folders.append((similar_vit, False))
    arrays, hashes = {}, {}
    for mode in ('plain', 'flip'):
        pair = []
        for folder, old_split in folders:
            path = folder / f'val_{mode}.npy'
            probs = np.load(path, mmap_mode='r')
            assert probs.shape == (103500 if old_split else 34500, 345)
            probs = np.asarray(probs[indices] if old_split else probs)
            assert np.isfinite(probs).all() and (probs >= 0).all()
            assert np.allclose(probs.sum(1), 1, atol=1e-5)
            pair.append(probs)
            hashes[str(path)] = sha256(path)
        arrays[mode] = pair
    for path in (split / 'manifest.json', split / 'val_indices.npy', split / 'val_labels.npy',
                 vit / 'best.json', efficientnet / 'best.json'):
        hashes[str(path)] = sha256(path)
    if similar_vit is not None:
        hashes[str(similar_vit / 'best.json')] = sha256(similar_vit / 'best.json')
    return labels, arrays, hashes


def combine(probabilities, recipe):
    weights = [recipe['vit_weight'], recipe['efficientnet_weight']]
    if len(probabilities) == 3:
        weights.append(recipe['similar_vit_weight'])
    return sum(weight * probs for weight, probs in zip(weights, probabilities))


def run(vit, efficientnet, output, similar_vit=None):
    labels, arrays, hashes = inputs(vit, efficientnet, similar_vit)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for mode, pair in arrays.items():
        for temperature in (1, 2, 4):
            probabilities = [temper(probs, temperature) for probs in pair]
            weights = [(step / 20, 1 - step / 20) for step in range(21)]
            if similar_vit is not None:
                weights = [(a / 20, b / 20, (20 - a - b) / 20)
                           for a in range(21) for b in range(21 - a)] + [(1 / 3,) * 3]
            for values in weights:
                recipe = {'mode': mode, 'temperature': temperature,
                          'vit_weight': values[0], 'efficientnet_weight': values[1]}
                if similar_vit is not None:
                    recipe['similar_vit_weight'] = values[2]
                predictions = combine(probabilities, recipe).argmax(1)
                correct = int((predictions == labels).sum())
                rows.append({**recipe, 'correct': correct, 'accuracy': correct / len(labels)})
    selected = max(rows, key=lambda row: row['correct'])
    probabilities = [temper(probs, selected['temperature']) for probs in arrays[selected['mode']]]
    np.save(output / 'val_predictions.npy', combine(probabilities, selected).argmax(1))
    baselines = [row for row in rows if row['temperature'] == 1
                 and all(row.get(name, 0) in (0, 1) for name in
                         ('vit_weight', 'efficientnet_weight', 'similar_vit_weight'))]
    fixed = [row for row in rows if row.get('similar_vit_weight', 0) == 0
             and (row['temperature'], row['vit_weight']) in ((1, .5), (4, .45))]
    summary = {'selected': selected, 'baselines': baselines, 'fixed_recipes': fixed,
               'best_by_mode': {mode: max((r for r in rows if r['mode'] == mode), key=lambda r: r['correct'])
                                for mode in arrays},
               'val_examples': len(labels), 'validation_only': True, 'candidates': len(rows),
               'vit': str(vit), 'efficientnet': str(efficientnet), 'sha256': hashes,
               'selection_on_validation': True}
    if similar_vit is not None:
        summary['similar_vit'] = str(similar_vit)
        summary['equal_weights'] = [r for r in rows if r['vit_weight'] == r['efficientnet_weight'] == r['similar_vit_weight']]
    pd.DataFrame(rows).sort_values('correct', ascending=False).to_csv(output / 'scores.csv', index=False)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    (output / 'source.py').write_text(Path(__file__).read_text())
    print(json.dumps({key: value for key, value in summary.items() if key != 'sha256'}), flush=True)


def verify(output):
    summary = json.loads((output / 'summary.json').read_text())
    for path, digest in summary['sha256'].items():
        assert sha256(Path(path)) == digest
    selected = summary['selected']
    similar_vit = Path(summary['similar_vit']) if 'similar_vit' in summary else None
    labels, arrays, _ = inputs(Path(summary['vit']), Path(summary['efficientnet']), similar_vit)
    probabilities = [temper(probs, selected['temperature']) for probs in arrays[selected['mode']]]
    predictions = combine(probabilities, selected).argmax(1)
    differences = int((predictions != np.load(output / 'val_predictions.npy')).sum())
    correct = int((predictions == labels).sum())
    assert differences == 0 and correct == selected['correct']
    result = {'val_examples': len(labels), 'correct': correct, 'accuracy': correct / len(labels),
              'recipe_prediction_differences': differences, 'input_sha256_verified': True,
              'same_val_order_verified': True, 'validation_only': True}
    (output / 'verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vit', default='output/pretrained-vit-95-20261015')
    parser.add_argument('--efficientnet', default='output/efficientnet-b3-resume-20261015')
    parser.add_argument('--similar-vit', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        verify(args.output)
    else:
        run(Path(args.vit), Path(args.efficientnet), args.output, args.similar_vit)
