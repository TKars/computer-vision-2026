import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from blend_pair95 import inputs
from blend_val import temper
from split95 import sha256


NAMES = ('vit_weight', 'efficientnet_weight', 'similar_vit_weight', 'convnext_weight')


def load(vit, efficientnet, similar_vit, convnext):
    labels, arrays, hashes = inputs(vit, efficientnet, similar_vit)
    assert np.array_equal(np.load(convnext / 'val_indices.npy'), np.load(vit / 'val_indices.npy'))
    for mode in arrays:
        path = convnext / f'val_{mode}.npy'
        probs = np.load(path, mmap_mode='r')
        assert probs.shape == (34500, 345) and np.isfinite(probs).all() and (probs >= 0).all()
        assert np.allclose(probs.sum(1), 1, atol=1e-5)
        arrays[mode].append(probs)
        hashes[str(path)] = sha256(path)
    for path in (convnext / 'val_indices.npy', convnext / 'best.json'):
        hashes[str(path)] = sha256(path)
    return labels, arrays, hashes


def mix(probabilities, recipe):
    return sum(recipe[name] * probs for name, probs in zip(NAMES, probabilities))


def run(vit, efficientnet, similar_vit, convnext, output):
    labels, arrays, hashes = load(vit, efficientnet, similar_vit, convnext)
    output.mkdir(parents=True, exist_ok=False)
    values = [(0.25,) * 4] + [(.3 * (1 - w), .25 * (1 - w), w, .45 * (1 - w))
                              for w in (0, .05, .1, .15, .2)]
    rows = []
    for mode, models in arrays.items():
        for temperature in (1, 2, 4):
            probabilities = [temper(probs, temperature) for probs in models]
            for row in values:
                recipe = {'mode': mode, 'temperature': temperature, **dict(zip(NAMES, row))}
                correct = int((mix(probabilities, recipe).argmax(1) == labels).sum())
                rows.append({**recipe, 'correct': correct, 'accuracy': correct / len(labels)})
    selected = max(rows, key=lambda row: row['correct'])
    probabilities = [temper(p, selected['temperature']) for p in arrays[selected['mode']]]
    predictions = mix(probabilities, selected).argmax(1)
    assert int((predictions == labels).sum()) == selected['correct']
    np.save(output / 'val_predictions.npy', predictions)
    summary = {'selected': selected, 'val_examples': len(labels), 'candidates': len(rows),
               'weight_candidates': len(values), 'search': 'six_fixed_weight_recipes',
               'best_by_mode': {mode: max((r for r in rows if r['mode'] == mode), key=lambda r: r['correct'])
                                for mode in arrays},
               'best_without_similar_vit': max((r for r in rows if r['similar_vit_weight'] == 0), key=lambda r: r['correct']),
               'vit': str(vit), 'efficientnet': str(efficientnet), 'similar_vit': str(similar_vit),
               'convnext': str(convnext), 'sha256': hashes, 'selection_on_validation': True,
               'validation_only': True}
    pd.DataFrame(rows).sort_values('correct', ascending=False).to_csv(output / 'scores.csv', index=False)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    (output / 'source.py').write_text(Path(__file__).read_text())
    print(json.dumps({k: v for k, v in summary.items() if k != 'sha256'}), flush=True)


def verify(output):
    summary = json.loads((output / 'summary.json').read_text())
    for path, digest in summary['sha256'].items():
        assert sha256(Path(path)) == digest
    labels, arrays, _ = load(*(Path(summary[k]) for k in ('vit', 'efficientnet', 'similar_vit', 'convnext')))
    recipe = summary['selected']
    probabilities = [temper(p, recipe['temperature']) for p in arrays[recipe['mode']]]
    predictions = mix(probabilities, recipe).argmax(1)
    assert np.array_equal(predictions, np.load(output / 'val_predictions.npy'))
    assert int((predictions == labels).sum()) == recipe['correct']
    result = {'val_examples': len(labels), 'correct': recipe['correct'], 'accuracy': recipe['accuracy'],
              'recipe_prediction_differences': 0, 'input_sha256_verified': True,
              'same_val_order_verified': True, 'validation_only': True}
    (output / 'verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vit', type=Path, default=Path('output/pretrained-vit-95-20261015'))
    parser.add_argument('--efficientnet', type=Path, default=Path('output/efficientnet-b3-resume-20261015'))
    parser.add_argument('--similar-vit', type=Path)
    parser.add_argument('--convnext', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        verify(args.output)
    else:
        assert args.similar_vit and args.convnext
        run(args.vit, args.efficientnet, args.similar_vit, args.convnext, args.output)
