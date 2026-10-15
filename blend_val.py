import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


def temper(probs, temperature):
    if temperature == 1:
        return probs
    probs = np.maximum(probs, 1e-12) ** (1 / temperature)
    return probs / probs.sum(1, keepdims=True)


def refresh_resnets(parent, output):
    import torch
    from torch.utils.data import DataLoader

    from ensemble import load_model
    from train import Drawings, predict

    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.25)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    output.mkdir()
    summary = json.loads((parent / 'summary.json').read_text())
    loader = DataLoader(Drawings(np.load('input/val_images.npy', mmap_mode='r')),
                        batch_size=512, num_workers=2, pin_memory=True, persistent_workers=True)
    weights = torch.load('input/baseline.pth', map_location='cpu', weights_only=True)
    labels = np.load('input/val_labels.npy')
    scores = []
    for index, spec in enumerate(summary['models']):
        model = load_model(spec, weights, Path('output'))
        probs = predict(model, loader, flip=True)
        np.save(output / f'val_{index}.npy', probs)
        score = {'resnet': index, 'correct': int((probs.argmax(1) == labels).sum())}
        scores.append(score)
        print(json.dumps(score), flush=True)
        del model, probs
        torch.cuda.empty_cache()
    metadata = {'source': str(parent), 'models': summary['models'], 'val_examples': len(labels),
                'scores': scores, 'validation_only': True, 'batch_size': 512,
                'gpu_memory_fraction': 0.25, 'precision': 'float32', 'deterministic': True}
    (output / 'summary.json').write_text(json.dumps(metadata, indent=2))


def run(vit, resnets, scratch, output, refresh):
    vit, resnets, output = Path(vit), Path(resnets), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if refresh:
        refresh_resnets(resnets, output / 'resnets')
        resnets = output / 'resnets'
    labels = np.load('input/val_labels.npy')
    assert len(labels) == 103500
    cnn_specs = json.loads((resnets / 'summary.json').read_text())['models']
    groups = {'vit': [str(vit / 'val_flip.npy')]}
    for index in range(len(cnn_specs)):
        groups[f'resnet_{index}'] = [str(resnets / f'val_{index}.npy')]
    groups['resnets'] = [str(resnets / f'val_{i}.npy') for i in range(len(cnn_specs))]
    if scratch:
        groups['scratch_vit'] = [str(Path(scratch) / 'val_flip.npy')]
    raw = {path: np.load(path, mmap_mode='r') for paths in groups.values() for path in paths}
    assert all(probs.shape == (103500, 345) for probs in raw.values())

    def group(name, temperature):
        return sum(temper(raw[path], temperature) for path in groups[name]) / len(groups[name])

    transformed = {(name, t): group(name, t) for name in groups for t in (1, 2, 4)}

    def combine(recipe, lookup):
        return sum(float(item['weight']) * lookup(item['group'], item['temperature']) for item in recipe)

    rows, recipes = [], []
    def score(recipe):
        probs = combine(recipe, lambda name, t: transformed[name, t])
        correct = int((probs.argmax(1) == labels).sum())
        rows.append({'index': len(rows), 'models': len(recipe), 'correct': correct,
                     'accuracy': correct / len(labels), 'recipe': json.dumps(recipe)})
        recipes.append(recipe)

    for name in groups:
        score([{'group': name, 'weight': 1.0, 'temperature': 1}])
    score([{'group': 'resnets', 'weight': 1.0, 'temperature': 4}])
    baseline_rows = rows.copy()
    for other in groups:
        if other == 'vit':
            continue
        for tv, to, weight in itertools.product((1, 2, 4), (1, 2, 4), (0.25, 0.5, 0.75, 0.9)):
            score([{'group': 'vit', 'weight': weight, 'temperature': tv},
                   {'group': other, 'weight': 1 - weight, 'temperature': to}])
    best_pair = max((row for row in rows if row['models'] == 2), key=lambda row: row['correct'])
    pair = recipes[best_pair['index']]
    for weight in np.arange(max(.05, pair[0]['weight'] - .15), min(.975, pair[0]['weight'] + .15) + .001, .025):
        score([{**pair[0], 'weight': float(weight)}, {**pair[1], 'weight': float(1 - weight)}])
    if scratch:
        ratios = [(.5, .25, .25), (.75, .125, .125), (.625, .125, .25),
                  (.625, .25, .125), (.5, .125, .375)]
        for temperatures, weights in itertools.product(itertools.product((1, 2, 4), repeat=3), ratios):
            score([{'group': name, 'weight': weight, 'temperature': t}
                   for name, weight, t in zip(('vit', 'scratch_vit', 'resnets'), weights, temperatures)])
    selected = max(rows, key=lambda row: row['correct'])
    recipe = recipes[selected['index']]
    predictions = combine(recipe, lambda name, t: transformed[name, t]).argmax(1)
    np.save(output / 'val_predictions.npy', predictions)
    repeated = combine(recipe, group).argmax(1)
    differences = int((repeated != predictions).sum())
    assert differences == 0
    correct = int((repeated == labels).sum())
    assert correct == selected['correct']
    table = pd.DataFrame(rows).sort_values('correct', ascending=False)
    table.to_csv(output / 'scores.csv', index=False)
    hashes = {}
    for path in raw:
        with open(path, 'rb') as stream:
            hashes[path] = hashlib.file_digest(stream, 'sha256').hexdigest()
    summary = {'selected': selected, 'recipe': recipe, 'groups': groups,
               'resnet_models': cnn_specs, 'vit_snapshot': json.loads((vit / 'best.json').read_text()),
               'baselines': baseline_rows,
               'best_pair': max((row for row in rows if row['models'] == 2), key=lambda row: row['correct']),
               'val_examples': len(labels), 'candidates': len(rows), 'refreshed_resnets': refresh,
               'above_6_threshold': selected['accuracy'] > .76102,
               'above_8_threshold': selected['accuracy'] > .77489,
               'above_10_threshold': selected['accuracy'] > .78013, 'probability_sha256': hashes}
    if scratch:
        summary['scratch_snapshot'] = json.loads((Path(scratch) / 'summary.json').read_text())['snapshot']
    verification = {'correct': correct, 'val_examples': len(labels), 'accuracy': correct / len(labels),
                    'recipe_prediction_differences': differences}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    (output / 'verification.json').write_text(json.dumps(verification, indent=2))
    (output / 'source.py').write_text(Path(__file__).read_text())
    print(json.dumps({'selected': selected, 'recipe': recipe, 'verification': verification}), flush=True)
    print(table.head(10).to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vit', default='output/pretrained-tta-20261015-0926/snapshot')
    parser.add_argument('--resnets', default='output/ensemble-20261014')
    parser.add_argument('--scratch', default='output/vit-tta-20261015-0025')
    parser.add_argument('--output', required=True)
    parser.add_argument('--refresh-resnets', action='store_true')
    args = parser.parse_args()
    run(args.vit, args.resnets, args.scratch, args.output, args.refresh_resnets)
