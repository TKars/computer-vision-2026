import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from improve import resize_model
from train import Drawings, make_model, predict


def temper(probabilities, temperature):
    if temperature == 1:
        return probabilities
    probs = np.maximum(probabilities, 1e-12) ** (1 / temperature)
    return probs / probs.sum(1, keepdims=True)


def load_model(spec, weights, root):
    model = resize_model(weights) if spec['stem'] == 56 else make_model(spec['stem'], weights)
    if spec['checkpoint']:
        model.load_state_dict(torch.load(root / spec['folder'] / spec['checkpoint'],
                                         map_location='cpu', weights_only=True))
    return model


def run(folders, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(folders[0]).parent
    labels = np.load('input/val_labels.npy')
    weights = torch.load('input/baseline.pth', map_location='cpu', weights_only=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    loaders = {split: DataLoader(Drawings(np.load(f'input/{split}_images.npy', mmap_mode='r')),
                                batch_size=512, pin_memory=True, num_workers=4,
                                persistent_workers=True)
               for split in ('val', 'test')}
    specs, original_recipes = [], []
    for folder in map(Path, folders):
        summary = json.loads((folder / 'summary.json').read_text())
        recipe = []
        for item in summary['models']:
            spec = {key: item[key] for key in ('stem', 'checkpoint')}
            spec['folder'] = item.get('folder', folder.name)
            spec['flip'] = item.get('flip', summary.get('flip_tta', False))
            if spec not in specs:
                specs.append(spec)
            recipe.append({**spec, 'weight': item.get('weight', 1 / len(summary['models'])),
                           'temperature': item.get('temperature', 1)})
        original_recipes.append(recipe)
        if (folder / 'model.pth').exists():
            config = json.loads((folder / 'config.json').read_text())
            spec = {'stem': config['stem'], 'checkpoint': 'model.pth',
                    'folder': folder.name, 'flip': True}
            if spec not in specs:
                specs.append(spec)
    probabilities = []
    for index, spec in enumerate(specs):
        model = load_model(spec, weights, root)
        probs = predict(model, loaders['val'], flip=spec['flip'])
        assert probs.shape == (len(labels), 345)
        probabilities.append(probs)
        np.save(output / f'val_{index}.npy', probs)
        del model
    scores, recipes = [], []
    def add(name, probs, recipe):
        correct = int((probs.argmax(1) == labels).sum())
        scores.append({'model': name, 'correct': correct, 'accuracy': correct / len(labels)})
        recipes.append(recipe)
    for index, probs in enumerate(probabilities):
        add(f'model_{index}', probs, [{**specs[index], 'weight': 1.0, 'temperature': 1}])
    for index, recipe in enumerate(original_recipes):
        combined = None
        for item in recipe:
            spec = {key: item[key] for key in ('stem', 'checkpoint', 'folder', 'flip')}
            probs = temper(probabilities[specs.index(spec)], item['temperature']) * item['weight']
            combined = probs if combined is None else combined + probs
        add(f'original_run_{index}', combined, recipe)
    transformed = {(index, temperature): temper(probs, temperature)
                   for index, probs in enumerate(probabilities) for temperature in (1, 2)}
    for first, second in itertools.combinations(range(len(specs)), 2):
        for t1, t2 in itertools.product((1, 2), repeat=2):
            for ratio in (0.25, 0.5, 0.75):
                probs = ratio * transformed[first, t1] + (1 - ratio) * transformed[second, t2]
                add(f'{first}_{second}_t{t1}_{t2}_w{ratio}', probs,
                    [{**specs[first], 'weight': ratio, 'temperature': t1},
                     {**specs[second], 'weight': 1 - ratio, 'temperature': t2}])
    for count in range(3, len(specs) + 1):
        for indices in itertools.combinations(range(len(specs)), count):
            for temperature in (1, 2):
                probs = sum(transformed[index, temperature] for index in indices) / count
                add(f'mean_{indices}_t{temperature}', probs,
                    [{**specs[index], 'weight': 1 / count, 'temperature': temperature}
                     for index in indices])
    table = pd.DataFrame(scores)
    table.to_csv(output / 'scores.csv', index=False)
    selected = int(table['accuracy'].idxmax())
    recipe = recipes[selected]
    test_probs = None
    for spec in recipe:
        model = load_model(spec, weights, root)
        probs = temper(predict(model, loaders['test'], flip=spec['flip']), spec['temperature']) * spec['weight']
        test_probs = probs if test_probs is None else test_probs + probs
        del model
    submission = pd.DataFrame({'id': np.arange(len(test_probs)), 'label': test_probs.argmax(1)})
    assert submission['label'].between(0, 344).all()
    submission.to_csv(output / 'submission.csv', index=False)
    summary = {'selected': scores[selected], 'models': recipe, 'val_examples': len(labels),
               'submission_rows': len(submission), 'val_above_6_threshold': scores[selected]['accuracy'] > .76102,
               'val_above_8_threshold': scores[selected]['accuracy'] > .77489}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    (output / 'source.py').write_text(Path(__file__).read_text())
    print(table.nlargest(10, 'accuracy').to_string(index=False))
    print(json.dumps(summary))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--output', default='output/ensemble')
    args = parser.parse_args()
    run(args.runs, args.output)
