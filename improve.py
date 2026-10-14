import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from train import Drawings, accuracy, fit, make_model, predict, prepare, set_seed


def resize_model(weights):
    return nn.Sequential(nn.Upsample(size=(56, 56), mode='bilinear', align_corners=False),
                         make_model(0, weights))


def run(parent, output, minutes=60, kind='resize56'):
    parent, output = Path(parent), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / 'history.csv').exists(), 'Укажите новую папку для запуска'
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    set_seed(43)
    weights = torch.load('input/baseline.pth', map_location='cpu', weights_only=True)
    arrays = {name: np.load(Path('input') / f'{name}.npy', mmap_mode='r')
              for name in ('train_images', 'train_labels', 'val_images', 'val_labels', 'test_images')}
    loaders = {split: DataLoader(Drawings(arrays[f'{split}_images'], arrays.get(f'{split}_labels')),
                                batch_size=512, shuffle=split == 'train', num_workers=4,
                                pin_memory=True, persistent_workers=True)
               for split in ('train', 'val', 'test')}
    stem, initial_state, initial_checkpoint = 56, None, 'baseline.pth'
    if kind == 'parent':
        plain_scores = pd.read_csv(parent / 'scores.csv')
        plain_scores = plain_scores[plain_scores['model'].isin(['resnet18_3', 'resnet18_7', 'finetuned'])]
        best = plain_scores.loc[plain_scores['accuracy'].idxmax(), 'model']
        initial_checkpoint = str(parent / f'{best}.pth')
        initial_state = torch.load(initial_checkpoint, map_location='cpu', weights_only=True)
        stem = initial_state['conv1.weight'].shape[-1]
    lr = 1e-3 if stem == 56 else 3e-4
    config = {'input_size': 56 if stem == 56 else 28, 'initial_weights': initial_checkpoint,
              'kind': kind, 'stem': stem, 'seed': 43,
              'mixup_alpha': 0.2, 'weight_decay': 0.01, 'lr': lr, 'epochs': 60,
              'minutes': minutes, 'torch': torch.__version__}
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    (output / 'source.py').write_text(Path(__file__).read_text())
    (output / 'train_source.py').write_text(Path('train.py').read_text())
    print(json.dumps(config), flush=True)
    model = resize_model(weights) if stem == 56 else make_model(stem, weights)
    if initial_state is not None:
        model.load_state_dict(initial_state)
    initial_accuracy = accuracy(model, loaders['val'])
    set_seed(43)
    fit(model, loaders['train'], loaders['val'], output, 'model', max(0, minutes - 8) * 60,
        60, lr, [], best_accuracy=initial_accuracy, mixup=True, weight_decay=0.01)
    before = accuracy(model, loaders['val'])
    original = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.reset_running_stats()
            layer.momentum = None
    set_seed(43)
    model.train()
    with torch.inference_mode():
        for step, (images, _) in enumerate(loaders['train']):
            model(prepare(images))
            if step == 99:
                break
    after = accuracy(model, loaders['val'])
    calibrated = after > before
    if not calibrated:
        model.load_state_dict(original)
    torch.save(model.state_dict(), output / 'model.pth')
    del original
    labels = arrays['val_labels']
    plain = predict(model, loaders['val'])
    flipped = predict(model, loaders['val'], flip=True)
    np.save(output / 'val_tta.npy', flipped)
    del model
    parent_summary = json.loads((parent / 'summary.json').read_text())
    parent_probs = None
    for spec in parent_summary['models']:
        model = make_model(spec['stem'], weights)
        if spec['checkpoint']:
            model.load_state_dict(torch.load(parent / spec['checkpoint'], map_location='cpu', weights_only=True))
        probs = predict(model, loaders['val'], flip=parent_summary['flip_tta'])
        parent_probs = probs if parent_probs is None else parent_probs + probs
        del model
    parent_probs /= len(parent_summary['models'])
    baseline_score = float(pd.read_csv(parent / 'scores.csv').query('model == "baseline"')['accuracy'].iloc[0])
    predictions = {'parent': parent_probs, 'regularized': plain,
                   'regularized_tta': flipped, 'ensemble': (parent_probs + flipped) / 2}
    scores = [{'model': name, 'accuracy': float((probs.argmax(1) == labels).mean())}
              for name, probs in predictions.items()]
    pd.DataFrame(scores).to_csv(output / 'scores.csv', index=False)
    selected = max(scores, key=lambda row: row['accuracy'])
    specs = []
    if selected['model'] in ('parent', 'ensemble'):
        weight = (0.5 if selected['model'] == 'ensemble' else 1) / len(parent_summary['models'])
        specs.extend({**spec, 'folder': parent.name, 'flip': parent_summary['flip_tta'], 'weight': weight}
                     for spec in parent_summary['models'])
    if selected['model'] != 'parent':
        specs.append({'stem': stem, 'checkpoint': 'model.pth', 'folder': output.name,
                      'flip': selected['model'] != 'regularized',
                      'weight': 0.5 if selected['model'] == 'ensemble' else 1})
    test_probs = None
    for spec in specs:
        model = resize_model(weights) if spec['stem'] == 56 else make_model(spec['stem'], weights)
        if spec['checkpoint']:
            model.load_state_dict(torch.load(output.parent / spec['folder'] / spec['checkpoint'],
                                             map_location='cpu', weights_only=True))
        probs = predict(model, loaders['test'], flip=spec['flip']) * spec['weight']
        test_probs = probs if test_probs is None else test_probs + probs
        del model
    submission = pd.DataFrame({'id': np.arange(len(test_probs)), 'label': test_probs.argmax(1)})
    assert len(submission) == len(arrays['test_images'])
    assert submission['label'].between(0, 344).all()
    submission.to_csv(output / 'submission.csv', index=False)
    summary = {'selected': selected, 'models': specs, 'baseline_accuracy': baseline_score,
               'delta_baseline': selected['accuracy'] - baseline_score,
               'bn_calibration': {'before': before, 'after': after, 'used': calibrated},
               'val_above_6_threshold': selected['accuracy'] > 0.76102,
               'val_above_8_threshold': selected['accuracy'] > 0.77489}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(pd.DataFrame(scores).to_string(index=False), flush=True)
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', default='output/run-20261014')
    parser.add_argument('--output', default='output/resize56')
    parser.add_argument('--minutes', type=float, default=60)
    parser.add_argument('--kind', choices=['resize56', 'parent'], default='resize56')
    args = parser.parse_args()
    run(args.parent, args.output, args.minutes, args.kind)
