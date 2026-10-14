import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ensemble import load_model, temper
from train import Drawings, predict


def verify(output, write_submission=False):
    output = Path(output)
    summary = json.loads((output / 'summary.json').read_text())
    specs = [{**item, 'folder': item.get('folder', output.name),
              'flip': item.get('flip', summary.get('flip_tta', False)),
              'weight': item.get('weight', 1 / len(summary['models'])),
              'temperature': item.get('temperature', 1)} for item in summary['models']]
    assert abs(sum(spec['weight'] for spec in specs) - 1) < 1e-8
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    weights = torch.load('input/baseline.pth', map_location='cpu', weights_only=True)
    labels = np.load('input/val_labels.npy')
    loaders = {split: DataLoader(Drawings(np.load(f'input/{split}_images.npy', mmap_mode='r')),
                                batch_size=512, num_workers=4, persistent_workers=True,
                                pin_memory=True) for split in ('val', 'test')}
    predictions = {}
    for split, loader in loaders.items():
        combined = None
        for spec in specs:
            model = load_model(spec, weights, output.parent)
            probs = temper(predict(model, loader, flip=spec['flip']), spec['temperature']) * spec['weight']
            combined = probs if combined is None else combined + probs
            del model
        predictions[split] = combined.argmax(1)
    correct = int((predictions['val'] == labels).sum())
    score = correct / len(labels)
    if write_submission:
        pd.DataFrame({'id': np.arange(len(predictions['test'])),
                      'label': predictions['test']}).to_csv(output / 'submission.csv', index=False)
        summary['selected'].update(correct=correct, accuracy=score)
        summary['inference'] = {'cudnn_benchmark': False, 'cudnn_deterministic': True,
                                'cudnn_allow_tf32': False}
        summary['val_above_6_threshold'] = score > .76102
        summary['val_above_8_threshold'] = score > .77489
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    assert abs(score - summary['selected']['accuracy']) <= 1e-4
    submission = pd.read_csv(output / 'submission.csv')
    assert submission.columns.tolist() == ['id', 'label']
    assert submission['id'].tolist() == list(range(len(predictions['test'])))
    assert submission['label'].between(0, 344).all()
    differences = int((submission['label'].to_numpy() != predictions['test']).sum())
    result = {'correct': correct, 'val_examples': len(labels), 'accuracy': score,
              'recorded_accuracy': summary['selected']['accuracy'],
              'submission_rows': len(submission), 'submission_differences': differences}
    (output / 'verification.json').write_text(json.dumps(result, indent=2))
    assert differences == 0, result
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output')
    parser.add_argument('--write-submission', action='store_true')
    args = parser.parse_args()
    verify(args.output, args.write_submission)
