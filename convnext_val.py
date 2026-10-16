import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from convnext import DATA, make_model, predict, setup
from split95 import sha256


def run(parent, output, epoch=None):
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = None
    if epoch is None:
        metadata = (parent / 'best.json').read_bytes()
        stamp = (parent / 'model.pth').stat().st_mtime_ns
        shutil.copy2(parent / 'model.pth', output / 'model.pth')
        assert metadata == (parent / 'best.json').read_bytes()
        assert stamp == (parent / 'model.pth').stat().st_mtime_ns
        assert (parent / 'best.json').stat().st_mtime_ns >= stamp
        for name in ('config.json', 'history.csv'):
            shutil.copy2(parent / name, output / name)
    else:
        path = parent / f'epoch-{epoch:02d}.pth'
        saved = torch.load(path, map_location='cpu', weights_only=False)
        row = saved['history'][-1]
        assert saved['epoch'] == row['epoch'] == epoch
        config = saved['config']
        assert saved['batch'] == (config['train_examples'] + config['batch_size'] - 1) // config['batch_size']
        state = 'ema' if row['ema_accuracy'] >= row['accuracy'] else 'raw'
        metadata = json.dumps({'epoch': epoch, 'state': state,
                               'accuracy': row['ema_accuracy'] if state == 'ema' else row['accuracy']}).encode()
        torch.save(saved['ema' if state == 'ema' else 'model'], output / 'model.pth')
        (output / 'config.json').write_text(json.dumps(config, indent=2))
        pd.DataFrame(saved['history']).to_csv(output / 'history.csv', index=False)
        checkpoint = {'path': str(path), 'sha256': sha256(path)}
        del saved
    (output / 'best.json').write_bytes(metadata)
    for name in ('val_indices.npy', 'manifest.json'):
        shutil.copy2(DATA / name, output / name)
    config = json.loads((output / 'config.json').read_text())
    assert config['split_sha256'] == sha256(DATA / 'manifest.json')
    assert config['classes_sha256'] == sha256(Path('input/classes.npy'))
    arrays = setup()
    assert len(arrays['val_labels']) == config['val_examples'] == 34500
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.25)
    model = make_model()
    model.load_state_dict(torch.load(output / 'model.pth', map_location='cpu', weights_only=True))
    scores = []
    for flip in (False, True):
        mode = 'flip' if flip else 'plain'
        probabilities = predict(model, arrays['val_images'], config['batch_size'], flip)
        np.save(output / f'val_{mode}.npy', probabilities)
        correct = int((probabilities.argmax(1) == arrays['val_labels']).sum())
        scores.append({'model': 'convnext_' + mode, 'correct': correct, 'accuracy': correct / len(probabilities)})
        print(json.dumps(scores[-1]), flush=True)
    assert scores[0]['accuracy'] == json.loads(metadata)['accuracy']
    pd.DataFrame(scores).to_csv(output / 'scores.csv', index=False)
    summary = {'source': str(parent), 'snapshot': json.loads(metadata), 'scores': scores,
               'val_examples': 34500, 'validation_only': True,
               'model_sha256': sha256(output / 'model.pth'), 'checkpoint': checkpoint}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    shutil.copy2(__file__, output / 'source.py')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--epoch', type=int)
    args = parser.parse_args()
    run(args.parent, args.output, args.epoch)
