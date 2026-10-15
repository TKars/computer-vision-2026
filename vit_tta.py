import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def run(parent, output):
    parent, output = Path(parent), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((parent / 'config.json').read_text())
    if config.get('pretrained'):
        from pretrained_vit import make_vit, predict, setup
    else:
        from train import predict
        from vit import make_vit, setup
    for attempt in range(3):
        try:
            metadata = (parent / 'best.json').read_bytes()
            stamp = (parent / 'vit.pth').stat().st_mtime_ns
            state = torch.load(parent / 'vit.pth', map_location='cpu', weights_only=True)
            if (metadata == (parent / 'best.json').read_bytes()
                    and stamp == (parent / 'vit.pth').stat().st_mtime_ns
                    and (parent / 'best.json').stat().st_mtime_ns >= stamp):
                break
        except (RuntimeError, EOFError):
            pass
        time.sleep(0.5)
    else:
        raise RuntimeError('Checkpoint is being updated')
    torch.save(state, output / 'vit.pth')
    del state
    best = json.loads(metadata)
    (output / 'best.json').write_bytes(metadata)
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    data_dir = Path(config.get('data_dir', 'input'))
    arrays, loaders = setup(data_dir=data_dir) if config.get('pretrained') else setup()
    assert len(arrays['val_labels']) == config.get('val_examples', 103500)
    for name in ('val_indices.npy', 'manifest.json'):
        if (data_dir / name).exists():
            shutil.copy2(data_dir / name, output / name)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.25)
    model = make_vit()
    model.load_state_dict(torch.load(output / 'vit.pth', map_location='cpu', weights_only=True))
    scores = []
    for flip in (False, True):
        probs = predict(model, loaders['val'], flip=flip)
        correct = int((probs.argmax(1) == arrays['val_labels']).sum())
        scores.append({'model': 'vit_flip' if flip else 'vit',
                       'correct': correct, 'accuracy': correct / len(probs)})
        np.save(output / ('val_flip.npy' if flip else 'val_plain.npy'), probs)
        print(json.dumps(scores[-1]), flush=True)
    pd.DataFrame(scores).to_csv(output / 'scores.csv', index=False)
    assert abs(scores[0]['accuracy'] - best['accuracy']) <= 1e-4
    summary = {'source': str(parent), 'snapshot': best, 'val_examples': len(arrays['val_labels']),
               'scores': scores, 'delta_flip': scores[1]['accuracy'] - scores[0]['accuracy']}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    (output / 'source.py').write_text(Path(__file__).read_text())
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', default='output/vit-20261014')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.parent, args.output)
