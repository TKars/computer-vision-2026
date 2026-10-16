import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from blend_val import temper
from split95 import sha256


def run(output):
    from pretrained_vit import make_network, predict, setup

    output = Path(output)
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['val_examples'] == 34500 and 'convnext' in summary
    recipe = summary['selected']
    flip = recipe['mode'] == 'flip'
    arrays, loaders = setup(False, 'input/split95-seed45')
    paths, hashes, weights_sha = {}, {}, {}
    for name, key, filename in [('vit', 'vit', 'vit.pth'), ('efficientnet', 'efficientnet', 'model.pth')]:
        if recipe[name + '_weight'] == 0:
            continue
        folder = Path(summary[key])
        config = json.loads((folder / 'config.json').read_text())
        path = output / f'test_{name}.npy'
        assert not path.exists(), 'Вероятности test уже существуют'
        model = make_network(config['model'])
        model.load_state_dict(torch.load(folder / filename, map_location='cpu', weights_only=True))
        probs = predict(model, loaders['test'], flip)
        np.save(path, probs)
        paths[name] = str(path)
        weights_sha[str(folder / filename)] = sha256(folder / filename)
        del model, probs
        torch.cuda.empty_cache()
    if recipe['convnext_weight'] > 0:
        folder = Path(summary['convnext'])
        paths['convnext'] = str(folder / ('test_flip.npy' if flip else 'test_plain.npy'))
        weights_sha[str(folder / 'model.pth')] = sha256(folder / 'model.pth')
    combined = None
    for name, path in paths.items():
        probs = np.load(path, mmap_mode='r')
        assert probs.shape == (172500, 345) and np.isfinite(probs).all() and (probs >= 0).all()
        assert np.allclose(probs.sum(1), 1, atol=1e-5)
        contribution = recipe[name + '_weight'] * temper(probs, recipe['temperature'])
        combined = contribution if combined is None else combined + contribution
        hashes[path] = sha256(Path(path))
    predictions = combined.argmax(1)
    pd.DataFrame({'id': np.arange(len(predictions)), 'label': predictions}).to_csv(output / 'submission.csv', index=False)
    metadata = {'recipe': recipe, 'probabilities': paths, 'sha256': hashes,
                'model_weights_sha256': weights_sha, 'submission_rows': len(predictions),
                'val_examples': len(arrays['val_labels']), 'kaggle_submitted': False}
    (output / 'submission_summary.json').write_text(json.dumps(metadata, indent=2))


def verify(output):
    output = Path(output)
    summary = json.loads((output / 'submission_summary.json').read_text())
    recipe = summary['recipe']
    assert recipe == json.loads((output / 'summary.json').read_text())['selected']
    for path, digest in {**summary['sha256'], **summary['model_weights_sha256']}.items():
        assert sha256(Path(path)) == digest
    combined = None
    for name, path in summary['probabilities'].items():
        probs = np.load(path, mmap_mode='r')
        contribution = recipe[name + '_weight'] * temper(probs, recipe['temperature'])
        combined = contribution if combined is None else combined + contribution
    predictions = combined.argmax(1)
    submission = pd.read_csv(output / 'submission.csv')
    assert submission.columns.tolist() == ['id', 'label'] and len(submission) == 172500
    assert submission['id'].tolist() == list(range(172500)) and submission['label'].between(0, 344).all()
    differences = int((predictions != submission['label'].to_numpy()).sum())
    assert differences == 0
    result = {'submission_rows': len(predictions), 'submission_differences': differences,
              'input_sha256_verified': True, 'recipe_reloaded': True, 'kaggle_submitted': False}
    (output / 'submission_verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify: verify(args.output)
    else: run(args.output)
