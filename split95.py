import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    source = Path('input')
    output = source / 'split95-seed45'
    output.mkdir(exist_ok=False)
    names = ['train_images', 'train_labels', 'val_images', 'val_labels']
    arrays = {name: np.load(source / f'{name}.npy', mmap_mode='r') for name in names}
    sources = {name + '.npy': sha256(source / f'{name}.npy') for name in names}
    assert np.array_equal(np.bincount(arrays['train_labels']), np.full(345, 1700))
    assert np.array_equal(np.bincount(arrays['val_labels']), np.full(345, 300))
    rng = np.random.default_rng(45)
    kept = np.sort(np.concatenate([
        rng.choice(np.flatnonzero(arrays['val_labels'] == label), 100, replace=False)
        for label in range(345)]))
    extra = np.setdiff1d(np.arange(len(arrays['val_labels'])), kept)
    train_indices = np.concatenate([np.arange(len(arrays['train_labels'])),
                                    len(arrays['train_labels']) + extra])
    val_indices = len(arrays['train_labels']) + kept
    assert not np.intersect1d(train_indices, val_indices).size
    assert np.array_equal(np.sort(np.concatenate([train_indices, val_indices])), np.arange(690000))
    np.save(output / 'train_indices.npy', train_indices)
    np.save(output / 'val_indices.npy', val_indices)
    for kind in ('images', 'labels'):
        np.save(output / f'train_{kind}.npy', np.concatenate([
            arrays[f'train_{kind}'], arrays[f'val_{kind}'][extra]]))
        np.save(output / f'val_{kind}.npy', arrays[f'val_{kind}'][kept])
    (output / 'test_images.npy').symlink_to('../test_images.npy')
    for split, count in [('train', 1900), ('val', 100)]:
        labels = np.load(output / f'{split}_labels.npy')
        assert np.array_equal(np.bincount(labels), np.full(345, count))
    assert all(sha256(source / name) == digest for name, digest in sources.items())
    manifest = {'seed': 45, 'index_order': 'original train, then original val',
                'train_examples': 655500, 'val_examples': 34500,
                'train_per_class': 1900, 'val_per_class': 100,
                'sources_sha256': sources,
                'files_sha256': {path.name: sha256(path) for path in sorted(output.glob('*.npy'))
                                 if not path.is_symlink()},
                'overlap': 0, 'test_used_for_split': False}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
