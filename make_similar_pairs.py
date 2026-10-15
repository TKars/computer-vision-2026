import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    split = Path('input/split95-seed45')
    train = np.load(split / 'train_indices.npy')
    val = np.load(split / 'val_indices.npy')
    old_train_size = len(np.load('input/train_labels.npy', mmap_mode='r'))
    moved = train[train >= old_train_size] - old_train_size
    assert len(moved) == 69000 and not np.intersect1d(train, val).size
    labels = np.load('input/val_labels.npy')[moved]
    assert np.array_equal(np.bincount(labels), np.full(345, 200))
    names = np.load('input/classes.npy')
    sources = {'vit_plain': Path('output/pretrained-tta-20261015-0926/snapshot/val_plain.npy'),
               'vit_flip': Path('output/pretrained-tta-20261015-0926/snapshot/val_flip.npy'),
               'efficientnet_flip': Path('output/efficientnet-b3-resume-20261015/val_flip.npy')}
    errors = {}
    for name, path in sources.items():
        predictions = np.load(path, mmap_mode='r')[moved].argmax(1)
        confusion = np.zeros((345, 345), dtype=np.int64)
        np.add.at(confusion, (labels, predictions), 1)
        errors[name] = confusion + confusion.T
    pairs = []
    for a, b in zip(*np.triu_indices(345, 1)):
        if errors['vit_plain'][a, b] >= 14 and all(errors[k][a, b] >= 10 for k in ('vit_flip', 'efficientnet_flip')):
            pairs.append({'ids': [int(a), int(b)], 'classes': [str(names[a]), str(names[b])],
                          'errors': {k: int(v[a, b]) for k, v in errors.items()}})
    pairs.sort(key=lambda row: -row['errors']['vit_plain'])
    result = {'source': '69000 examples moved to train; retained val excluded',
              'split_sha256': sha256(split / 'manifest.json'),
              'train_indices_sha256': sha256(split / 'train_indices.npy'),
              'classes_sha256': sha256(Path('input/classes.npy')),
              'examples': len(moved), 'seed': 45,
              'thresholds': {'vit_plain': 14, 'vit_flip': 10, 'efficientnet_flip': 10},
              'probability_sha256': {str(p): sha256(p) for p in sources.values()},
              'pairs': pairs}
    assert pairs
    Path('similar_pairs.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'pairs': len(pairs), 'classes': len({k for p in pairs for k in p['ids']}),
                      'top_pairs': pairs[:5]}), flush=True)


if __name__ == '__main__':
    main()
