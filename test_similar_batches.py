import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

from similar_batches import SimilarBatchSampler, training_loader


class SimilarBatchesTest(unittest.TestCase):
    def test_resume_and_epoch_seed(self):
        labels = np.arange(253) % 3
        pairs = [(0, 1), (1, 2)]
        full = list(SimilarBatchSampler(labels, pairs, epoch=2))
        self.assertEqual(full, list(SimilarBatchSampler(labels, pairs, epoch=2)))
        resumed = SimilarBatchSampler(labels, pairs, epoch=2, start_batch=2)
        self.assertEqual(len(resumed), len(full) - 2)
        self.assertEqual(full[2:], list(resumed))
        self.assertNotEqual(full, list(SimilarBatchSampler(labels, pairs, epoch=3)))
        dataset = TensorDataset(torch.arange(len(labels)))
        loader = training_loader(dataset, labels, 2, 2, pairs, workers=0)
        self.assertEqual(full[2:], [batch[0].tolist() for batch in loader])

    def test_tails_and_unique_indices(self):
        for size in [63, 64, 65, 127, 128, 129, 253]:
            labels = np.arange(size) % 3
            batches = list(SimilarBatchSampler(labels, [(0, 1)], epoch=1))
            self.assertEqual(sum(map(len, batches)), size)
            for batch in batches:
                self.assertEqual(len(set(batch)), len(batch))
                self.assertTrue(all(0 <= k < size for k in batch))

    def test_baseline_order_unchanged(self):
        size = 1001
        dataset = TensorDataset(torch.arange(size))
        expected = torch.randperm(size, generator=torch.Generator().manual_seed(48))
        for skipped in [0, 2]:
            loader = training_loader(dataset, np.arange(size) % 3, 3, skipped, workers=0)
            actual = torch.cat([batch[0] for batch in loader])
            self.assertTrue(torch.equal(actual, expected[skipped * 64:]))

    def test_real_training_epoch(self):
        labels = np.load('input/split95-seed45/train_labels.npy', mmap_mode='r')
        config = json.loads(Path('similar_pairs.json').read_text())
        pairs = [p['ids'] for p in config['pairs']]
        allowed = {tuple(p) for p in pairs}
        total, targeted, counts = 0, 0, np.zeros(345, dtype=np.int64)
        seen = np.zeros(len(labels), dtype=bool)
        for batch in SimilarBatchSampler(labels, pairs, epoch=1):
            self.assertEqual(len(set(batch)), len(batch))
            self.assertTrue(all(0 <= k < len(labels) for k in batch))
            if len(batch) == 64:
                classes, sizes = np.unique(labels[batch[:16]], return_counts=True)
                self.assertIn(tuple(classes), allowed)
                self.assertEqual(sizes.tolist(), [8, 8])
                targeted += 16
            total += len(batch)
            counts += np.bincount(labels[batch], minlength=345)
            seen[batch] = True
        self.assertEqual(total, 655500)
        self.assertTrue((counts > 0).all())
        self.assertAlmostEqual(targeted / total, .25, places=4)
        print(json.dumps({'examples': total, 'targeted_in_full_batches': targeted,
                          'classes_seen': int((counts > 0).sum()),
                          'unique_images': int(seen.sum()),
                          'min_class_samples': int(counts.min()), 'max_class_samples': int(counts.max())}))

    def test_pair_source_excludes_retained_val(self):
        split = Path('input/split95-seed45')
        train = np.load(split / 'train_indices.npy')
        val = np.load(split / 'val_indices.npy')
        self.assertEqual(np.intersect1d(train, val).size, 0)
        config = json.loads(Path('similar_pairs.json').read_text())
        self.assertEqual(config['examples'], 69000)
        self.assertEqual(hashlib.sha256((split / 'manifest.json').read_bytes()).hexdigest(), config['split_sha256'])
        self.assertEqual(hashlib.sha256((split / 'train_indices.npy').read_bytes()).hexdigest(), config['train_indices_sha256'])
        old_size = len(np.load('input/train_labels.npy', mmap_mode='r'))
        moved = train[train >= old_size] - old_size
        labels = np.load('input/val_labels.npy')[moved].astype(np.int64)
        self.assertEqual(len(moved), 69000)
        errors = {}
        for name, path in zip(['vit_plain', 'vit_flip', 'efficientnet_flip'], config['probability_sha256']):
            predictions = np.load(path, mmap_mode='r')[moved].argmax(1)
            matrix = np.bincount(labels * 345 + predictions, minlength=345 ** 2).reshape(345, 345)
            errors[name] = matrix + matrix.T
        mask = (errors['vit_plain'] >= 14) & (errors['vit_flip'] >= 10) & (errors['efficientnet_flip'] >= 10)
        expected = set(zip(*np.where(np.triu(mask, 1))))
        self.assertEqual(expected, {tuple(p['ids']) for p in config['pairs']})
        for pair in config['pairs']:
            a, b = pair['ids']
            self.assertEqual(pair['errors'], {k: int(v[a, b]) for k, v in errors.items()})


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
