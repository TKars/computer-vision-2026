import numpy as np
import torch
from torch.utils.data import DataLoader


class SimilarBatchSampler:
    def __init__(self, labels, pairs, epoch, start_batch=0, batch_size=64, seed=45):
        self.labels = np.asarray(labels)
        self.pools = {int(k): np.flatnonzero(self.labels == k) for k in np.unique(pairs)}
        self.pairs = [(int(a), int(b)) for a, b in pairs if len(self.pools[a]) and len(self.pools[b])]
        assert self.pairs and all(a != b for a, b in self.pairs), 'Нет пар классов в train'
        self.epoch, self.start_batch = epoch, start_batch
        self.batch_size, self.seed = batch_size, seed
        self.batches = (len(labels) + batch_size - 1) // batch_size
        assert 0 <= start_batch <= self.batches

    def __len__(self):
        return self.batches - self.start_batch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(self.labels))
        cursor = 0
        for step in range(self.batches):
            size = min(self.batch_size, len(self.labels) - step * self.batch_size)
            a, b = self.pairs[rng.integers(len(self.pairs))]
            count = min(size // 8, len(self.pools[a]), len(self.pools[b]))
            batch = np.concatenate([rng.choice(self.pools[k], count, replace=False) for k in (a, b)]).tolist()
            reserved = set(batch)
            while len(batch) < size:
                index = int(order[cursor])
                cursor += 1
                if index not in reserved:
                    batch.append(index)
            if step >= self.start_batch:
                yield batch


def training_loader(dataset, labels, epoch, start_batch=0, pairs=None, workers=4):
    generator = torch.Generator().manual_seed(45 + epoch)
    options = dict(num_workers=workers, persistent_workers=workers > 0,
                   pin_memory=True, generator=generator)
    if pairs is not None:
        sampler = SimilarBatchSampler(labels, pairs, epoch, start_batch)
        return DataLoader(dataset, batch_sampler=sampler, **options)
    order = torch.randperm(len(dataset), generator=generator).tolist()[start_batch * 64:]
    return DataLoader(dataset, batch_size=64, sampler=order, **options)
