import json
import os
import signal
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import convnext


class ConvNeXtTest(unittest.TestCase):
    def test_preprocessing(self):
        images = torch.stack([torch.zeros(1, 28, 28, dtype=torch.uint8),
                              torch.full((1, 28, 28), 255, dtype=torch.uint8)])
        result = convnext.prepare(images)
        self.assertEqual(tuple(result.shape), (2, 1, 128, 128))
        self.assertTrue(torch.equal(result[0], -torch.ones_like(result[0])))
        self.assertTrue(torch.equal(result[1], torch.ones_like(result[1])))
        augmented = convnext.prepare(images, True)
        self.assertTrue(torch.isfinite(augmented).all())
        self.assertTrue(((augmented >= -1) & (augmented <= 1)).all())

    def test_batch_resume_order(self):
        images = np.zeros((600, 784), dtype=np.uint8)
        labels = np.arange(600)
        full = torch.cat([y for x, y in convnext.loader(images, labels, 256, epoch=1)])
        resumed = torch.cat([y for x, y in convnext.loader(images, labels, 256, epoch=1, start=1)])
        torch.testing.assert_close(full[256:], resumed, rtol=0, atol=0)
        self.assertEqual(len(torch.unique(full)), 600)


def gpu_check():
    root = Path('output/convnext-check-20261016')
    original_step = torch.optim.AdamW.step
    def stop_after_step(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        os.kill(os.getpid(), signal.SIGTERM)
        return result
    with patch.object(torch.optim.AdamW, 'step', stop_after_step):
        assert not convnext.run(root / 'interrupted', epochs=2, smoke=True)
    interrupted = torch.load(root / 'interrupted/last.pth', map_location='cpu', weights_only=False)
    assert interrupted['steps'] == 1 and interrupted['batch'] == 1
    assert convnext.run(root / 'resumed', smoke=True, resume=root / 'interrupted')
    convnext.verify(root / 'resumed')
    assert convnext.run(root / 'uninterrupted', epochs=2, smoke=True)
    a = torch.load(root / 'resumed/last.pth', map_location='cpu', weights_only=False)
    b = torch.load(root / 'uninterrupted/last.pth', map_location='cpu', weights_only=False)
    assert a['steps'] == b['steps'] == 4
    differences = {name: max(float((a[name][k] - b[name][k]).abs().max()) for k in a[name])
                   for name in ('model', 'ema')}
    assert max(differences.values()) == 0, differences
    assert a['scheduler'] == b['scheduler']
    assert all(torch.isfinite(v).all() for state in a['optimizer']['state'].values()
               for v in state.values() if isinstance(v, torch.Tensor))
    result = {'steps': a['steps'], 'optimizer_states': len(a['optimizer']['state']),
              'resume_max_difference': differences, 'batch_size': 256,
              'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20, 'passed': True}
    (root / 'checks.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    tests = unittest.main(exit=False)
    assert tests.result.wasSuccessful()
    gpu_check()
