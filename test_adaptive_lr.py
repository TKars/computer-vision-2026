import contextlib
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import pretrained_vit as trainer
from similar_batches import training_loader


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(784, 345)
        self.pretrained_cfg = {'architecture': 'vit_small_patch16_224', 'mean': [.5], 'std': [.5]}

    def forward(self, images):
        return self.head(images)


class AdaptiveLRTest(unittest.TestCase):
    def test_checkpoint_and_resume(self):
        self.check_checkpoint_and_resume()

    def test_weighted_checkpoint_and_resume(self):
        self.check_checkpoint_and_resume('class_loss_weights.json')

    def check_checkpoint_and_resume(self, class_weights=None):
        arrays = {name: np.zeros((256 if name == 'train_images' else 64, 784), dtype=np.uint8)
                  for name in ['train_images', 'val_images', 'test_images']}
        arrays.update(train_labels=np.zeros(256, dtype=np.int64), val_labels=np.zeros(64, dtype=np.int64))
        dataset = TensorDataset(torch.zeros(64, 1, 28, 28), torch.zeros(64, dtype=torch.long))
        loaders = {name: DataLoader(dataset, batch_size=64) for name in ['train', 'val', 'test']}
        training = TensorDataset(torch.zeros(256, 1, 28, 28), torch.zeros(256, dtype=torch.long))
        loaders['train'] = DataLoader(training, batch_size=64)
        probabilities = np.eye(345, dtype=np.float32)[np.zeros(64, dtype=int)]
        original_load = torch.load
        original_step = torch.optim.AdamW.step
        rates = []

        def step(optimizer, *args, **kwargs):
            rates.append(optimizer.param_groups[0]['lr'])
            return original_step(optimizer, *args, **kwargs)

        def load(path, **kwargs):
            if str(path) in ['input/baseline.pth', 'output/regularized-fixed-20261014/model.pth']:
                return {}
            return original_load(path, **kwargs)

        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            for target, replacement in [
                ('setup', lambda *args: (arrays, loaders)),
                ('make_network', lambda *args, **kwargs: Tiny()),
                ('make_model', lambda *args: Mock()),
                ('prepare', lambda x, **kwargs: x.float().reshape(-1, 784)),
                ('predict', lambda *args, **kwargs: probabilities),
                ('resnet_predict', lambda *args, **kwargs: probabilities),
                ('training_loader', lambda *args: training_loader(*args, workers=0))]:
                stack.enter_context(patch.object(trainer, target, replacement))
            stack.enter_context(patch.object(torch.Tensor, 'cuda', lambda x, **kwargs: x))
            stack.enter_context(patch.object(torch.cuda, 'get_device_name', lambda *args: 'CPU test'))
            stack.enter_context(patch.object(torch.cuda, 'get_rng_state_all', lambda: []))
            stack.enter_context(patch.object(torch.cuda, 'set_rng_state_all', lambda *args: None))
            stack.enter_context(patch.object(torch, 'load', load))
            stack.enter_context(patch.object(torch.optim.AdamW, 'step', step))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            first, second = Path(directory) / 'first', Path(directory) / 'resumed'
            trainer.run(first, time.time() + 120, smoke=True, extra_epochs=3, schedule='plateau', class_weights=class_weights,
                        data_dir='input/split95-seed45' if class_weights else None)
            a = original_load(first / 'last.pth', map_location='cpu', weights_only=False)
            self.assertEqual(a['scheduler']['last_epoch'], 3)
            self.assertAlmostEqual(a['optimizer']['param_groups'][0]['lr'], .00005)
            self.assertAlmostEqual(a['history'][-1]['next_lr'], .00005)
            self.assertEqual(rates[:2], [.00005, .0001])
            trainer.run(second, time.time() + 120, smoke=True, resume=first, extra_epochs=1)
            b = original_load(second / 'last.pth', map_location='cpu', weights_only=False)
            self.assertEqual(b['config']['schedule'], 'plateau')
            self.assertEqual(b['steps'], 16)
            self.assertEqual(b['scheduler']['last_epoch'], 4)
            self.assertEqual(b['scheduler']['num_bad_epochs'], 1)
            self.assertEqual(b['optimizer']['param_groups'][0]['lr'], a['optimizer']['param_groups'][0]['lr'])
            self.assertTrue(all(state['step'].item() == 16 for state in b['optimizer']['state'].values()))
            if class_weights:
                self.assertEqual((first / 'class_weights.json').read_bytes(), (second / 'class_weights.json').read_bytes())
                self.assertEqual(a['config']['class_weights_sha256'], b['config']['class_weights_sha256'])
                self.assertEqual(b['config']['sampling'], 'random permutation')
            config = json.loads((second / 'config.json').read_text())
            optimizer = torch.optim.AdamW([{'params': [nn.Parameter(torch.zeros(1))], 'lr': .0001},
                                           {'params': [nn.Parameter(torch.zeros(1))], 'lr': .001}])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config['plateau'])
            for _ in range(40):
                scheduler.step(1.0)
            self.assertEqual([group['lr'] for group in optimizer.param_groups], [1e-6, 1e-5])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
