import json
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from pretrained_vit import classification_loss
from split95 import sha256


class ClassWeightsTest(unittest.TestCase):
    def test_loss_and_gradient_weights(self):
        logits = torch.zeros(3, 3, requires_grad=True)
        targets = torch.arange(3)
        weights = torch.tensor([.5, 1., .5])
        loss = classification_loss(logits, targets, weights)
        smoothed = F.one_hot(targets, 3) * .95 + .05 / 3
        expected = (-(smoothed * logits.log_softmax(1)).sum(1) * weights).sum() / weights.sum()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        norms = logits.grad.norm(dim=1)
        torch.testing.assert_close(norms[1], 2 * norms[0])
        torch.testing.assert_close(norms[1], 2 * norms[2])

    def test_unweighted_and_all_ones(self):
        logits = torch.randn(9, 7)
        targets = torch.arange(9) % 7
        ordinary = F.cross_entropy(logits, targets, label_smoothing=.05)
        torch.testing.assert_close(classification_loss(logits, targets), ordinary, rtol=0, atol=0)
        torch.testing.assert_close(classification_loss(logits, targets, torch.ones(7)), ordinary)

    def test_frozen_classes(self):
        config = json.loads(Path('class_loss_weights.json').read_text())
        assert config['classes_sha256'] == sha256(Path('input/classes.npy'))
        rows = json.loads(Path('output/class-distribution95-20261015/classes.json').read_text())
        expected = np.ones(345)
        for row in rows:
            if row['correct'] < 50 or row['correct'] >= 90:
                expected[row['class_id']] = .5
        np.testing.assert_array_equal(config['weights'], expected)
        self.assertEqual(len(config['hard']), 11)
        self.assertEqual(len(config['easy']), 56)
        self.assertEqual(int((expected == 1).sum()), 278)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
