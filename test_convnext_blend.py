import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import blend_pair95
import ensemble_submission95
from split95 import sha256


class ConvNeXtBlendTest(unittest.TestCase):
    def test_submission_roundtrip_and_modified_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            path = output / 'probabilities.npy'
            probs = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(172500, 345))
            eye = np.eye(345, dtype=np.float32)
            for start in range(0, 172500, 4096):
                stop = min(start + 4096, 172500)
                probs[start:stop] = eye[np.arange(start, stop) % 345]
            probs.flush()
            del probs
            recipe = {'mode': 'plain', 'temperature': 4, 'vit_weight': 0., 'efficientnet_weight': 0., 'convnext_weight': 1.}
            (output / 'summary.json').write_text(json.dumps({'selected': recipe}))
            (output / 'submission_summary.json').write_text(json.dumps({'recipe': recipe,
                'probabilities': {'convnext': str(path)}, 'sha256': {str(path): sha256(path)}, 'model_weights_sha256': {}}))
            submission = pd.DataFrame({'id': np.arange(172500), 'label': np.arange(172500) % 345})
            submission.to_csv(output / 'submission.csv', index=False)
            ensemble_submission95.verify(output)
            submission.loc[0, 'label'] = 1
            submission.to_csv(output / 'submission.csv', index=False)
            with self.assertRaises(AssertionError):
                ensemble_submission95.verify(output)

    def test_third_model_and_pair_boundaries(self):
        rng = np.random.default_rng(2)
        arrays = {mode: [rng.dirichlet(np.ones(345), 12).astype(np.float32) for i in range(3)]
                  for mode in ('plain', 'flip')}
        labels = (.35 * arrays['flip'][0] + .5 * arrays['flip'][1] + .15 * arrays['flip'][2]).argmax(1)
        with tempfile.TemporaryDirectory() as directory, patch.object(blend_pair95, 'inputs', return_value=(labels, arrays, {})):
            output = Path(directory) / 'triple'
            blend_pair95.run(Path('vit'), Path('efficientnet'), output, convnext=Path('convnext'))
            blend_pair95.verify(output)
            summary = json.loads((output / 'summary.json').read_text())
            selected = summary['selected']
            self.assertEqual(selected['correct'], len(labels))
            self.assertEqual(summary['candidates'], 1392)
            self.assertIn('convnext_weight', selected)
            self.assertNotIn('similar_vit_weight', selected)
            self.assertAlmostEqual(sum(selected[key] for key in ('vit_weight', 'efficientnet_weight', 'convnext_weight')), 1)
            table = pd.read_csv(output / 'scores.csv')
            self.assertEqual(int((table['convnext_weight'] == 0).sum()), 126)
            probs = [p ** (1 / selected['temperature']) for p in arrays[selected['mode']]]
            probs = [p / p.sum(1, keepdims=True) for p in probs]
            mixed = sum(selected[key] * p for key, p in zip(('vit_weight', 'efficientnet_weight', 'convnext_weight'), probs))
            np.testing.assert_array_equal(mixed.argmax(1), np.load(output / 'val_predictions.npy'))


if __name__ == '__main__':
    unittest.main()
