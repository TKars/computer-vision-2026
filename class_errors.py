import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def run(predictions, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = np.load('input/classes.npy')
    labels = np.load('input/val_labels.npy')
    images = np.load('input/val_images.npy', mmap_mode='r')
    probs = np.load(predictions, mmap_mode='r')
    assert probs.shape == (len(labels), len(names))
    predicted = probs.argmax(1)
    matrix = np.zeros((len(names), len(names)), dtype=np.int64)
    np.add.at(matrix, (labels, predicted), 1)
    rows, examples, pictures = [], [], []
    for k, name in enumerate(names):
        count, correct = int(matrix[k].sum()), int(matrix[k, k])
        wrong = matrix[k].copy()
        wrong[k] = 0
        confusions = [int(j) for j in np.argsort(wrong)[::-1][:3] if wrong[j]]
        rows.append({'class': str(name), 'class_id': k, 'correct': correct, 'count': count,
                     'accuracy': correct / count,
                     'confusions': [{'predicted': str(names[j]), 'count': int(wrong[j])}
                                    for j in confusions]})
    rows.sort(key=lambda row: (row['accuracy'], row['class_id']))
    table = pd.DataFrame([{**row, 'confusions': json.dumps(row['confusions'])} for row in rows])
    table.to_csv(output / 'per_class.csv', index=False)
    np.save(output / 'confusion.npy', matrix)
    for row in rows[:6]:
        k = row['class_id']
        for confusion in row['confusions'][:2]:
            j = int(np.flatnonzero(names == confusion['predicted'])[0])
            candidates = np.flatnonzero((labels == k) & (predicted == j))
            index = int(candidates[np.argmax(probs[candidates, j])])
            pictures.append(images[index].reshape(28, 28))
            examples.append({'index': index, 'true': row['class'], 'predicted': str(names[j]),
                             'confidence': float(probs[index, j]), 'class_accuracy': row['accuracy']})
    np.save(output / 'examples.npy', np.stack(pictures))
    (output / 'examples.json').write_text(json.dumps(examples, indent=2))
    wrong = matrix.copy()
    np.fill_diagonal(wrong, 0)
    pairs = []
    for index in np.argsort(wrong.ravel())[::-1][:10]:
        k, j = np.unravel_index(index, wrong.shape)
        pairs.append({'true': str(names[k]), 'predicted': str(names[j]), 'count': int(wrong[k, j])})
    summary = {'predictions': str(predictions), 'val_examples': len(labels),
               'accuracy': float((predicted == labels).mean()), 'classes': len(names),
               'examples_per_class': [int(matrix[k].sum()) for k in range(len(names))],
               'worst': rows[:15], 'most_frequent_confusions': pairs}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.predictions, args.output)
