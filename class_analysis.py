import argparse
import hashlib
import itertools
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


SOURCES = {
    'vit_plain': ('output/pretrained-tta-20261015-0926/snapshot/val_plain.npy', 78549),
    'vit_flip': ('output/pretrained-tta-20261015-0926/snapshot/val_flip.npy', 78760),
    'efficientnet_flip': ('output/efficientnet-b3-resume-20261015/val_flip.npy', 78687),
}


def analyse(probs, labels, names, output, prefix):
    n = len(names)
    assert probs.shape == (len(labels), n)
    assert np.isfinite(probs).all() and (probs >= 0).all()
    assert np.allclose(probs.sum(1), 1, atol=1e-5)
    top = np.argsort(-probs, axis=1, kind='stable')[:, :3].copy()
    values = np.take_along_axis(probs, top, axis=1)
    predictions = top[:, 0]
    confusion = np.zeros((n, n), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    counts = np.bincount(labels, minlength=n)
    assert (counts == 300).all()
    means = np.stack([probs[labels == k].mean(0) for k in range(n)])
    gram = probs.T @ probs
    norm = np.sqrt(np.diag(gram))
    cosine = gram / np.maximum(norm[:, None] * norm[None, :], 1e-12)
    top2, near, recoverable = [np.zeros((n, n), dtype=np.int64) for _ in range(3)]
    mass = values[:, :2].sum(1)
    margin = (values[:, 0] - values[:, 1]) / np.maximum(mass, 1e-12)
    a, b = np.sort(top[:, :2], axis=1).T
    relevant = (labels == a) | (labels == b)
    close = relevant & (margin <= .2) & (mass >= .25)
    repairable = relevant & (predictions != labels)
    for matrix, mask in [(top2, relevant), (near, close), (recoverable, repairable)]:
        np.add.at(matrix, (a[mask], b[mask]), 1)
    rows = []
    for k, name in enumerate(names):
        wrong = confusion[k].copy()
        wrong[k] = 0
        neighbors = [int(i) for i in np.argsort(-wrong)[:3] if wrong[i]]
        rows.append({'class_id': k, 'class': name, 'correct': int(confusion[k, k]), 'count': int(counts[k]),
                     'accuracy': confusion[k, k] / counts[k], 'errors': int(wrong.sum()),
                     'precision': confusion[k, k] / max(1, confusion[:, k].sum()),
                     'top2_recall': float((top[labels == k, :2] == k).any(1).mean()),
                     'top3_recall': float((top[labels == k] == k).any(1).mean()),
                     'confusions': '; '.join(f'{names[i]}: {wrong[i]}' for i in neighbors)})
    i, j = np.triu_indices(n, 1)
    pairs = pd.DataFrame({'id_a': i, 'id_b': j, 'class_a': names[i], 'class_b': names[j],
                          'a_to_b': confusion[i, j], 'b_to_a': confusion[j, i],
                          'errors': confusion[i, j] + confusion[j, i],
                          'top2_on_pair': top2[i, j], 'near_ties_on_pair': near[i, j],
                          'recoverable_top2_errors': recoverable[i, j], 'score_cosine': cosine[i, j],
                          'mean_cross_probability': (means[i, j] + means[j, i]) / 2})
    assert pairs['errors'].sum() == (predictions != labels).sum()
    pd.DataFrame(rows).to_csv(output / f'{prefix}_classes.csv', index=False)
    pairs.to_csv(output / f'{prefix}_pairs.csv', index=False)
    np.save(output / f'{prefix}_confusion.npy', confusion)
    summary = {'correct': int((predictions == labels).sum()), 'accuracy': float((predictions == labels).mean()),
               'top2_oracle': float((top[:, :2] == labels[:, None]).any(1).mean()),
               'top3_oracle': float((top == labels[:, None]).any(1).mean()),
               'near_ties_on_true_pair': int(close.sum()), 'near_tie_errors': int((close & (predictions != labels)).sum()),
               'wrong_probability_ge_0_8': int(((values[:, 0] >= .8) & (predictions != labels)).sum()),
               'errors': int((predictions != labels).sum())}
    return pd.DataFrame(rows), pairs, summary, predictions


def pictures(probs, labels, names, pairs, output):
    images = np.load('input/val_images.npy', mmap_mode='r')
    predictions = probs.argmax(1)
    canvas = Image.new('RGB', (880, 1600), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    records = []
    for row, pair in enumerate(pairs.head(10).itertuples()):
        for direction, (a, b) in enumerate([(pair.id_a, pair.id_b), (pair.id_b, pair.id_a)]):
            candidates = np.flatnonzero((labels == a) & (predictions == b))
            selected = candidates[np.argsort(-probs[candidates, b])[:2]]
            for item, index in enumerate(selected):
                x, y = (direction * 2 + item) * 220, row * 160
                image = Image.fromarray(255 - images[index].reshape(28, 28)).resize((84, 84), Image.Resampling.NEAREST)
                canvas.paste(image, (x + 65, y + 3))
                draw.text((x + 5, y + 90), f'True: {names[a]}', fill='black', font=font)
                draw.text((x + 5, y + 108), f'Pred: {names[b]}', fill='black', font=font)
                draw.text((x + 5, y + 126), f'p={probs[index, b]:.3f}; id={index}', fill='black', font=font)
                records.append({'index': int(index), 'true': names[a], 'predicted': names[b], 'p': float(probs[index, b])})
    canvas.save(output / 'confusion_examples.png')
    (output / 'confusion_examples.json').write_text(json.dumps(records, indent=2))


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    names = np.load('input/classes.npy')
    labels = np.load('input/val_labels.npy')
    assert len(names) == 345 and len(labels) == 103500
    annotations = json.loads(Path('class_groups.json').read_text())
    reviewed = [name for group in annotations['review_groups'].values() for name in group]
    assert len(reviewed) == len(set(reviewed)) == len(names) and set(reviewed) == set(names)
    group = {name: key for key, values in annotations['review_groups'].items() for name in values}
    candidates = {}
    for entry in annotations['candidate_sets']:
        assert set(entry['classes']) <= set(names)
        for a, b in itertools.combinations(entry['classes'], 2):
            candidates.setdefault(tuple(sorted((a, b))), []).append(entry['name'])
    summaries, classes, pairs, hashes = {}, None, None, {}
    for prefix, (path, expected) in SOURCES.items():
        probs = np.load(path, mmap_mode='r')
        c, p, summary, predictions = analyse(probs, labels, names, output, prefix)
        assert summary['correct'] == expected, (prefix, summary)
        summaries[prefix] = summary
        with open(path, 'rb') as stream:
            hashes[path] = hashlib.file_digest(stream, 'sha256').hexdigest()
        if classes is None:
            classes = c.copy()
            classes['review_group'] = [group[name] for name in names]
            pairs = p.copy()
        else:
            classes[prefix + '_correct'] = c['correct']
            classes[prefix + '_accuracy'] = c['accuracy']
            for column in ('errors', 'recoverable_top2_errors', 'near_ties_on_pair', 'score_cosine'):
                pairs[prefix + '_' + column] = p[column]
        print(prefix, json.dumps(summary), flush=True)
    ensemble_path = 'output/efficientnet-blend-refined-20261015/val_predictions.npy'
    ensemble = np.load(ensemble_path)
    assert ensemble.shape == labels.shape and int((ensemble == labels).sum()) == 80199
    ensemble_matrix = np.zeros((345, 345), dtype=np.int64)
    np.add.at(ensemble_matrix, (labels, ensemble), 1)
    classes['ensemble_correct'] = ensemble_matrix.diagonal()
    i, j = pairs['id_a'].to_numpy(), pairs['id_b'].to_numpy()
    pairs['ensemble_errors'] = ensemble_matrix[i, j] + ensemble_matrix[j, i]
    pairs['shared_review_group'] = [group[a] == group[b] for a, b in zip(pairs['class_a'], pairs['class_b'])]
    pairs['candidate_sets'] = ['; '.join(candidates.get(tuple(sorted((a, b))), []))
                                for a, b in zip(pairs['class_a'], pairs['class_b'])]
    classes.sort_values('accuracy').to_csv(output / 'all_classes.csv', index=False)
    pairs.sort_values('errors', ascending=False).to_csv(output / 'all_pairs.csv', index=False)
    pairs.sort_values('score_cosine', ascending=False).to_csv(output / 'score_similarity.csv', index=False)
    pairs.sort_values('near_ties_on_pair', ascending=False).to_csv(output / 'close_scores.csv', index=False)
    graph = nx.Graph()
    for row in pairs.itertuples():
        if row.errors >= 20 and row.vit_flip_errors >= 15 and row.efficientnet_flip_errors >= 15:
            graph.add_edge(row.id_a, row.id_b, weight=row.errors)
    communities = []
    for community in nx.community.greedy_modularity_communities(graph, weight='weight'):
        edges = pairs[pairs['id_a'].isin(community) & pairs['id_b'].isin(community)]
        communities.append({'classes': [str(names[k]) for k in sorted(community)], 'size': len(community),
                            'vit_errors': int(edges['errors'].sum()),
                            'vit_flip_errors': int(edges['vit_flip_errors'].sum()),
                            'efficientnet_errors': int(edges['efficientnet_flip_errors'].sum()),
                            'ensemble_errors': int(edges['ensemble_errors'].sum())})
    communities.sort(key=lambda c: c['vit_errors'], reverse=True)
    errors = summaries['vit_plain']['errors']
    worst = classes.sort_values('errors', ascending=False)
    recoverable = int(pairs['recoverable_top2_errors'].sum())
    assert recoverable == round(summaries['vit_plain']['top2_oracle'] * len(labels)) - summaries['vit_plain']['correct']
    summary = {'models': summaries, 'class_names_reviewed': len(reviewed), 'pairs_checked': len(pairs),
               'val_examples': len(labels), 'examples_per_class': 300, 'communities': communities,
               'worst_class_errors': {str(k): int(worst.head(k)['errors'].sum()) for k in (10, 20, 30, 50)},
               'total_errors': errors, 'errors_inside_candidate_sets': int(pairs.loc[pairs['candidate_sets'] != '', 'errors'].sum()),
               'recoverable_top2_errors': recoverable, 'extra_correct_needed_for_80_plain': 82800 - summaries['vit_plain']['correct'],
               'near_tie_definition': 'top two contain true class; (p1-p2)/(p1+p2)<=0.2; p1+p2>=0.25',
               'score_similarity': 'cosine between probability columns on all 103500 examples; no text embeddings',
               'community_rule': 'edge if plain ViT errors>=20 and flip ViT/B3 errors>=15 each; weighted greedy modularity',
               'probability_sha256': hashes, 'validation_only': True}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    (output / 'source.py').write_text(Path(__file__).read_text())
    (output / 'class_groups.json').write_text(Path('class_groups.json').read_text())
    primary = pairs.sort_values('errors', ascending=False)
    pictures(np.load(SOURCES['vit_plain'][0], mmap_mode='r'), labels, names, primary, output)
    print(json.dumps(summary), flush=True)
    print(classes.sort_values('accuracy').head(20).to_string(index=False), flush=True)
    print(primary.head(25).to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    run(parser.parse_args().output)
