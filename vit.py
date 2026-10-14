import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from timm.models.vision_transformer import VisionTransformer
from timm.utils import ModelEmaV3
from torch import nn
from torch.utils.data import DataLoader

from train import Drawings, accuracy, make_model, predict, prepare, set_seed


def make_vit():
    return VisionTransformer(img_size=28, patch_size=4, in_chans=1, num_classes=345,
                             embed_dim=192, depth=6, num_heads=3, global_pool='avg',
                             drop_path_rate=0.1).cuda().to(memory_format=torch.channels_last)


def setup(smoke=False):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    arrays = {name: np.load(f'input/{name}.npy', mmap_mode='r')
              for name in ('train_images', 'train_labels', 'val_images', 'val_labels', 'test_images')}
    if smoke:
        arrays = {name: value[:512 if name.startswith('train') else 64]
                  for name, value in arrays.items()}
    loaders = {split: DataLoader(Drawings(arrays[f'{split}_images'], arrays.get(f'{split}_labels')),
                                batch_size=512, shuffle=split == 'train', num_workers=4,
                                persistent_workers=True, pin_memory=True)
               for split in ('train', 'val', 'test')}
    return arrays, loaders


def run(output, deadline, smoke=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / 'history.csv').exists(), 'Укажите новую папку для запуска'
    arrays, loaders = setup(smoke)
    reference = 'output/regularized-fixed-20261014/model.pth'
    weights = torch.load('input/baseline.pth', map_location='cpu', weights_only=True)
    resnet = make_model(3, weights)
    resnet.load_state_dict(torch.load(reference, map_location='cpu', weights_only=True))
    labels = arrays['val_labels']
    scores = []
    for flip in (False, True):
        probs = predict(resnet, loaders['val'], flip=flip)
        correct = int((probs.argmax(1) == labels).sum())
        scores.append({'model': 'resnet_flip' if flip else 'resnet',
                       'correct': correct, 'accuracy': correct / len(labels)})
    del resnet, weights, probs
    set_seed(44)
    model = make_vit()
    config = {'architecture': {'input': 28, 'patch': 4, 'embed_dim': 192, 'depth': 6,
                               'heads': 3, 'pool': 'avg', 'drop_path': 0.1},
              'initialization': 'random', 'pretrained': False, 'seed': 44,
              'parameters': sum(p.numel() for p in model.parameters()),
              'lr': 0.001, 'weight_decay': 0.05, 'mixup_alpha': 0.2,
              'label_smoothing': 0.05, 'ema_decay': 0.999, 'warmup_epochs': 5,
              'batch_size': 512, 'deadline_unix': deadline, 'smoke': smoke,
              'train_examples': len(arrays['train_images']), 'val_examples': len(labels),
              'resnet_checkpoint': reference, 'torch': torch.__version__,
              'gpu': torch.cuda.get_device_name(0)}
    (output / 'source.py').write_text(Path(__file__).read_text())
    (output / 'train_source.py').write_text(Path('train.py').read_text())
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    print(json.dumps(config), flush=True)
    named = list(model.named_parameters())
    decay = [p for n, p in named if p.ndim > 1 and n not in ('pos_embed', 'cls_token')]
    other = [p for n, p in named if p.ndim <= 1 or n in ('pos_embed', 'cls_token')]
    optimizer = torch.optim.AdamW([{'params': decay}, {'params': other, 'weight_decay': 0}],
                                 lr=0.001, weight_decay=0.05)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    ema = ModelEmaV3(model, decay=0.999)
    training_start = time.time()
    training_deadline = deadline - (20 if smoke else 600)
    epoch, steps, best, history = 0, 0, -1.0, []
    epochs = 1 if smoke else 200
    warmup_steps = len(loaders['train']) * (1 if smoke else 5)
    while epoch < epochs and time.time() < training_deadline:
        epoch += 1
        start = time.time()
        model.train()
        loss_sum, correct, count = 0.0, 0, 0
        for images, targets in loaders['train']:
            steps += 1
            progress = max(0, min(1, (steps - warmup_steps) /
                                 max(1, epochs * len(loaders['train']) - warmup_steps)))
            lr = (1e-5 + 0.5 * (0.001 - 1e-5) * (1 + math.cos(math.pi * progress)))
            lr *= min(1, steps / warmup_steps)
            for group in optimizer.param_groups:
                group['lr'] = lr
            images = prepare(images, augment=True)
            targets = targets.cuda(non_blocking=True)
            ratio = float(np.random.beta(0.2, 0.2))
            ratio = max(ratio, 1 - ratio)
            if epoch > epochs - 10 and not smoke:
                ratio = 1.0
            order = torch.randperm(len(images), device='cuda')
            images = ratio * images + (1 - ratio) * images[order]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                logits = model(images)
                loss = ratio * criterion(logits, targets) + (1 - ratio) * criterion(logits, targets[order])
            assert torch.isfinite(loss).item(), 'Loss is not finite'
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            loss_sum += loss.item() * len(targets)
            correct += (logits.detach().argmax(1) == targets).sum().item()
            count += len(targets)
            if time.time() >= training_deadline:
                break
        raw_score = accuracy(model, loaders['val'])
        ema_score = accuracy(ema.module, loaders['val'])
        row = {'epoch': epoch, 'loss': loss_sum / count, 'train_accuracy': correct / count,
               'accuracy': raw_score, 'ema_accuracy': ema_score, 'examples': count,
               'lr': lr, 'seconds': time.time() - start}
        history.append(row)
        pd.DataFrame(history).to_csv(output / 'history.csv', index=False)
        for name, candidate, score in [('raw', model, raw_score), ('ema', ema.module, ema_score)]:
            if score > best:
                best = score
                torch.save(candidate.state_dict(), output / 'vit.pth')
                (output / 'best.json').write_text(json.dumps({'epoch': epoch, 'state': name,
                                                            'accuracy': score}, indent=2))
        print(json.dumps(row), flush=True)
        if epoch == 1 and not smoke:
            epochs = max(6, int((training_deadline - training_start) / row['seconds']))
            config['estimated_epochs'] = epochs
            (output / 'config.json').write_text(json.dumps(config, indent=2))
    assert history, 'Time limit expired before training'
    model.load_state_dict(torch.load(output / 'vit.pth', map_location='cpu', weights_only=True))
    for flip in (False, True):
        probs = predict(model, loaders['val'], flip=flip)
        correct = int((probs.argmax(1) == labels).sum())
        scores.append({'model': 'vit_flip' if flip else 'vit',
                       'correct': correct, 'accuracy': correct / len(labels)})
    selected = max(scores[2:], key=lambda row: row['accuracy'])
    flip = selected['model'] == 'vit_flip'
    probs = predict(model, loaders['test'], flip=flip)
    pd.DataFrame({'id': np.arange(len(probs)), 'label': probs.argmax(1)}).to_csv(output / 'submission.csv', index=False)
    pd.DataFrame(scores).to_csv(output / 'scores.csv', index=False)
    summary = {'selected': selected, 'flip': flip, 'val_examples': len(labels),
               'submission_rows': len(probs), 'resnet_checkpoint': reference,
               'delta_resnet_plain': scores[2]['accuracy'] - scores[0]['accuracy'],
               'delta_resnet_flip': scores[3]['accuracy'] - scores[1]['accuracy'],
               'val_above_6_threshold': len(labels) == 103500 and selected['accuracy'] > .76102,
               'val_above_8_threshold': len(labels) == 103500 and selected['accuracy'] > .77489}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


def verify(output):
    output = Path(output)
    config = json.loads((output / 'config.json').read_text())
    summary = json.loads((output / 'summary.json').read_text())
    arrays, loaders = setup(config['smoke'])
    model = make_vit()
    model.load_state_dict(torch.load(output / 'vit.pth', map_location='cpu', weights_only=True))
    val = predict(model, loaders['val'], flip=summary['flip']).argmax(1)
    test = predict(model, loaders['test'], flip=summary['flip']).argmax(1)
    correct = int((val == arrays['val_labels']).sum())
    submission = pd.read_csv(output / 'submission.csv')
    assert submission.columns.tolist() == ['id', 'label']
    assert submission['id'].tolist() == list(range(len(test)))
    assert submission['label'].between(0, 344).all()
    differences = int((test != submission['label'].to_numpy()).sum())
    result = {'correct': correct, 'val_examples': len(val), 'accuracy': correct / len(val),
              'submission_rows': len(test), 'submission_differences': differences}
    assert result['accuracy'] == summary['selected']['accuracy'], result
    assert differences == 0, result
    (output / 'verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='output/vit-20261014')
    parser.add_argument('--deadline', type=float, default=None)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        verify(args.output)
    else:
        run(args.output, args.deadline or time.time() + (120 if args.smoke else 10800), args.smoke)
