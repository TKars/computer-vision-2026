import argparse
import json
import math
import hashlib
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from timm.utils import ModelEmaV3
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from train import Drawings, make_model, predict as resnet_predict, prepare as drawing_prepare, set_seed


NAME = 'vit_small_patch16_224.augreg_in21k_ft_in1k'


def make_vit(pretrained=False):
    return timm.create_model(NAME, pretrained=pretrained, in_chans=1, num_classes=345).cuda()


def setup(smoke=False):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    arrays = {name: np.load(f'input/{name}.npy', mmap_mode='r')
              for name in ('train_images', 'train_labels', 'val_images', 'val_labels', 'test_images')}
    if smoke:
        arrays = {name: value[:64] for name, value in arrays.items()}
    loaders = {split: DataLoader(Drawings(arrays[f'{split}_images'], arrays.get(f'{split}_labels')),
                                batch_size=64, shuffle=split == 'train', num_workers=4,
                                persistent_workers=True, pin_memory=True)
               for split in ('train', 'val', 'test')}
    return arrays, loaders


def prepare(images, augment=False):
    images = drawing_prepare(images, augment=augment).neg()
    return F.interpolate(images, size=(224, 224), mode='bicubic', align_corners=False).clamp_(-1, 1)


@torch.inference_mode()
def predict(model, loader, flip=False):
    model.eval()
    probabilities = []
    for batch in loader:
        images = prepare(batch[0] if isinstance(batch, (tuple, list)) else batch)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(images)
            mirrored = model(images.flip(-1)) if flip else None
        probs = logits.float().softmax(1)
        if flip:
            probs = (probs + mirrored.float().softmax(1)) / 2
        probabilities.append(probs.cpu().numpy())
    return np.concatenate(probabilities)


def run(output, deadline, smoke=False, resume=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / 'config.json').exists(), 'Укажите новую папку для запуска'
    arrays, loaders = setup(smoke)
    set_seed(45)
    source = Path(resume) if resume else None
    if source and source.is_dir():
        source = source / ('last.pth' if (source / 'last.pth').exists() else 'vit.pth')
    saved = torch.load(source, map_location='cpu', weights_only=False) if source else None
    full_resume = saved is not None and 'optimizer' in saved
    parent = json.loads((source.parent / 'config.json').read_text()) if source else {}
    assert not source or parent['smoke'] == smoke, 'Проверочный checkpoint нельзя использовать для полного запуска'
    model = make_vit(pretrained=not bool(source))
    if source:
        model.load_state_dict(saved['model'] if full_resume else saved)
    config = {'model': NAME, 'pretrained': True, 'pretrained_config': model.pretrained_cfg,
              'input': 224, 'patch': 16, 'channels': 1, 'inverted': True,
              'resize': 'bicubic, align_corners=False, clamp [-1,1]', 'normalization': [0.5, 0.5],
              'parameters': sum(p.numel() for p in model.parameters()), 'batch_size': 64,
              'optimizer': 'AdamW', 'lr': 0.00003 if source else 0.0001,
              'head_lr': 0.0003 if source else 0.001, 'weight_decay': 0.01,
              'label_smoothing': 0.05, 'ema_decay': 0.99, 'precision': 'bfloat16 autocast',
              'seed': 45, 'deadline_unix': deadline, 'smoke': smoke,
              'train_examples': len(arrays['train_images']), 'val_examples': len(arrays['val_labels']),
              'torch': torch.__version__, 'timm': timm.__version__, 'gpu': torch.cuda.get_device_name(0)}
    if source:
        with source.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        config.update(resume=str(source), resume_sha256=digest, optimizer_restored=full_resume)
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    (output / 'source.py').write_text(Path(__file__).read_text())
    (output / 'train_source.py').write_text(Path('train.py').read_text())
    print(json.dumps(config), flush=True)
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if not n.startswith('head.')], 'scale': 1},
        {'params': model.head.parameters(), 'scale': 10, 'lr': config['head_lr']}], lr=config['lr'], weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    ema = ModelEmaV3(model, decay=0.99)
    history, best, steps = [], -1.0, 0
    start_epoch, start_batch, metrics = 1, 0, (0.0, 0, 0)
    if source:
        best_info = json.loads((source.parent / 'best.json').read_text())
        best = best_info['accuracy']
        shutil.copy2(source.parent / 'vit.pth', output / 'vit.pth')
        shutil.copy2(source.parent / 'best.json', output / 'best.json')
        if full_resume:
            optimizer.load_state_dict(saved['optimizer'])
            ema.module.load_state_dict(saved['ema'])
            history, steps = saved['history'], saved['steps']
            start_epoch, start_batch, metrics = saved['epoch'], saved['batch'], saved['metrics']
            if start_batch == len(loaders['train']):
                start_epoch, start_batch, metrics = start_epoch + 1, 0, (0.0, 0, 0)
            else:
                history = [row for row in history if row['epoch'] < start_epoch]
            random.setstate(saved['python_rng'])
            np.random.set_state(saved['numpy_rng'])
            torch.set_rng_state(saved['torch_rng'])
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
            config['lr'] = optimizer.param_groups[0]['lr']
            config['head_lr'] = optimizer.param_groups[1]['lr']
        else:
            previous = pd.read_csv(source.parent / 'history.csv')
            history = previous[previous['epoch'] <= best_info['epoch']].to_dict('records')
            start_epoch = best_info['epoch'] + 1
            steps = best_info['epoch'] * len(loaders['train'])
            initial = float((predict(model, loaders['val']).argmax(1) == arrays['val_labels']).mean())
            assert abs(initial - best) < 1e-7, (initial, best)
            config['initial_accuracy'] = initial
        pd.DataFrame(history).to_csv(output / 'history.csv', index=False)
        config.update(start_epoch=start_epoch, start_batch=start_batch)
        (output / 'config.json').write_text(json.dumps(config, indent=2))
        print(json.dumps({'resumed_epoch': start_epoch, 'resumed_batch': start_batch,
                          'optimizer_restored': full_resume, 'best_accuracy': best}), flush=True)
    del saved
    base_epoch, base_steps = start_epoch, steps
    epochs = 1 if smoke else 10
    epochs += start_epoch - 1
    training_deadline = deadline - (5 if smoke else 900)
    training_start = time.time()
    warmup = 0 if full_resume else max(1, len(loaders['train']) // 2)
    checkpoint_time = time.monotonic()

    def checkpoint(epoch, batch, metrics):
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'ema': ema.module.state_dict(), 'epoch': epoch, 'batch': batch,
                 'steps': steps, 'metrics': metrics, 'history': history, 'config': config,
                 'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}
        with (output / 'last.tmp').open('wb') as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        (output / 'last.tmp').replace(output / 'last.pth')
        (output / 'checkpoint.json').write_text(json.dumps({'epoch': epoch, 'batch': batch,
                                                           'steps': steps, 'saved_at': time.time()}))

    checkpoint(start_epoch, start_batch, metrics)
    for epoch in range(start_epoch, start_epoch + 100):
        if epoch > epochs or time.time() >= training_deadline:
            break
        start = time.time()
        model.train()
        loss_sum, correct, count = metrics if epoch == start_epoch else (0.0, 0, 0)
        skipped = start_batch if epoch == start_epoch else 0
        generator = torch.Generator().manual_seed(45 + epoch)
        order = torch.randperm(len(arrays['train_images']), generator=generator).tolist()[skipped * 64:]
        loader = DataLoader(loaders['train'].dataset, batch_size=64, sampler=order,
                            num_workers=4, persistent_workers=True, pin_memory=True,
                            generator=generator)
        for batch, (images, targets) in enumerate(loader, skipped + 1):
            steps += 1
            local_steps = steps - base_steps
            progress = max(0, min(1, (local_steps - warmup) / max(1, (epochs - base_epoch + 1) * len(loaders['train']) - warmup)))
            lr = (1e-6 + 0.5 * (config['lr'] - 1e-6) * (1 + math.cos(math.pi * progress)))
            lr *= min(1, local_steps / warmup) if warmup else 1
            for group in optimizer.param_groups:
                group['lr'] = lr * group['scale']
            images = prepare(images, augment=True)
            targets = targets.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(images)
                loss = criterion(logits, targets)
            assert torch.isfinite(loss).item(), 'Loss is not finite'
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)
            loss_sum += loss.item() * len(targets)
            correct += (logits.detach().argmax(1) == targets).sum().item()
            count += len(targets)
            if local_steps == 1:
                print(json.dumps({'first_batch_loss': loss.item(), 'examples': count}), flush=True)
            if time.monotonic() - checkpoint_time >= 60:
                checkpoint(epoch, batch, (loss_sum, correct, count))
                checkpoint_time = time.monotonic()
            if time.time() >= training_deadline:
                break
        raw = float((predict(model, loaders['val']).argmax(1) == arrays['val_labels']).mean())
        averaged = float((predict(ema.module, loaders['val']).argmax(1) == arrays['val_labels']).mean())
        row = {'epoch': epoch, 'loss': loss_sum / count, 'train_accuracy': correct / count,
               'accuracy': raw, 'ema_accuracy': averaged, 'examples': count,
               'lr': lr, 'seconds': time.time() - start}
        history.append(row)
        pd.DataFrame(history).to_csv(output / 'history.tmp', index=False)
        (output / 'history.tmp').replace(output / 'history.csv')
        for state, candidate, score in [('raw', model, raw), ('ema', ema.module, averaged)]:
            if score > best:
                best = score
                torch.save(candidate.state_dict(), output / 'vit.tmp')
                (output / 'vit.tmp').replace(output / 'vit.pth')
                (output / 'best.json').write_text(json.dumps({'epoch': epoch, 'state': state, 'accuracy': score}, indent=2))
        print(json.dumps(row), flush=True)
        if epoch == start_epoch and not smoke:
            epochs = start_epoch - 1 + max(1, min(100, int((training_deadline - training_start) / row['seconds'])))
            config['estimated_epochs'] = epochs
            (output / 'config.json').write_text(json.dumps(config, indent=2))
        checkpoint(epoch, batch, (loss_sum, correct, count))
    assert history, 'Time limit expired before training'
    del optimizer, ema
    model.load_state_dict(torch.load(output / 'vit.pth', map_location='cpu', weights_only=True))
    scores = []
    for flip in (False, True):
        probabilities = predict(model, loaders['val'], flip=flip)
        np.save(output / ('val_flip.npy' if flip else 'val_plain.npy'), probabilities)
        correct = int((probabilities.argmax(1) == arrays['val_labels']).sum())
        scores.append({'model': 'vit_flip' if flip else 'vit', 'correct': correct,
                       'accuracy': correct / len(arrays['val_labels'])})
    selected = max(scores, key=lambda row: row['accuracy'])
    flip = selected['model'] == 'vit_flip'
    test = predict(model, loaders['test'], flip=flip).argmax(1)
    pd.DataFrame({'id': np.arange(len(test)), 'label': test}).to_csv(output / 'submission.csv', index=False)
    del model
    torch.cuda.empty_cache()
    resnet = make_model(3, torch.load('input/baseline.pth', map_location='cpu', weights_only=True))
    resnet.load_state_dict(torch.load('output/regularized-fixed-20261014/model.pth', map_location='cpu', weights_only=True))
    for mirrored in (False, True):
        predictions = resnet_predict(resnet, loaders['val'], flip=mirrored).argmax(1)
        correct = int((predictions == arrays['val_labels']).sum())
        scores.append({'model': 'resnet_flip' if mirrored else 'resnet', 'correct': correct,
                       'accuracy': correct / len(arrays['val_labels'])})
    pd.DataFrame(scores).to_csv(output / 'scores.csv', index=False)
    summary = {'selected': selected, 'flip': flip, 'val_examples': len(arrays['val_labels']),
               'submission_rows': len(test), 'val_above_6_threshold': selected['accuracy'] > .76102,
               'val_above_8_threshold': selected['accuracy'] > .77489}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


def verify(output):
    output = Path(output)
    config = json.loads((output / 'config.json').read_text())
    summary = json.loads((output / 'summary.json').read_text())
    arrays, loaders = setup(config['smoke'])
    model = make_vit()
    model.load_state_dict(torch.load(output / 'vit.pth', map_location='cpu', weights_only=True))
    val = predict(model, loaders['val'], summary['flip']).argmax(1)
    test = predict(model, loaders['test'], summary['flip']).argmax(1)
    submission = pd.read_csv(output / 'submission.csv')
    assert submission.columns.tolist() == ['id', 'label']
    assert submission['id'].tolist() == list(range(len(test)))
    assert submission['label'].between(0, 344).all()
    correct = int((val == arrays['val_labels']).sum())
    result = {'correct': correct, 'val_examples': len(val), 'accuracy': correct / len(val),
              'submission_rows': len(test), 'submission_differences': int((test != submission['label'].to_numpy()).sum())}
    assert result['accuracy'] == summary['selected']['accuracy'], result
    assert result['submission_differences'] == 0, result
    (output / 'verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='output/pretrained-vit-20261015')
    parser.add_argument('--deadline', type=float)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--resume')
    args = parser.parse_args()
    if args.verify:
        verify(args.output)
    else:
        run(args.output, args.deadline or time.time() + (120 if args.smoke else 7200), args.smoke, args.resume)
