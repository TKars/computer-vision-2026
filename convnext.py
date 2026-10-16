import argparse
import copy
import json
import math
import os
import random
import shutil
import signal
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from split95 import sha256
from train import Drawings, set_seed


MODEL = 'convnext_tiny.in12k_ft_in1k'
DATA = Path('input/split95-seed45')


def setup(smoke=False):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    arrays = {name: np.load(DATA / (name + '.npy'), mmap_mode='r') for name in
              ('train_images', 'train_labels', 'val_images', 'val_labels', 'test_images')}
    if smoke:
        indices = np.random.default_rng(2).choice(len(arrays['train_labels']), 512, replace=False)
        arrays = {name: value[indices] if name.startswith('train_') else value[:64]
                  for name, value in arrays.items()}
    return arrays


def make_model(pretrained=False):
    return timm.create_model(MODEL, pretrained=pretrained, in_chans=1, num_classes=345,
                             drop_path_rate=.3).cuda().to(memory_format=torch.channels_last)


def prepare(images, augment=False):
    x = images.cuda(non_blocking=True).float().div_(255)
    if augment:
        n = len(x)
        angle = (torch.rand(n, device='cuda') - .5) * 2 * math.radians(15)
        scale = 1 + (torch.rand(n, device='cuda') - .5) * .2
        shift = (torch.rand(n, 2, device='cuda') - .5) * .2
        theta = torch.stack([torch.stack([angle.cos() / scale, -angle.sin() / scale, shift[:, 0]], 1),
                             torch.stack([angle.sin() / scale, angle.cos() / scale, shift[:, 1]], 1)], 1)
        x = F.grid_sample(x, F.affine_grid(theta, (n, 1, 128, 128), align_corners=False), align_corners=False)
    else:
        x = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)
    return x.mul_(2).sub_(1).contiguous(memory_format=torch.channels_last)


def loader(images, labels=None, batch_size=256, epoch=None, start=0):
    generator = torch.Generator().manual_seed(2 + (epoch or 0))
    order = torch.randperm(len(images), generator=generator).tolist()[start * batch_size:] if epoch else None
    return DataLoader(Drawings(images, labels), batch_size=batch_size, sampler=order, num_workers=4,
                      persistent_workers=True, pin_memory=True, generator=generator)


@torch.inference_mode()
def predict(model, images, batch_size=256, flip=False):
    model.eval()
    result = []
    for images in loader(images, batch_size=batch_size):
        x = prepare(images)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            raw = model(x).float().softmax(1)
            probs = (raw + model(x.flip(-1)).float().softmax(1)) / 2 if flip else raw
        result.append(probs.cpu().numpy())
    return np.concatenate(result)


def finish(model, arrays, output, batch_size):
    model.load_state_dict(torch.load(output / 'model.pth', map_location='cpu', weights_only=True))
    scores = []
    for flip in (False, True):
        mode = 'flip' if flip else 'plain'
        val = predict(model, arrays['val_images'], batch_size, flip)
        np.save(output / f'val_{mode}.npy', val)
        correct = int((val.argmax(1) == arrays['val_labels']).sum())
        scores.append({'model': 'convnext_tiny_' + mode, 'correct': correct, 'accuracy': correct / len(val)})
        np.save(output / f'test_{mode}.npy', predict(model, arrays['test_images'], batch_size, flip))
        print(json.dumps(scores[-1]), flush=True)
    selected = max(scores, key=lambda r: r['correct'])
    assert scores[0]['accuracy'] == json.loads((output / 'best.json').read_text())['accuracy']
    mode = 'flip' if selected['model'].endswith('flip') else 'plain'
    test = np.load(output / f'test_{mode}.npy', mmap_mode='r').argmax(1)
    pd.DataFrame({'id': np.arange(len(test)), 'label': test}).to_csv(output / 'submission.csv', index=False)
    pd.DataFrame(scores).to_csv(output / 'scores.csv', index=False)
    (output / 'summary.json').write_text(json.dumps({'selected': selected, 'flip': mode == 'flip',
        'val_examples': len(arrays['val_labels']), 'submission_rows': len(test)}, indent=2))


def run(output, epochs=20, batch_size=256, deadline=None, smoke=False, resume=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / 'config.json').exists(), 'Укажите новую папку для запуска'
    arrays = setup(smoke)
    set_seed(2)
    parent = Path(resume) if resume else None
    saved = torch.load(parent / 'last.pth', map_location='cpu', weights_only=False) if parent else None
    config = saved['config'].copy() if saved else {'model': MODEL, 'input': 128, 'channels': 1,
        'normalization': {'mean': .5, 'std': .5}, 'inverted': False, 'seed': 2,
        'batch_size': batch_size, 'epochs': epochs, 'lr': .0005, 'weight_decay': .05,
        'drop_path': .3, 'mixup_alpha': .2, 'ema_decay': .9995, 'ema_warmup': '(1+steps)/(10+steps)',
        'schedule': 'OneCycleLR', 'pct_start': .1, 'label_smoothing': .1,
        'precision': 'bfloat16 autocast', 'smoke': smoke, 'data_dir': str(DATA),
        'train_examples': len(arrays['train_labels']), 'val_examples': len(arrays['val_labels']),
        'split_sha256': sha256(DATA / 'manifest.json'), 'classes_sha256': sha256(Path('input/classes.npy')),
        'augmentation': 'rotation +/-15deg, scale 0.9-1.1, translation +/-0.1, no random flip'}
    assert config['smoke'] == smoke and config['split_sha256'] == sha256(DATA / 'manifest.json')
    assert config['classes_sha256'] == sha256(Path('input/classes.npy'))
    batch_size, epochs = config['batch_size'], config['epochs']
    batches = math.ceil(len(arrays['train_labels']) / batch_size)
    model = make_model(pretrained=not bool(parent))
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['lr'],
                    total_steps=epochs * batches, pct_start=config['pct_start'])
    history, best, steps, start_epoch, start_batch, metrics = [], -1., 0, 1, 0, (0., 0., 0)
    if saved:
        model.load_state_dict(saved['model']); ema.load_state_dict(saved['ema'])
        optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        history, best, steps = saved['history'], saved['best'], saved['steps']
        start_epoch, start_batch, metrics = saved['epoch'], saved['batch'], saved['metrics']
        if start_batch == batches and history and history[-1]['epoch'] == start_epoch:
            start_epoch, start_batch, metrics = start_epoch + 1, 0, (0., 0., 0)
        random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
        for name in ('model.pth', 'best.json'):
            if (parent / name).exists():
                shutil.copy2(parent / name, output / name)
    del saved
    config.update(parameters=sum(p.numel() for p in model.parameters()), torch=torch.__version__, timm=timm.__version__,
                  gpu=torch.cuda.get_device_name(0), pretrained_config=model.pretrained_cfg, deadline_unix=deadline)
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    (output / 'source.py').write_text(Path(__file__).read_text())
    for name in ('val_indices.npy', 'train_indices.npy', 'manifest.json'):
        shutil.copy2(DATA / name, output / name)
    print(json.dumps(config), flush=True)
    stopped = [False]
    def stop(signum, frame):
        stopped[0] = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    def progress(stage, epoch, batch, elapsed=0):
        record = {'stage': stage, 'epoch': epoch, 'batch': batch, 'batches': batches, 'elapsed': elapsed,
                  'remaining': elapsed / max(1, batch - (start_batch if epoch == start_epoch else 0)) * (batches - batch),
                  'updated_at': time.time()}
        (output / 'progress.tmp').write_text(json.dumps(record)); (output / 'progress.tmp').replace(output / 'progress.json')
    def checkpoint(epoch, batch, metrics):
        state = {'model': model.state_dict(), 'ema': ema.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'history': history, 'best': best, 'steps': steps,
            'epoch': epoch, 'batch': batch, 'metrics': metrics, 'config': config,
            'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}
        with (output / 'last.tmp').open('wb') as stream:
            torch.save(state, stream); stream.flush(); os.fsync(stream.fileno())
        (output / 'last.tmp').replace(output / 'last.pth')
        (output / 'checkpoint.json').write_text(json.dumps({'epoch': epoch, 'batch': batch, 'steps': steps, 'saved_at': time.time()}))
    ema_t = [v for v in ema.state_dict().values() if v.is_floating_point()]
    model_t = [v for v in model.state_dict().values() if v.is_floating_point()]
    checkpoint(start_epoch, start_batch, metrics)
    last_save = time.monotonic()
    epoch, batch = min(start_epoch, epochs), start_batch
    for epoch in range(start_epoch, epochs + 1):
        model.train(); start = time.monotonic(); last_progress = 0
        loss_sum, correct, count = metrics if epoch == start_epoch else (0., 0., 0)
        skipped = start_batch if epoch == start_epoch else 0
        batch = skipped
        for batch, (images, targets) in enumerate(loader(arrays['train_images'], arrays['train_labels'], batch_size, epoch, skipped), skipped + 1):
            x = prepare(images, True); y = targets.cuda(non_blocking=True)
            lam = float(np.random.beta(.2, .2)); order = torch.randperm(len(y), device='cuda')
            x = lam * x + (1 - lam) * x[order]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
                loss = lam * F.cross_entropy(logits, y, label_smoothing=.1) + (1 - lam) * F.cross_entropy(logits, y[order], label_smoothing=.1)
            assert torch.isfinite(loss).item(), 'Loss is not finite'
            loss.backward(); optimizer.step(); scheduler.step(); steps += 1
            decay = min(.9995, (1 + steps) / (10 + steps))
            with torch.no_grad():
                torch._foreach_mul_(ema_t, decay); torch._foreach_add_(ema_t, model_t, alpha=1 - decay)
            loss_sum += loss.item() * len(y)
            correct += (lam * (logits.detach().argmax(1) == y).sum().item()
                        + (1 - lam) * (logits.detach().argmax(1) == y[order]).sum().item())
            count += len(y)
            if steps == 1:
                print(json.dumps({'first_batch_loss': loss.item(), 'vram_peak_mib': torch.cuda.max_memory_allocated() / 2**20}), flush=True)
            if time.monotonic() - last_progress >= 5:
                progress('train', epoch, batch, time.monotonic() - start); last_progress = time.monotonic()
            if time.monotonic() - last_save >= 60 or stopped[0]:
                checkpoint(epoch, batch, (loss_sum, correct, count)); last_save = time.monotonic()
            if stopped[0]:
                progress('stopped', epoch, batch); return False
            if deadline and time.time() >= deadline - (30 if smoke else 900):
                break
        progress('val', epoch, batch, time.monotonic() - start)
        raw = float((predict(model, arrays['val_images'], batch_size).argmax(1) == arrays['val_labels']).mean())
        averaged = float((predict(ema, arrays['val_images'], batch_size).argmax(1) == arrays['val_labels']).mean())
        history.append({'epoch': epoch, 'loss': loss_sum / count, 'train_accuracy': correct / count,
            'accuracy': raw, 'ema_accuracy': averaged, 'examples': count, 'lr': optimizer.param_groups[0]['lr'],
            'seconds': time.monotonic() - start})
        for state, candidate, score in [('raw', model, raw), ('ema', ema, averaged)]:
            if score > best:
                best = score; torch.save(candidate.state_dict(), output / 'best.tmp')
                (output / 'best.tmp').replace(output / 'model.pth')
                (output / 'best.json').write_text(json.dumps({'epoch': epoch, 'state': state, 'accuracy': score}, indent=2))
        pd.DataFrame(history).to_csv(output / 'history.tmp', index=False); (output / 'history.tmp').replace(output / 'history.csv')
        checkpoint(epoch, batch, (loss_sum, correct, count))
        if batch == batches: os.link(output / 'last.pth', output / f'epoch-{epoch:02d}.pth')
        print(json.dumps(history[-1]), flush=True)
        if deadline and time.time() >= deadline - (30 if smoke else 900): break
    progress('inference', epoch, batch)
    del ema, optimizer
    finish(model, arrays, output, batch_size)
    return True


def verify(output):
    output = Path(output); config = json.loads((output / 'config.json').read_text())
    assert config['split_sha256'] == sha256(DATA / 'manifest.json')
    arrays = setup(config['smoke']); model = make_model()
    model.load_state_dict(torch.load(output / 'model.pth', map_location='cpu', weights_only=True))
    summary = json.loads((output / 'summary.json').read_text()); flip = summary['flip']
    val = predict(model, arrays['val_images'], config['batch_size'], flip).argmax(1)
    test = predict(model, arrays['test_images'], config['batch_size'], flip).argmax(1)
    submission = pd.read_csv(output / 'submission.csv')
    assert submission.columns.tolist() == ['id', 'label'] and submission['id'].tolist() == list(range(len(test)))
    result = {'val_examples': len(val), 'correct': int((val == arrays['val_labels']).sum()),
        'accuracy': float((val == arrays['val_labels']).mean()), 'submission_rows': len(test),
        'submission_differences': int((submission['label'].to_numpy() != test).sum())}
    assert result['accuracy'] == summary['selected']['accuracy'] and result['submission_differences'] == 0
    mode = 'flip' if flip else 'plain'
    result['val_probability_differences'] = int((val != np.load(output / f'val_{mode}.npy').argmax(1)).sum())
    result['test_probability_differences'] = int((test != np.load(output / f'test_{mode}.npy').argmax(1)).sum())
    assert result['val_probability_differences'] == result['test_probability_differences'] == 0
    (output / 'verification.json').write_text(json.dumps(result, indent=2)); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--deadline', type=float)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify: verify(args.output)
    else:
        success = run(args.output, args.epochs, args.batch_size, args.deadline, args.smoke, args.resume)
        if not success: raise SystemExit(2)
