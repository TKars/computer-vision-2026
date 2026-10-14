import argparse
import json
import math
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import timm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18


class Drawings(Dataset):
    def __init__(self, images, labels=None):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = torch.from_numpy(self.images[index].copy()).reshape(1, 28, 28)
        if self.labels is None:
            return image
        return image, int(self.labels[index])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_model(stem, weights):
    model = resnet18(weights=None, num_classes=345)
    model.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
    model.load_state_dict(weights)
    if stem == 3:
        kernel = model.conv1.weight.detach()[:, :, 2:5, 2:5].clone()
        model.conv1 = nn.Conv2d(1, 64, 3, padding=1, bias=False)
        with torch.no_grad():
            model.conv1.weight.copy_(kernel)
    elif stem == 7:
        model.conv1.stride = (1, 1)
    if stem:
        model.maxpool = nn.Identity()
    return model.cuda().to(memory_format=torch.channels_last)


def prepare(images, augment=False):
    images = images.cuda(non_blocking=True).float().div_(255)
    if augment:
        count = len(images)
        angle = torch.empty(count, device='cuda').uniform_(-10, 10) * math.pi / 180
        scale = torch.empty(count, device='cuda').uniform_(0.9, 1.1)
        flip = torch.where(torch.rand(count, device='cuda') < 0.5, -1.0, 1.0)
        theta = torch.zeros(count, 2, 3, device='cuda')
        theta[:, 0, 0] = angle.cos() / scale * flip
        theta[:, 0, 1] = -angle.sin() / scale
        theta[:, 1, 0] = angle.sin() / scale * flip
        theta[:, 1, 1] = angle.cos() / scale
        theta[:, :, 2] = torch.empty(count, 2, device='cuda').uniform_(-0.1, 0.1)
        grid = F.affine_grid(theta, images.shape, align_corners=False)
        images = F.grid_sample(images, grid, align_corners=False)
    return images.mul_(2).sub_(1).contiguous(memory_format=torch.channels_last)


@torch.inference_mode()
def accuracy(model, loader):
    model.eval()
    correct = 0
    for images, labels in loader:
        predictions = model(prepare(images)).argmax(1)
        correct += (predictions == labels.cuda(non_blocking=True)).sum().item()
    return correct / len(loader.dataset)


@torch.inference_mode()
def predict(model, loader, flip=False):
    model.eval()
    probabilities = []
    for batch in loader:
        images = prepare(batch[0] if isinstance(batch, (tuple, list)) else batch)
        probs = model(images).softmax(1)
        if flip:
            probs = (probs + model(images.flip(-1)).softmax(1)) / 2
        probabilities.append(probs.cpu().numpy())
    return np.concatenate(probabilities)


def fit(model, train_loader, val_loader, output, name, seconds, epochs, lr,
        history, best_accuracy=0.0):
    checkpoint = output / f'{name}.pth'
    torch.save(model.state_dict(), checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    deadline = perf_counter() + seconds
    for epoch in range(1, epochs + 1):
        if perf_counter() >= deadline:
            break
        start = perf_counter()
        model.train()
        loss_sum, correct, count = 0.0, 0, 0
        for step, (images, labels) in enumerate(train_loader, 1):
            images = prepare(images, augment=True)
            labels = labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(labels)
            correct += (logits.detach().argmax(1) == labels).sum().item()
            count += len(labels)
            if step == 10 or step % 200 == 0:
                print(f'{name} epoch={epoch} step={step}/{len(train_loader)} loss={loss_sum/count:.4f}', flush=True)
            if perf_counter() >= deadline:
                break
        score = accuracy(model, val_loader)
        row = {'model': name, 'epoch': epoch, 'loss': loss_sum / count,
               'train_accuracy': correct / count, 'accuracy': score, 'examples': count,
               'lr': optimizer.param_groups[0]['lr'], 'seconds': perf_counter() - start}
        history.append(row)
        pd.DataFrame(history).to_csv(output / 'history.csv', index=False)
        print(f'{name} epoch={epoch} accuracy={score:.5f} seconds={row["seconds"]:.1f}', flush=True)
        if score > best_accuracy:
            best_accuracy = score
            torch.save(model.state_dict(), checkpoint)
        scheduler.step()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    return best_accuracy


def run(minutes=165, output='output/run', batch_size=512, smoke=False):
    assert torch.cuda.is_available(), 'PyTorch не видит NVIDIA GPU'
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / 'history.csv').exists(), 'Укажите новую папку для запуска'
    start = perf_counter()
    deadline = start + minutes * 60
    set_seed(42)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    data = Path('input')
    arrays = {name: np.load(data / f'{name}.npy', mmap_mode='r')
              for name in ('train_images', 'train_labels', 'val_images', 'val_labels', 'test_images')}
    if smoke:
        arrays = {name: values[:64] for name, values in arrays.items()}
        batch_size = 16
    loaders = {}
    for split in ('train', 'val', 'test'):
        dataset = Drawings(arrays[f'{split}_images'], arrays.get(f'{split}_labels'))
        loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=split == 'train',
                                   num_workers=0 if smoke else 4, pin_memory=True,
                                   persistent_workers=not smoke)
    weights = torch.load(data / 'baseline.pth', map_location='cpu', weights_only=True)
    config = {'minutes': minutes, 'batch_size': batch_size, 'seed': 42, 'stems': [3, 7],
              'compare_epochs': 40, 'finetune_epochs': 30, 'compare_lr': 3e-4,
              'finetune_lr': 1e-4, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)}
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    (output / 'source.py').write_text(Path(__file__).read_text())
    print(json.dumps(config), flush=True)
    baseline = timm.create_model('resnet18', pretrained=False, num_classes=345, in_chans=1).cuda().eval()
    baseline.load_state_dict(weights)
    baseline_probs = predict(baseline, loaders['val'])
    labels = arrays['val_labels']
    baseline_accuracy = float((baseline_probs.argmax(1) == labels).mean())
    np.save(output / 'baseline_val_probs.npy', baseline_probs)
    scores = [{'model': 'baseline', 'accuracy': baseline_accuracy}]
    pd.DataFrame(scores).to_csv(output / 'scores.csv', index=False)
    print(f'Baseline accuracy={baseline_accuracy:.8f} correct={int((baseline_probs.argmax(1)==labels).sum())}/{len(labels)}', flush=True)
    del baseline, baseline_probs
    torch.cuda.empty_cache()
    history, results = [], []
    seconds = max(0, (deadline - perf_counter() - (30 if smoke else 600)) / 3)
    for stem in (3, 7):
        set_seed(42)
        model = make_model(stem, weights)
        score = fit(model, loaders['train'], loaders['val'], output, f'resnet18_{stem}',
                    seconds, 1 if smoke else 40, 3e-4, history)
        results.append({'stem': stem, 'accuracy': score})
        scores.append({'model': f'resnet18_{stem}', 'accuracy': score})
        pd.DataFrame(scores).to_csv(output / 'scores.csv', index=False)
        del model
        torch.cuda.empty_cache()
    best = max(results, key=lambda row: row['accuracy'])
    set_seed(42)
    model = make_model(best['stem'], weights)
    model.load_state_dict(torch.load(output / f'resnet18_{best["stem"]}.pth', map_location='cpu', weights_only=True))
    final_score = fit(model, loaders['train'], loaders['val'], output, 'finetuned',
                      max(0, deadline - perf_counter() - (30 if smoke else 600)),
                      1 if smoke else 30, 1e-4, history, best['accuracy'])
    scores.append({'model': 'finetuned', 'accuracy': final_score})
    del model
    torch.cuda.empty_cache()
    other_stem = 7 if best['stem'] == 3 else 3
    specifications = [{'stem': best['stem'], 'checkpoint': 'finetuned.pth'},
                      {'stem': other_stem, 'checkpoint': f'resnet18_{other_stem}.pth'}]
    probabilities = []
    for spec in specifications:
        model = make_model(spec['stem'], weights)
        model.load_state_dict(torch.load(output / spec['checkpoint'], map_location='cpu', weights_only=True))
        probs = predict(model, loaders['val'], flip=True)
        probabilities.append(probs)
        np.save(output / f'{spec["checkpoint"][:-4]}_val_tta.npy', probs)
        scores.append({'model': spec['checkpoint'][:-4] + '_tta', 'accuracy': float((probs.argmax(1) == labels).mean())})
        del model
    ensemble_accuracy = float(((probabilities[0] + probabilities[1]).argmax(1) == labels).mean())
    scores.append({'model': 'ensemble_tta', 'accuracy': ensemble_accuracy})
    pd.DataFrame(scores).to_csv(output / 'scores.csv', index=False)
    selected = max(scores, key=lambda row: row['accuracy'])
    if selected['model'] == 'ensemble_tta':
        chosen, flip = specifications, True
    elif selected['model'] == 'baseline':
        chosen, flip = [{'stem': 0, 'checkpoint': None}], False
    elif selected['model'].startswith('finetuned'):
        chosen, flip = [specifications[0]], selected['model'].endswith('_tta')
    else:
        stem = int(selected['model'].split('_')[1])
        chosen, flip = [{'stem': stem, 'checkpoint': f'resnet18_{stem}.pth'}], selected['model'].endswith('_tta')
    test_probs = None
    for spec in chosen:
        model = make_model(spec['stem'], weights)
        if spec['checkpoint']:
            model.load_state_dict(torch.load(output / spec['checkpoint'], map_location='cpu', weights_only=True))
        probs = predict(model, loaders['test'], flip=flip)
        test_probs = probs if test_probs is None else test_probs + probs
        del model
    submission = pd.DataFrame({'id': np.arange(len(test_probs)), 'label': test_probs.argmax(1)})
    assert submission['label'].between(0, 344).all()
    submission.to_csv(output / 'submission.csv', index=False)
    summary = {'selected': selected, 'models': chosen, 'flip_tta': flip,
               'delta_baseline': selected['accuracy'] - baseline_accuracy,
               'val_above_6_threshold': selected['accuracy'] > 0.76102,
               'val_above_8_threshold': selected['accuracy'] > 0.77489,
               'seconds': perf_counter() - start, 'submission_rows': len(submission)}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(pd.DataFrame(scores).to_string(index=False), flush=True)
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--minutes', type=float, default=165)
    parser.add_argument('--output', default='output/run')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.minutes, args.output, args.batch_size, args.smoke)
