from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.models.resnet import BasicBlock
from tqdm.auto import tqdm

assert torch.cuda.is_available(), "PyTorch не видит NVIDIA GPU"
device = torch.device("cuda")
print(torch.cuda.get_device_name(0), torch.__version__)

seed = 42
batch_size = 512
compare_epochs = 20
finetune_epochs = 30

def set_seed():
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()
torch.backends.cudnn.benchmark = True
data_dir = Path("input")
baseline_path = data_dir / "baseline.pth"
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

train_images = np.load(data_dir / "train_images.npy", mmap_mode="r")
train_labels = np.load(data_dir / "train_labels.npy", mmap_mode="r")
val_images = np.load(data_dir / "val_images.npy", mmap_mode="r")
val_labels = np.load(data_dir / "val_labels.npy", mmap_mode="r")
test_images = np.load(data_dir / "test_images.npy", mmap_mode="r")
classes = np.load(data_dir / "classes.npy")
num_classes = len(classes)

assert len(train_images) == len(train_labels)
assert len(val_images) == len(val_labels)
print(f"Train: {len(train_images)}, validation: {len(val_images)}, test: {len(test_images)}, classes: {num_classes}")

augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(10, translate=(0.05, 0.05), scale=(0.9, 1.1),
                            interpolation=transforms.InterpolationMode.BILINEAR),
])

class Drawings(Dataset):
    def __init__(self, images, labels=None, augment=False):
        self.images = images
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = torch.from_numpy(self.images[index].copy()).reshape(1, 28, 28).float() / 255
        if self.augment:
            image = augmentation(image)
        image = (image - 0.5) / 0.5
        if self.labels is None:
            return image
        return image, int(self.labels[index])

train_loader = DataLoader(Drawings(train_images, train_labels, augment=True),
                          batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(Drawings(val_images, val_labels), batch_size=batch_size,
                        shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(Drawings(test_images), batch_size=batch_size,
                         shuffle=False, num_workers=0, pin_memory=True)

@torch.inference_mode()
def accuracy(model):
    model.eval()
    correct = 0
    for images, labels in val_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        correct += (model(images).argmax(1) == labels).sum().item()
    return correct / len(val_labels)

baseline = resnet18(weights=None, num_classes=num_classes)
baseline.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
baseline.load_state_dict(torch.load(baseline_path, map_location="cpu", weights_only=True))
baseline.to(device)
baseline_accuracy = accuracy(baseline)
baseline.cpu()
print(f"Baseline accuracy: {baseline_accuracy:.5f}")

def make_model(width):
    layers = [
        nn.Conv2d(1, width, 3, padding=1, bias=False),
        nn.BatchNorm2d(width),
        nn.ReLU(inplace=True),
    ]
    in_channels = width
    for stage in range(4):
        out_channels = width * 2 ** stage
        stride = 1 if stage == 0 else 2
        downsample = None
        if stride == 2:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers.extend([
            BasicBlock(in_channels, out_channels, stride=stride, downsample=downsample),
            BasicBlock(out_channels, out_channels),
        ])
        in_channels = out_channels
    layers.extend([nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(in_channels, num_classes)])
    model = nn.Sequential(*layers)
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
    return model

history = []


def fit(model, epochs, lr, name, best_accuracy=0.0):
    checkpoint = output_dir / f"{name}.pth"
    torch.save(model.state_dict(), checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    for epoch in range(1, epochs + 1):
        start = perf_counter()
        model.train()
        loss_sum = 0.0
        for images, labels in tqdm(train_loader, desc=f"{name} {epoch}/{epochs}", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(labels)

        val_accuracy = accuracy(model)
        row = {"model": name, "epoch": epoch, "loss": loss_sum / len(train_labels),
               "accuracy": val_accuracy, "lr": optimizer.param_groups[0]["lr"],
               "seconds": perf_counter() - start}
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        print(f"{name} {epoch}/{epochs}: loss={row['loss']:.4f}, accuracy={val_accuracy:.5f}, seconds={row['seconds']:.0f}")
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), checkpoint)
        scheduler.step()

    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return best_accuracy

results = []
for width in (32, 64):
    set_seed()
    model = make_model(width).to(device)
    set_seed()
    score = fit(model, compare_epochs, 1e-3, f"resnet{width}")
    results.append({"width": width, "accuracy": score})
    pd.DataFrame(results).to_csv(output_dir / "comparison.csv", index=False)
    del model
    torch.cuda.empty_cache()

pd.DataFrame(results)

best = max(results, key=lambda row: row["accuracy"])
best_width = best["width"]
set_seed()
model = make_model(best_width).to(device)
model.load_state_dict(torch.load(output_dir / f"resnet{best_width}.pth", map_location="cpu", weights_only=True))
print(f"ResNet {best_width}, accuracy before finetuning: {best['accuracy']:.5f}")
final_accuracy = fit(model, finetune_epochs, 3e-4, "best_model", best["accuracy"])

scores = pd.DataFrame([
    {"model": "baseline", "accuracy": baseline_accuracy},
    *[{"model": f"resnet{row['width']}", "accuracy": row["accuracy"]} for row in results],
    {"model": f"resnet{best_width}_finetuned", "accuracy": final_accuracy},
])
scores.to_csv(output_dir / "scores.csv", index=False)
scores

if baseline_accuracy > final_accuracy:
    del model
    model = baseline.to(device)
    selected_model = "baseline"
else:
    selected_model = f"resnet{best_width}_finetuned"
model.eval()

predictions = []
with torch.inference_mode():
    for images in tqdm(test_loader, desc="Test"):
        images = images.to(device, non_blocking=True)
        predictions.extend(model(images).argmax(1).cpu().tolist())

submission = pd.DataFrame({"id": np.arange(len(test_images)), "label": predictions})
assert len(submission) == len(test_images)
assert submission["label"].between(0, num_classes - 1).all()
submission.to_csv(output_dir / "submission.csv", index=False)
print(f"Model: {selected_model}, validation accuracy: {max(baseline_accuracy, final_accuracy):.5f}")
submission.head()
