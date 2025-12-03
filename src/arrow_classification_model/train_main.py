import os
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.transforms.functional import normalize


CLASS_TO_IDX = {
    "left_arrow": 0,
    "right_arrow": 1,
    "none": 2,
}


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32
    val_ratio: float = 0.2
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 30
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_weighted_sampler: bool = False
    use_class_weights: bool = True


def get_env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def get_env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_train_transform():
    return transforms.Compose(
        [
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.1
            ),
        ]
    )


class Arrow3ClassDataset(Dataset):
    """
    3-class dataset: left_arrow, right_arrow, none.

    Layout:
      train/
        arrow/
          _classes.csv   (filename,left,right)
          *.jpg          (filenames from CSV)
        none/
          *.jpg          (no-arrow examples)
    """

    def __init__(
        self,
        train_root: Path,
        transform=None,
        samples: list[tuple[Path, int]] | None = None,
        *,
        quiet: bool = False,
    ):
        self.train_root = Path(train_root)
        self.arrow_dir = self.train_root / "arrow"
        self.none_dir = self.train_root / "none"
        self.transform = transform

        self.samples: list[tuple[Path, int]] = samples if samples is not None else []

        if samples is None:
            self.samples = self._load_samples()
            if not quiet:
                self._log_counts()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # ---- OpenCV load ----
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

        # HWC uint8 -> CHW float32 in [0, 1]
        img = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0

        if self.transform is not None:
            img = self.transform(img)

        # Normalize with ImageNet stats
        img = normalize(
            img,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        return img, label

    def _load_samples(self) -> list[tuple[Path, int]]:
        samples: list[tuple[Path, int]] = []
        exts = {".jpg", ".jpeg", ".png", ".bmp"}

        # ---- 1) load left/right from _classes.csv ----
        csv_path = self.arrow_dir / "_classes.csv"
        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            fname = row["filename"]
            left = int(row["left"])
            right = int(row["right"])

            if left == 1 and right == 0:
                label_name = "left_arrow"
            elif left == 0 and right == 1:
                label_name = "right_arrow"
            else:
                # if something weird is in the CSV, skip it
                continue

            img_path = self.arrow_dir / fname
            if not img_path.exists():
                continue

            samples.append((img_path, CLASS_TO_IDX[label_name]))

        # ---- 2) load "none" images ----
        if self.none_dir.exists():
            for p in sorted(self.none_dir.rglob("*")):
                if p.suffix.lower() in exts:
                    samples.append((p, CLASS_TO_IDX["none"]))

        # ---- 3) load custom camera captures (optional) ----
        custom_root = self.train_root / "custom_arrow"
        if custom_root.exists():
            for class_name in CLASS_TO_IDX.keys():
                class_dir = custom_root / class_name
                if not class_dir.exists():
                    continue
                for p in sorted(class_dir.rglob("*")):
                    if p.suffix.lower() in exts:
                        samples.append((p, CLASS_TO_IDX[class_name]))

        if not samples:
            raise RuntimeError(f"No samples found under {self.train_root}")

        self._df = df
        return samples

    def _log_counts(self):
        counts = class_counts(self.samples)
        print(f"Loaded {len(self.samples)} samples total:")
        for name, idx in CLASS_TO_IDX.items():
            print(f"  {name}: {counts.get(idx, 0)}")


def split_samples(
    samples: list[tuple[Path, int]], val_ratio: float, seed: int
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(samples), generator=generator).tolist()

    val_size = int(len(samples) * val_ratio)
    val_indices = set(indices[:val_size])

    train_samples = [s for i, s in enumerate(samples) if i not in val_indices]
    val_samples = [s for i, s in enumerate(samples) if i in val_indices]

    return train_samples, val_samples


def class_counts(samples: list[tuple[Path, int]]) -> dict[int, int]:
    counts: dict[int, int] = {i: 0 for i in CLASS_TO_IDX.values()}
    for _, label in samples:
        counts[label] = counts.get(label, 0) + 1
    return counts


def make_class_weights(counts: dict[int, int]) -> torch.Tensor:
    total = sum(counts.values())
    num_classes = len(CLASS_TO_IDX)
    weights = []
    for cls_idx in range(num_classes):
        count = counts.get(cls_idx, 1)
        weights.append(total / (num_classes * max(count, 1)))
    return torch.tensor(weights, dtype=torch.float32)


def make_sample_weights(samples: list[tuple[Path, int]], class_weights: torch.Tensor):
    return torch.tensor([class_weights[label] for _, label in samples], dtype=torch.float32)


def parse_weight_override(env_value: str | None) -> torch.Tensor | None:
    if not env_value:
        return None
    try:
        parts = [float(x.strip()) for x in env_value.split(",")]
    except ValueError:
        return None
    if len(parts) != len(CLASS_TO_IDX):
        return None
    return torch.tensor(parts, dtype=torch.float32)


def main():
    project_root = Path(__file__).resolve().parents[2]
    config = TrainConfig(
        epochs=get_env_int("TRAIN_EPOCHS", 1000),
        batch_size=get_env_int("TRAIN_BATCH_SIZE", 32),
        num_workers=get_env_int("TRAIN_NUM_WORKERS", 0),
        lr=get_env_float("TRAIN_LR", 3e-4),
        weight_decay=get_env_float("TRAIN_WEIGHT_DECAY", 1e-4),
        use_weighted_sampler=os.environ.get("TRAIN_WEIGHTED_SAMPLER", "0") == "1",
        use_class_weights=os.environ.get("TRAIN_CLASS_WEIGHTS", "1") != "0",
    )

    # adjust this if your train dir lives somewhere else
    train_root = project_root / "train"
    out_path = project_root / "arrow_classifier_resnet18_3class.pth"

    set_seed(config.seed)

    # ---- dataset & split ----
    base_ds = Arrow3ClassDataset(train_root)
    train_samples, val_samples = split_samples(
        base_ds.samples, val_ratio=config.val_ratio, seed=config.seed
    )

    train_ds = Arrow3ClassDataset(
        train_root,
        transform=build_train_transform(),
        samples=train_samples,
        quiet=True,
    )
    val_ds = Arrow3ClassDataset(train_root, samples=val_samples, quiet=True)

    device = torch.device(config.device)
    pin_memory = device.type == "cuda"

    train_counts = class_counts(train_samples)
    val_counts = class_counts(val_samples)
    print(f"Train split counts: {train_counts}")
    print(f"Val split counts:   {val_counts}")

    override_weights = parse_weight_override(os.environ.get("TRAIN_CLASS_WEIGHT_OVERRIDE"))
    class_weights = (
        override_weights
        if override_weights is not None
        else (make_class_weights(train_counts) if config.use_class_weights else None)
    )
    sample_weights = (
        make_sample_weights(train_samples, class_weights) if class_weights is not None else None
    )
    sampler = (
        WeightedRandomSampler(
            sample_weights, num_samples=len(train_samples), replacement=True
        )
        if (config.use_weighted_sampler and sample_weights is not None)
        else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=sampler is None,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        sampler=sampler,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    # ---- model ----
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(CLASS_TO_IDX))  # 3 classes now
    model.to(device)

    weight_tensor = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(label_smoothing=0.02, weight=weight_tensor)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.lr * 0.1
    )

    best_val_acc = 0.0
    best_state = None

    for epoch in range(config.epochs):
        # ---- train ----
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        # ---- validate ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                _, preds = outputs.max(1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        scheduler.step()

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_state = model.state_dict()

        print(
            f"Epoch {epoch+1}/{config.epochs} | "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.3f} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.3f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

    final_state = best_state if best_state is not None else model.state_dict()
    torch.save(final_state, out_path)
    print(
        f"Saved best model to {out_path} "
        f"(best val acc: {best_val_acc*100:.1f}% over {config.epochs} epochs)"
    )


if __name__ == "__main__":
    main()
