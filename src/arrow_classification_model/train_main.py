from pathlib import Path

import pandas as pd
import cv2
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models
from torchvision.transforms.functional import normalize


CLASS_TO_IDX = {
    "left_arrow": 0,
    "right_arrow": 1,
    "none": 2,
}


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

    def __init__(self, train_root: Path, transform=None):
        self.train_root = Path(train_root)
        self.arrow_dir = self.train_root / "arrow"
        self.none_dir = self.train_root / "none"
        self.transform = transform

        self.samples: list[tuple[Path, int]] = []

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
                # optional: warn or skip quietly
                continue

            self.samples.append((img_path, CLASS_TO_IDX[label_name]))

        # ---- 2) load "none" images ----
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        if self.none_dir.exists():
            for p in sorted(self.none_dir.rglob("*")):
                if p.suffix.lower() in exts:
                    self.samples.append((p, CLASS_TO_IDX["none"]))

        if not self.samples:
            raise RuntimeError(f"No samples found under {self.train_root}")

        print(f"Loaded {len(self.samples)} samples total:")
        print(f"  arrow images: {len(df)}")
        print(
            f"  none images:  "
            f"{sum(1 for _, idx in self.samples if idx == CLASS_TO_IDX['none'])}"
        )

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

        # Normalize with ImageNet stats
        img = normalize(
            img,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def main():
    # ---- paths ----
    # adjust this if your train dir lives somewhere else
    project_root = Path(__file__).resolve().parents[2]
    train_root = project_root / "train"

    # ---- dataset & split ----
    full_ds = Arrow3ClassDataset(train_root)

    val_ratio = 0.2
    val_size = int(len(full_ds) * val_ratio)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

    # ---- model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(CLASS_TO_IDX))  # 3 classes now
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 5  # bump when things look sane

    for epoch in range(num_epochs):
        # ---- train ----
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        # ---- validate ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                _, preds = outputs.max(1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.3f} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.3f}"
        )

    out_path = project_root / "arrow_classifier_resnet18_3class.pth"
    torch.save(model.state_dict(), out_path)
    print(f"Saved 3-class model to {out_path}")


if __name__ == "__main__":
    main()
