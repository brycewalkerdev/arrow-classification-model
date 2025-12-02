from pathlib import Path
from collections import Counter

import cv2
import torch
from torch import nn
from torchvision import models
from torchvision.transforms.functional import normalize

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn


# 3-class mapping: must match training
IDX_TO_CLASS = {
    0: "left_arrow",
    1: "right_arrow",
    2: "none",
}

console = Console()


def load_model(model_path: Path, device: torch.device) -> nn.Module:
    """
    Load a 3-class ResNet18 model from checkpoint.
    """
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 3)  # left, right, none

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def preprocess_image(path: Path, device: torch.device) -> torch.Tensor:
    """
    OpenCV -> RGB -> 224x224 -> tensor -> normalize.
    """
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

    # HWC uint8 -> CHW float32 in [0, 1]
    img = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0

    img = normalize(
        img,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    img = img.unsqueeze(0)  # batch dimension
    return img.to(device)


def predict_image(model: nn.Module, img_path: Path, device: torch.device):
    """
    Run the model on a single image, return (pred_class_name, confidence_float).
    """
    x = preprocess_image(img_path, device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)

    pred_class = IDX_TO_CLASS[int(pred_idx)]
    return pred_class, float(conf)


def iter_images(root: Path):
    """
    Recursively yield all image files under root.
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in exts:
            yield p


def infer_actual_label(img_path: Path) -> str | None:
    """
    Determine the ground-truth label from either the parent directory
    (old layout) or the filename prefix (new flat layout: left#, right#, none#).
    """
    parent = img_path.parent.name.lower()
    stem = img_path.stem.lower()

    class_names = set(IDX_TO_CLASS.values())
    if parent in class_names:
        return parent

    for name in class_names:
        # accept stems like "left1", "left_arrow_02", "right-003"
        prefix = name.split("_")[0] if "_" in name else name
        if stem.startswith(prefix):
            return name
    return None


def main():
    # project_root: .../arrow-classification-model/
    project_root = Path(__file__).resolve().parents[2]

    # model and predict directory are relative to project root
    model_path = project_root / "arrow_classifier_resnet18_3class.pth"
    predict_root = project_root / "predict"

    if not model_path.exists():
        console.print(f"[red]Model not found:[/red] {model_path}")
        return

    if not predict_root.exists():
        console.print(f"[red]Predict folder not found:[/red] {predict_root}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    image_paths = list(iter_images(predict_root))
    if not image_paths:
        console.print(f"[yellow]No images found under {predict_root}[/yellow]")
        return

    console.print(
        f"[bold cyan]Running predictions on {len(image_paths)} images "
        f"under {predict_root}...[/bold cyan]\n"
    )

    results = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Predicting", total=len(image_paths))

        for img_path in image_paths:
            actual = infer_actual_label(img_path) or "unknown"

            pred_class, conf = predict_image(model, img_path, device)
            correct = pred_class == actual

            results.append(
                {
                    "path": img_path,
                    "rel_path": img_path.relative_to(predict_root),
                    "actual": actual,
                    "pred": pred_class,
                    "conf": conf,
                    "correct": correct,
                }
            )

            progress.update(task, advance=1)

    # ---- Build result table ----
    table = Table(title="Arrow Classification Results")
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("Image", style="white")
    table.add_column("Actual", style="cyan")
    table.add_column("Predicted", style="magenta")
    table.add_column("Conf", justify="right")
    table.add_column("Correct?", justify="center")

    for i, r in enumerate(results, start=1):
        conf_pct = f"{r['conf'] * 100:.1f}%"
        correct_str = "[green]✔[/green]" if r["correct"] else "[red]✘[/red]"

        table.add_row(
            str(i),
            str(r["rel_path"]),
            r["actual"],
            r["pred"],
            conf_pct,
            correct_str,
        )

    console.print()
    console.print(table)

    # ---- Summary: overall & per-class accuracy ----
    console.print()

    # overall
    total = len(results)
    total_correct = sum(1 for r in results if r["correct"])
    overall_acc = total_correct / total if total > 0 else 0.0
    console.print(
        f"[bold]Overall accuracy:[/bold] {total_correct}/{total} "
        f"({overall_acc*100:.1f}%)"
    )

    # per-class
    per_class_total = Counter()
    per_class_correct = Counter()
    for r in results:
        per_class_total[r["actual"]] += 1
        if r["correct"]:
            per_class_correct[r["actual"]] += 1

    console.print("\n[bold]Per-class accuracy:[/bold]")
    for cls in sorted(per_class_total.keys()):
        tot = per_class_total[cls]
        cor = per_class_correct[cls]
        acc = cor / tot if tot > 0 else 0.0
        console.print(f"  {cls}: {cor}/{tot} ({acc*100:.1f}%)")


if __name__ == "__main__":
    main()
