"""
Live camera arrow classification demo using OpenCV.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from rich.console import Console
from torch import nn
from torchvision.transforms.functional import normalize

from arrow_classification_model.predict_main import IDX_TO_CLASS, load_model

console = Console()


def preprocess_frame(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert a BGR frame to normalized tensor batch shaped for ResNet18.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    tensor = normalize(
        tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    return tensor.unsqueeze(0).to(device)


def predict_frame(model: nn.Module, frame: np.ndarray, device: torch.device) -> tuple[str, float]:
    """
    Run a forward pass on a single frame and return (class_name, confidence).
    """
    with torch.no_grad():
        batch = preprocess_frame(frame, device)
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)

    return IDX_TO_CLASS[int(pred_idx)], float(conf)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    model_path = project_root / "arrow_classifier_resnet18_3class.pth"

    if not model_path.exists():
        console.print(f"[red]Model not found:[/red] {model_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        console.print("[red]Could not open default camera (index 0).[/red]")
        return

    console.print(
        "[bold cyan]Camera feed running.[/bold cyan] "
        "Press 'q' to quit."
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                console.print("[red]Failed to read frame from camera.[/red]")
                break

            pred_class, conf = predict_frame(model, frame, device)
            overlay = f"{pred_class} ({conf * 100:.1f}%)"

            cv2.putText(
                frame,
                overlay,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Arrow Classifier - press 'q' to quit", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
