import os
import sys
import argparse
import tempfile
from ultralytics import YOLO


def build_coco_yaml(
    coco_root: str,
    train_images: str,
    val_images: str,
    test_images: str | None = None,
) -> str:
    """
    Create a minimal Ultralytics data.yaml for COCO-style datasets.

    Expected filesystem under coco_root:
      - images/train2017 (or custom)
      - images/val2017 (or custom)
      - images/test2017 (optional)
      - annotations/instances_train2017.json
      - annotations/instances_val2017.json
      - annotations/instances_test2017.json (optional)

    Ultralytics infers annotation JSONs from the split names by default,
    looking under {path}/annotations/instances_{split_name}.json where
    split_name is the last directory name of each split (e.g. "train2017").
    """
    coco_root = os.path.abspath(coco_root)

    yaml_lines: list[str] = [
        f"path: {coco_root}",
        f"train: {train_images}",
        f"val: {val_images}",
    ]
    if test_images:
        yaml_lines.append(f"test: {test_images}")

    # names can be omitted for COCO; Ultralytics will use IDs if not provided
    yaml_text = "\n".join(yaml_lines) + "\n"
    return yaml_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO on a COCO-style dataset using Ultralytics.")
    parser.add_argument(
        "--coco_root",
        type=str,
        required=True,
        help=(
            "Root folder of the COCO dataset (contains 'images/' and 'annotations/' subfolders). "
            "Example: F:/datasets/coco"
        ),
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="images/train2017",
        help="Relative path (from coco_root) to training images folder. Default: images/train2017",
    )
    parser.add_argument(
        "--val_split",
        type=str,
        default="images/val2017",
        help="Relative path (from coco_root) to validation images folder. Default: images/val2017",
    )
    parser.add_argument(
        "--test_split",
        type=str,
        default=None,
        help="Optional relative path (from coco_root) to test images folder. Example: images/test2017",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt" if os.path.exists("yolo11n.pt") else "yolov8n.pt",
        help="Model to train (path to .pt). Defaults to yolo11n.pt if present else yolov8n.pt",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs. Default: 100")
    parser.add_argument("--batch", type=int, default=16, help="Batch size. Default: 16")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training/inference. Default: 640")
    parser.add_argument("--device", type=str, default=None, help="Device: 'cpu', 'cuda', or index like '0'.")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers. Default: 8")
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Project directory for saving runs. Default: runs/detect",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="train",
        help="Run name. Results saved to {project}/{name}. Default: train",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training if a matching run exists.",
    )

    args = parser.parse_args()

    coco_root = args.coco_root
    train_split = args.train_split
    val_split = args.val_split
    test_split = args.test_split

    # Validate COCO filesystem expectations
    def _must_exist(path: str, kind: str) -> None:
        if not os.path.exists(path):
            print(f"ERROR: {kind} not found: {path}")
            sys.exit(1)

    _must_exist(os.path.join(coco_root, "images"), "images folder")
    _must_exist(os.path.join(coco_root, "annotations"), "annotations folder")
    _must_exist(os.path.join(coco_root, train_split), "train images folder")
    _must_exist(os.path.join(coco_root, val_split), "val images folder")
    if test_split:
        _must_exist(os.path.join(coco_root, test_split), "test images folder")

    # Heuristic check for expected COCO annotation JSONs
    train_leaf = os.path.basename(train_split.rstrip("/\\"))
    val_leaf = os.path.basename(val_split.rstrip("/\\"))
    train_json = os.path.join(coco_root, "annotations", f"instances_{train_leaf}.json")
    val_json = os.path.join(coco_root, "annotations", f"instances_{val_leaf}.json")
    if not os.path.isfile(train_json) or not os.path.isfile(val_json):
        print(
            "WARNING: COCO annotation JSONs not found at expected locations:\n"
            f"  {train_json}\n  {val_json}\n"
            "Ultralytics infers these by default. If your JSONs are named differently,"
            " please rename them to match 'instances_<split>.json' or reorganize to the standard layout."
        )

    # Build a temporary data.yaml describing this dataset
    data_yaml_text = build_coco_yaml(coco_root, train_split, val_split, test_split)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        tmp.write(data_yaml_text)
        data_yaml_path = tmp.name

    print("Using data.yaml:\n" + data_yaml_text)

    # Load model and start training
    if not os.path.exists(args.model):
        print(f"ERROR: Model weights not found: {args.model}")
        sys.exit(1)

    model = YOLO(args.model, task="detect")

    train_kwargs = {
        "data": data_yaml_path,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "resume": args.resume,
        "exist_ok": True,
    }

    results = model.train(**train_kwargs)
    print("Training completed. Best weights:", results.best if hasattr(results, "best") else "<see run directory>")


if __name__ == "__main__":
    main()


