import argparse
import sys
from pathlib import Path

try:
	from ultralytics import YOLO
except Exception as exc:  # pragma: no cover
	print("Error: Failed to import 'ultralytics'. Please install it with 'pip install ultralytics ncnn'.")
	print(f"Details: {exc}")
	sys.exit(1)


def parse_imgsz(imgsz_arg: str) -> tuple[int, int]:
	"""Parse image size input which can be a single int (e.g., '640') or WxH (e.g., '640x480')."""
	text = str(imgsz_arg).lower().replace("*", "x").replace(" ", "")
	if "x" in text:
		w_str, h_str = text.split("x", 1)
		w, h = int(w_str), int(h_str)
		if w <= 0 or h <= 0:
			raise ValueError("Width and height must be positive integers.")
		return (w, h)
	# single number -> square
	sz = int(text)
	if sz <= 0:
		raise ValueError("Image size must be a positive integer.")
	return (sz, sz)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Convert a YOLO .pt model to NCNN format using Ultralytics export."
	)
	parser.add_argument(
		"--model",
		required=True,
		help="Path to YOLO .pt model file (e.g., 'yolo11n.pt' or a custom model).",
	)
	parser.add_argument(
		"--imgsz",
		default="640",
		help="Inference image size. Accepts a single int (e.g., '640') or 'WxH' (e.g., '640x480').",
	)
	parser.add_argument(
		"--device",
		default="cpu",
		help="Device for export pre-checks (e.g., 'cpu', 'cuda:0'). Exported NCNN runs on CPU by default.",
	)
	parser.add_argument(
		"--half",
		action="store_true",
		help="Export with half-precision where supported (FP16).",
	)
	parser.add_argument(
		"--simplify",
		action="store_true",
		help="Simplify the model graph prior to export where supported.",
	)
	parser.add_argument(
		"--project",
		default=None,
		help="Project directory for saving exports. Defaults to current working directory.",
	)
	parser.add_argument(
		"--name",
		default=None,
		help="Run name for the export folder. Defaults to '<weights_name>_ncnn_model'.",
	)
	parser.add_argument(
		"--exist-ok",
		action="store_true",
		help="Allow existing project/name directory without incrementing.",
	)
	parser.add_argument(
		"--verbose",
		action="store_true",
		help="Enable verbose logging from Ultralytics during export.",
	)

	args = parser.parse_args()

	weights_path = Path(args.model)
	if not weights_path.exists():
		print(f"Error: Model file not found: {weights_path}")
		sys.exit(1)

	try:
		imgsz = parse_imgsz(args.imgsz)
	except Exception as exc:
		print(f"Error parsing --imgsz: {exc}")
		sys.exit(1)

	# Load and export
	print(f"Loading model: {weights_path}")
	model = YOLO(str(weights_path))

	export_kwargs = {
		"format": "ncnn",
		"imgsz": imgsz,
		"half": bool(args.half),
		"simplify": bool(args.simplify),
		"device": args.device,
		"verbose": bool(args.verbose),
	}

	# Optional project/name control
	if args.project is not None:
		export_kwargs["project"] = args.project
	if args.name is not None:
		export_kwargs["name"] = args.name
	if args.exist_ok:
		export_kwargs["exist_ok"] = True

	print("Starting export to NCNN...")
	result_path = model.export(**export_kwargs)
	# Ultralytics returns a path to the exported artifact or directory depending on format
	print("Export complete.")
	print(f"NCNN model saved to: {result_path}")
	print("\nNext steps:")
	print(" - Use the exported '<name>_ncnn_model' folder with yolo_detect.py, e.g.")
	print("   python yolo_detect.py --model=<name>_ncnn_model --source=usb0 --resolution=640x480")


if __name__ == "__main__":
	main()


