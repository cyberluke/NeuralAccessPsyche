import argparse, json, os
from .config import TrainingConfig
from .fixtures import generate_fixture
from .manifests import atomic_json, make_manifest
from .trainer import train

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m nram_training"); sub = p.add_subparsers(dest="command", required=True)
    for name in ("doctor", "inspect-data", "smoke", "validate", "package"): sub.add_parser(name)
    t = sub.add_parser("train"); t.add_argument("--adapter", choices=("nontoxic", "toxic"), required=True); t.add_argument("--output-dir", default="artifacts/training/adapters"); t.add_argument("--resume", default="auto")
    t.add_argument("--model", default=None); t.add_argument("--max-steps", type=int, default=None)
    args = p.parse_args(argv)
    if args.command == "doctor": print(json.dumps({"python": os.sys.version.split()[0], "cuda": "UNAVAILABLE", "status": "CPU_IMPORTS_OK"})); return 0
    if args.command == "inspect-data": print(json.dumps({"dataset": TrainingConfig().dataset, "thresholds": [0.05, 0.80], "status": "MANIFEST_REQUIRED"})); return 0
    if args.command == "smoke":
        for delta in (False, True): generate_fixture("artifacts/training/fixtures", delta=delta)
        train("nontoxic", "artifacts/training/smoke", smoke=True); train("toxic", "artifacts/training/smoke", smoke=True); return 0
    if args.command == "train": train(args.adapter, args.output_dir, resume=args.resume, model_name=args.model, max_steps=args.max_steps); return 0
    if args.command == "validate": print(json.dumps({"status": "CPU_CONTRACT_ONLY", "gpu_runtime": "UNAVAILABLE"})); return 0
    atomic_json("artifacts/training/package-manifest.json", make_manifest(TrainingConfig().as_dict())); return 0
