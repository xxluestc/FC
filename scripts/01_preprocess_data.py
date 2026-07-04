from pathlib import Path
import argparse, json, sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fc_power.data.preprocess import canonicalize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    a = p.parse_args()
    files = sorted(
        f
        for f in a.input_dir.glob("*.csv")
        if "故障" not in f.parts and "故障" not in f.name
    )
    data, info = canonicalize(files)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(a.output, index=False)
    a.summary.parent.mkdir(parents=True, exist_ok=True)
    a.summary.write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
