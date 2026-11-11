"""Build or verify the deterministic P6-CAL protocol from a user-supplied GBS1 split."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import DEFAULT_CONFIG, load_config, supplied_path  # noqa: E402
from splits import build_protocol  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", help="GBS1 split JSON; or set GBS1_SPLIT")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default="outputs/p6_cal/protocol.json")
    args = parser.parse_args()
    source = supplied_path(args.split, "GBS1_SPLIT")
    protocol = build_protocol(source, load_config(args.config))
    output = Path(args.output)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != protocol:
            raise ValueError("Existing P6-CAL protocol differs; refusing to overwrite it")
        print(f"Verified frozen protocol: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
        print(f"Wrote protocol: {output}")
    print(json.dumps(protocol["counts"], indent=2))


if __name__ == "__main__":
    main()
