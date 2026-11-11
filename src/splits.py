"""Construct the frozen P6-CAL train/calibration/final-test protocol."""
import json
from pathlib import Path

import numpy as np


def build_protocol(source_split, config):
    source = json.loads(Path(source_split).read_text(encoding="utf-8"))
    expected = config["protocol"]
    train = source["main_pretrain_files"]
    heldout = source["eval_heldout_files"]
    defective = source["bad_files"]
    counts = (len(train), len(heldout), len(defective))
    wanted = (expected["train_normal"], expected["heldout_normal"], expected["test_defective"])
    if counts != wanted:
        raise ValueError(f"Unexpected GBS1 counts {counts}; expected {wanted}")
    if set(train) & set(heldout):
        raise ValueError("The source split has train/held-out overlap")

    rng = np.random.RandomState(expected["calibration_seed"])
    order = rng.permutation(len(heldout))
    n_cal = expected["calibration_normal"]
    calibration = [heldout[i] for i in order[:n_cal]]
    test_normal = [heldout[i] for i in order[n_cal:]]
    protocol = {
        "name": expected["name"],
        "source_split": "user-supplied GBS1 split JSON",
        "calibration_seed": expected["calibration_seed"],
        "train_normal": train,
        "calibration_normal": calibration,
        "test_normal": test_normal,
        "test_defective": defective,
        "counts": {
            "train": len(train),
            "calibration": len(calibration),
            "test_normal": len(test_normal),
            "test_defective": len(defective),
        },
        "rule": "Thresholds use calibration_normal only; test_normal and all defective images are final-test only.",
    }
    overlap = (set(train) & set(calibration)) | (set(train) & set(test_normal)) | (set(calibration) & set(test_normal))
    if overlap:
        raise AssertionError("P6-CAL partition overlap detected")
    if len(test_normal) != expected["test_normal"]:
        raise AssertionError("Unexpected final-test normal count")
    return protocol
