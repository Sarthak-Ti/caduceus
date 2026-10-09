#!/usr/bin/env python3
"""Extract both observed CAGE targets in one chunk-aligned pass; publish on success."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time

import numcodecs
import numpy as np
import zarr


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def extract(config, workers=4):
    numcodecs.blosc.set_nthreads(1)
    started = time.monotonic()
    out = Path(config["data_dir"])
    stage = out.with_name(out.name + ".partial")
    if out.exists() or stage.exists():
        raise FileExistsError(f"Refusing to overwrite {out} or {stage}")
    source = zarr.open_group(config["source_zarr"], mode="r")
    with open(config["source_targets"]) as handle:
        targets = {int(row["index"]): row for row in csv.DictReader(handle, delimiter="\t")}
    cells = list(config["cells"])
    indices = [config["cells"][cell]["track_index"] for cell in cells]
    for cell in cells:
        expected = config["cells"][cell]
        observed = targets[expected["track_index"]]
        for key, column in [("track_identifier", "identifier"), ("description", "description")]:
            if observed[column].strip() != expected[key].strip():
                raise ValueError(f"Track metadata mismatch: {cell}, {key}")
    counts = Counter()
    with open(config["sequences_bed"]) as handle:
        for line in handle:
            split = line.rstrip().split("\t")[3]
            counts["val" if split == "valid" else split] += 1
    splits = ["train", "val", "test"]
    for split in splits:
        array = source[split]
        if array.shape[0] != counts[split] or array.shape[1] != 896:
            raise ValueError(f"Label/BED dimensions do not match for {split}: {array.shape}, {counts}")
        if max(indices) >= array.shape[2] or np.dtype(array.dtype) != np.dtype("float32"):
            raise ValueError(f"Unexpected source tracks/dtype: {array.shape}, {array.dtype}")

    stage.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config["sequences_bed"], stage / "sequences_enformer.bed")
    report = {
        "status": "extracting", "source_zarr": config["source_zarr"],
        "source_targets": config["source_targets"],
        "source_targets_sha256": sha256(config["source_targets"]),
        "sequences_bed_sha256": sha256(config["sequences_bed"]),
        "content": "Observed FANTOM CAGE labels from the existing Enformer target store",
        "bin_size_bp": 128, "output_bins": 896, "dtype": "float32",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tracks": {cell: {"source_index": config["cells"][cell]["track_index"],
                           "output_index": 0,
                           "identifier": config["cells"][cell]["track_identifier"],
                           "description": config["cells"][cell]["description"]} for cell in cells},
        "splits": {}, "block_checks": [],
    }
    destinations = {}
    for cell in cells:
        group = zarr.open_group(str(stage / f"{cell}_CAGE.zarr"), mode="w-", zarr_format=2)
        group.attrs.update({"source_zarr": config["source_zarr"],
                            "source_track_index": config["cells"][cell]["track_index"],
                            "description": config["cells"][cell]["description"],
                            "bin_size_bp": 128,
                            "sequences_bed_sha256": report["sequences_bed_sha256"]})
        destinations[cell] = {}
        for split in splits:
            n, bins, _ = source[split].shape
            destinations[cell][split] = group.create_array(
                split, shape=(n, bins, 1), chunks=(source[split].chunks[0], bins, 1),
                dtype="float32", compressors=None,
            )
    work = []
    for split in splits:
        array = source[split]
        report["splits"][split] = {"source_shape": list(array.shape),
                                   "source_chunks": list(array.chunks),
                                   "output_shape": [array.shape[0], 896, 1]}
        work.extend((split, lo, min(lo + array.chunks[0], array.shape[0]))
                    for lo in range(0, array.shape[0], array.chunks[0]))

    def copy_block(item):
        split, lo, hi = item
        # Both columns are selected in ONE operation on a source-aligned chunk.
        selected = np.asarray(source[split].oindex[lo:hi, :, indices])
        if selected.shape != (hi - lo, 896, len(cells)):
            raise ValueError(f"Unexpected extraction shape: {selected.shape}")
        if not np.isfinite(selected).all() or (selected < 0).any():
            raise ValueError(f"Nonfinite/negative CAGE labels at {split}:{lo}-{hi}")
        result = {"split": split, "start": lo, "stop": hi, "cells": {}}
        for j, cell in enumerate(cells):
            values = np.ascontiguousarray(selected[:, :, j:j + 1])
            destinations[cell][split][lo:hi] = values
            result["cells"][cell] = {"sha256": array_hash(values),
                                     "nonzero": int(np.count_nonzero(values)),
                                     "sum": float(values.sum(dtype=np.float64))}
        return result

    print(f"Extracting {len(work)} source chunks for {cells}; workers={workers}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(copy_block, item) for item in work]
        for i, future in enumerate(as_completed(futures), 1):
            report["block_checks"].append(future.result())
            if i % 32 == 0 or i == len(work):
                elapsed = time.monotonic() - started
                print(f"Copied {i}/{len(work)} chunks ({elapsed / 60:.1f} min)", flush=True)
                write_json(stage / "progress.json", {"completed_chunks": i, "total_chunks": len(work)})

    print("Verifying every extracted value against its source-block digest", flush=True)
    for block in report["block_checks"]:
        for cell in cells:
            values = destinations[cell][block["split"]][block["start"]:block["stop"]]
            if array_hash(values) != block["cells"][cell]["sha256"]:
                raise ValueError(f"Output verification failed for {cell}, {block}")
    for cell in cells:
        for split in splits:
            if not any(b["cells"][cell]["nonzero"] for b in report["block_checks"] if b["split"] == split):
                raise ValueError(f"Entire extracted split is zero: {cell}/{split}")
    report["block_checks"].sort(key=lambda b: (b["split"], b["start"]))
    report["status"] = "complete"
    report["verification"] = "All output values exactly match source-selected float32 values (SHA256 per block)"
    report["elapsed_seconds"] = time.monotonic() - started
    write_json(stage / "extraction_manifest.json", report)
    (stage / "SUCCESS").write_text("All source blocks extracted and verified.\n")
    if out.exists():
        raise FileExistsError(out)
    stage.rename(out)
    print(f"COMPLETE: {out} ({report['elapsed_seconds'] / 60:.1f} min)", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    extract(json.loads(args.config.read_text()), args.workers)
