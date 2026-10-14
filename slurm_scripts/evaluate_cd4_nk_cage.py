#!/usr/bin/env python3
"""Save held-out CD4/NK CAGE profiles and within/across-region correlations."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, pearsonr, spearmanr


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def correlation(x, y):
    """Undefined correlations stay null; do not count constant arrays as zero."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"pearson": None, "spearman": None, "n": int(x.size)}
    return {"pearson": float(pearsonr(x, y).statistic),
            "spearman": float(spearmanr(x, y).statistic), "n": int(x.size)}


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return {"mean": float(finite.mean()) if finite.size else None,
            "median": float(np.median(finite)) if finite.size else None,
            "n_defined": int(finite.size), "n_undefined": int(values.size - finite.size)}


def score_profiles(predictions, targets):
    """Each row is a region and each column a 128-bp output bin."""
    p, y = np.asarray(predictions, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    if p.ndim != 2 or p.shape != y.shape:
        raise ValueError(f"Expected matching (regions, bins) profiles, got {p.shape}, {y.shape}")
    if not np.isfinite(p).all() or not np.isfinite(y).all() or (p < 0).any() or (y < 0).any():
        raise ValueError("Profiles must be complete, finite and nonnegative")
    within = [correlation(a, b) for a, b in zip(p, y)]
    log_within = [correlation(np.log1p(a), np.log1p(b)) for a, b in zip(p, y)]
    pearson = np.array([r["pearson"] if r["pearson"] is not None else np.nan for r in within])
    spearman = np.array([r["spearman"] if r["spearman"] is not None else np.nan for r in within])
    log_pearson = np.array([r["pearson"] if r["pearson"] is not None else np.nan for r in log_within])
    predicted_sum, observed_sum = p.sum(axis=1), y.sum(axis=1)
    poisson = (p - y * np.log(p + 1e-6)).mean(axis=1)
    summary = {
        "n_regions": int(p.shape[0]), "n_bins_per_region": int(p.shape[1]),
        "within_region_raw": {"pearson": distribution(pearson), "spearman": distribution(spearman)},
        "within_region_log1p": {"pearson": distribution(log_pearson)},
        "across_regions_sum_raw": correlation(predicted_sum, observed_sum),
        "across_regions_sum_log1p": correlation(np.log1p(predicted_sum), np.log1p(observed_sum)),
        "pooled_bins_raw": correlation(p.ravel(), y.ravel()),
        "pooled_bins_log1p": correlation(np.log1p(p.ravel()), np.log1p(y.ravel())),
        "poisson_loss_mean": float(poisson.mean()),
        "zero_signal_regions": int((observed_sum == 0).sum()),
        "definitions": {
            "within_region": "Correlate 896 bins separately per region; summarize defined correlations without weighting.",
            "across_regions": "Sum all 896 bins first, then correlate the 1,937 predicted and observed region totals.",
            "log1p": "Natural log(1+x); for region totals apply after summing bins.",
            "pooled_bins": "Flatten all regions and bins; each stored bin occurrence gets equal weight.",
            "poisson": "Mean(pred - target * log(pred + 1e-6)), matching training with full=False; may be negative.",
            "undefined": "Constant-profile correlations are NA in TSV/NPY, null in JSON, and excluded from correlation summaries only.",
        },
    }
    rows = pd.DataFrame({"predicted_sum": predicted_sum, "observed_sum": observed_sum,
                         "pearson": pearson, "spearman": spearman,
                         "pearson_log1p": log_pearson, "poisson_loss": poisson})
    return summary, rows


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("cell", choices=["CD4", "NK"])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    experiment = json.loads(Path(manifest["experiment"]).read_text())
    selected = manifest["cells"][args.cell]
    checkpoint = Path(selected["checkpoint"])
    output = Path(selected["output_dir"])
    if (output / "SUCCESS").exists():
        previous = json.loads((output / "provenance.json").read_text())
        if previous["checkpoint_sha256"] != selected["checkpoint_sha256"]:
            raise ValueError("Existing evaluation used a different checkpoint")
        print(f"Evaluation already completed: {output}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    if list(output.iterdir()):
        raise FileExistsError(f"Partial evaluation already exists; inspect before choosing a new output directory: {output}")
    if sha256(checkpoint) != selected["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed after validation-based selection")
    saved_config = checkpoint.parent.parent / ".hydra/config.yaml"
    if sha256(saved_config) != selected["hydra_config_sha256"]:
        raise ValueError("Selected checkpoint's adjacent model configuration changed")

    repo = Path(experiment["caduceus_repo"])
    sys.path.insert(0, str(repo))
    import random
    import scipy
    import torch
    import zarr
    from evals.evals_utils_joint import Evals
    from evals.eval_joint_expression import evaluate_all

    started = time.time()
    random.seed(experiment["seed"])
    np.random.seed(experiment["seed"])
    torch.manual_seed(experiment["seed"])
    data_dir = Path(experiment["data_dir"])
    extraction = json.loads((data_dir / "extraction_manifest.json").read_text())
    if not (data_dir / "SUCCESS").is_file() or extraction["status"] != "complete":
        raise ValueError("Extraction has not completed successfully")
    bed = data_dir / "sequences_enformer.bed"
    if sha256(bed) != extraction["sequences_bed_sha256"]:
        raise ValueError("BED changed after extraction")
    expected_shape = tuple(extraction["splits"]["test"]["output_shape"])
    assert expected_shape == (1937, 896, 1), expected_shape
    target_path = data_dir / f"{args.cell}_CAGE.zarr"
    source = np.asarray(zarr.open_group(str(target_path), mode="r")["test"][:])
    regions = pd.read_csv(bed, sep="\t", header=None)
    regions = regions.loc[regions[3] == "test"].reset_index(drop=True)
    assert len(regions) == expected_shape[0]

    provenance = {"cell_type": args.cell, "started_utc": datetime.now(timezone.utc).isoformat(),
                  "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                  "checkpoint": str(checkpoint), "checkpoint_sha256": selected["checkpoint_sha256"],
                  "hydra_config_sha256": selected["hydra_config_sha256"],
                  "selected_epoch_zero_based": selected["epoch_zero_based"],
                  "selection_validation_loss": selected["validation_loss"],
                  "selection_rule": "Minimum validation Poisson loss; checkpoint fixed before test evaluation",
                  "test_targets": str(target_path), "test_array": "test", "output_track_index": 0,
                  "source_enformer_track_index": experiment["cells"][args.cell]["track_index"],
                  "sequences_bed_sha256": extraction["sequences_bed_sha256"],
                  "array_shape": [1937, 1, 896, 1],
                  "array_axes": ["test_region", "cell_type", "128bp_bin", "CAGE_track"],
                  "input_length_bp": 524288, "output_length_bp": 114688, "bin_size_bp": 128,
                  "inference": "Existing Evals helper: bf16 autocast, forward orientation, no shifts or masking",
                  "softplus": "Once inside EnformerDecoder; Evals.skip_softplus=True prevents a second transform",
                  "storage_dtype": "float32", "load_data_in_memory": True,
                  "torch_version": torch.__version__, "numpy_version": np.__version__,
                  "scipy_version": scipy.__version__, "script_sha256": sha256(__file__),
                  "source_sha256": {path: sha256(repo / path) for path in [
                      "evals/eval_joint_expression.py", "evals/evals_utils_joint.py",
                      "src/tasks/encoders.py", "src/tasks/decoders.py",
                      "src/models/sequence/dna_embedding.py",
                      "src/dataloaders/datasets/general_dataset.py"]}}
    save_json(output / "provenance.json", provenance)
    print(f"Loading {args.cell} epoch {selected['epoch_zero_based']} for all 1,937 test regions", flush=True)
    evaluator = Evals(str(checkpoint), split="test", load_data=True, device="cuda")
    evaluator.skip_softplus = True
    ds = evaluator.dataset
    assert ds.sequences.equals(regions), "Dataset/BED test order differs"
    assert len(ds) == 1937 and ds.celltypes == 1 and ds.length == 524288
    assert ds.mlm == ds.acc_mlm == ds.shift_sequences == 0 and not ds.rc_aug
    assert ds.data_path == experiment["cells"][args.cell]["dnase_npz"]
    assert ds.additional_data_path == str(target_path) and list(ds.additional_data_idxs) == [0]
    print("Checkpoint loaded strictly; test ordering, CAGE track, and inputs verified", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        pearson, spearman, predictions, targets = evaluate_all(
            evaluator, pool=1, num_workers=args.workers,
            save_outputs=True, save_targets=True, dtype=np.float32)
    assert predictions.shape == targets.shape == (1937, 1, 896, 1)
    assert np.array_equal(targets[:, 0], source), "Returned targets differ from extracted test data"
    metrics, region_scores = score_profiles(predictions[:, 0, :, 0], targets[:, 0, :, 0])
    assert np.allclose(pearson[:, 0, 0], region_scores.pearson, equal_nan=True, atol=2e-6)
    assert np.allclose(spearman[:, 0, 0], region_scores.spearman, equal_nan=True, atol=2e-6)
    for tag, values in [("outputs", predictions), ("targets", targets),
                        ("pearson", pearson), ("spearman", spearman)]:
        temporary = output / f"expression_{tag}.tmp.npy"
        np.save(temporary, values)
        temporary.replace(output / f"expression_{tag}.npy")

    regions.columns = ["chrom", "bed_start", "bed_end", "split"]
    regions.insert(0, "region_index", np.arange(len(regions)))
    regions.insert(1, "cell_type", args.cell)
    regions["input_start"] = regions.bed_start - (524288 - (regions.bed_end - regions.bed_start)) // 2
    regions["input_end"] = regions.input_start + 524288
    regions["output_start"] = regions.input_start + (524288 - 114688) // 2
    regions["output_end"] = regions.output_start + 114688
    pd.concat([regions, region_scores], axis=1).to_csv(output / "regions.tsv", sep="\t", index=False, na_rep="NA")
    metrics.update(cell_type=args.cell, selected_epoch_zero_based=selected["epoch_zero_based"],
                   selection_validation_loss=selected["validation_loss"], elapsed_seconds=time.time() - started)
    save_json(output / "metrics.json", metrics)
    summary = {"cell_type": args.cell, "epoch_zero_based": selected["epoch_zero_based"],
               "n_regions": metrics["n_regions"], "within_region_pearson_mean": metrics["within_region_raw"]["pearson"]["mean"],
               "within_region_spearman_mean": metrics["within_region_raw"]["spearman"]["mean"],
               "within_region_pearson_log1p_mean": metrics["within_region_log1p"]["pearson"]["mean"],
               "across_regions_sum_pearson": metrics["across_regions_sum_raw"]["pearson"],
               "across_regions_sum_spearman": metrics["across_regions_sum_raw"]["spearman"],
               "across_regions_sum_log1p_pearson": metrics["across_regions_sum_log1p"]["pearson"],
               "pooled_bins_pearson": metrics["pooled_bins_raw"]["pearson"],
               "pooled_bins_spearman": metrics["pooled_bins_raw"]["spearman"],
               "poisson_loss_mean": metrics["poisson_loss_mean"]}
    pd.DataFrame([summary]).to_csv(output / "metrics.tsv", sep="\t", index=False, na_rep="NA")
    provenance.update(completed_utc=datetime.now(timezone.utc).isoformat(),
                      target_values_exactly_match_extracted_test=True,
                      output_sha256={path.name: sha256(path) for path in output.glob("expression_*.npy")})
    save_json(output / "provenance.json", provenance)
    (output / "SUCCESS").write_text(provenance["completed_utc"] + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    print(f"Completed and saved all test predictions: {output}", flush=True)


if __name__ == "__main__":
    main()
