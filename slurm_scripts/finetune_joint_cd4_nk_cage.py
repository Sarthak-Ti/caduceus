#!/usr/bin/env python3
"""Start the existing CAGE fine-tuning entrypoint with explicit per-cell inputs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_overrides(config, cell, phase, workers=7, resume=False):
    if resume and phase != "train":
        raise ValueError("Only full training can be resumed")
    entry = config["cells"][cell]
    output = Path(config["run_dir"]) / cell / phase
    # JSON quoting here is Hydra argument quoting, NOT shell interpolation.
    quote = json.dumps
    overrides = [
        "experiment=hg38/joint_finetune", "model=caduceus",
        "model._name_=dna_embedding_caduceus", "model.config.d_model=256",
        "model.config.n_layer=16", "model.config.bidirectional=true",
        "model.config.bidirectional_strategy=add", "model.config.bidirectional_weight_tie=true",
        "model.config.rcps=false", "model.config.vocab_size=1",
        "model.config.pad_vocab_size_multiple=1", "+model.config.skip_embedding=true",
        "encoder._name_=jointcnn", "encoder.joint=false", "encoder.combine=true",
        "+encoder.d_input1=6", "+encoder.d_input2=2",
        "dataset.acc_type=continuous", "dataset.length=524288", "dataset.batch_size=1",
        "dataset.batch_size_eval=2", f"dataset.num_workers={workers}",
        f"loader.num_workers={workers}", "dataset.load_in=false",
        "dataset.shift_sequences=0", "dataset.rc_aug=false", "dataset.mlm=0", "dataset.acc_mlm=0",
        "+dataset.mask_only=true",
        f"dataset.data_path={quote(entry['dnase_npz'])}",
        f"+dataset.genome_seq_file={quote(config['genome_seq'])}",
        f"+dataset.sequences_bed_file={quote(str(Path(config['data_dir']) / 'sequences_enformer.bed'))}",
        f"dataset.additional_data={quote(str(Path(config['data_dir']) / f'{cell}_CAGE.zarr'))}",
        # Each extracted store contains one CAGE track at index 0.
        "+dataset.additional_data_idxs=0",
        "decoder._name_=enformer", "+decoder.d_model=256", "+decoder.d_output=1",
        "+decoder.conjoin_train=false", "+decoder.conjoin_test=false", "+decoder.convolutions=true",
        "+decoder.yshape=114688", "+decoder.bin_size=128",
        f"optimizer.lr={config['learning_rate']}",
        "train.pretrained_model_strict_load=false",
        "train.pretrained_model_path=null" if resume else
        f"train.pretrained_model_path={quote(entry['pretrained_checkpoint'])}",
        "train.pretrained_model_state_hook.load_encoder=true",
        "+train.pretrained_model_state_hook.load_decoder=false",
        "train.pretrained_model_state_hook.freeze_backbone=false",
        "+train.pretrained_model_state_hook.freeze_encoder=false",
        # train.py unconditionally passes this value to fsspec in a log statement:
        # null raises there. A nonexistent path in a NEW run has the intended fresh-start semantics.
        f"train.ckpt={quote(str(output / 'checkpoints/last.ckpt'))}" if resume else
        "train.ckpt=checkpoints/last.ckpt", "train.test=false",
        "+train.remove_test_loader_in_eval=true", f"train.seed={config['seed']}",
        "trainer.devices=2", "trainer.num_nodes=1", "trainer.precision=bf16",
        "trainer.accumulate_grad_batches=4", f"trainer.max_epochs={config['max_epochs']}",
        "trainer.num_sanity_val_steps=0", "trainer.limit_train_batches=1.0",
        "trainer.limit_val_batches=1.0", "callbacks.model_checkpoint.save_top_k=-1",
        "callbacks.model_checkpoint.save_last=true",
        f"wandb.name=cismamba_{cell.lower()}_cage_20261003", "wandb.group=cd4_nk_cage_finetune",
        "wandb.mode=offline", f"hydra.run.dir={quote(str(output))}",
    ]
    if resume:
        # load_state_dict applies the pretraining hook even during Trainer.fit
        # restoration. Disable it so the trained CAGE decoder is also restored.
        overrides += ["train.pretrained_model_state_hook._name_=null"]
    if phase == "smoke":
        overrides += ["+trainer.fast_dev_run=1", "wandb=null", "trainer.num_sanity_val_steps=0"]
    return overrides


def validate_resume(config, cell):
    import torch

    path = Path(config["run_dir"]) / cell / "train/checkpoints/last.ckpt"
    checkpoint = torch.load(path, map_location="cpu")
    for key in ["state_dict", "optimizer_states", "lr_schedulers", "loops"]:
        if not checkpoint.get(key):
            raise ValueError(f"Cannot resume: checkpoint has no {key}: {path}")
    if not any(key.startswith("decoder.") for key in checkpoint["state_dict"]):
        raise ValueError("Resume checkpoint does not contain the trained CAGE decoder")
    dataset = checkpoint["hyper_parameters"]["dataset"]
    expected = {
        "data_path": config["cells"][cell]["dnase_npz"],
        "additional_data": str(Path(config["data_dir"]) / f"{cell}_CAGE.zarr"),
        "additional_data_idxs": 0,
    }
    for key, value in expected.items():
        if dataset[key] != value:
            raise ValueError(f"Resume checkpoint has unexpected {key}: {dataset[key]}")
    if any(not torch.isfinite(value).all().item()
           for value in checkpoint["state_dict"].values()
           if torch.is_tensor(value) and value.is_floating_point()):
        raise ValueError("Resume checkpoint contains nonfinite weights")
    return {"checkpoint": str(path), "sha256": file_hash(path),
            "epoch": checkpoint["epoch"], "global_step": checkpoint["global_step"],
            "optimizer_states": len(checkpoint["optimizer_states"]),
            "scheduler_states": len(checkpoint["lr_schedulers"])}


def validate_inputs(config, cell):
    import numpy as np
    import zarr

    data = Path(config["data_dir"])
    if not (data / "SUCCESS").is_file():
        raise RuntimeError("Extraction has not passed validation")
    manifest = json.loads((data / "extraction_manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["source_zarr"] != config["source_zarr"]:
        raise ValueError("Unexpected extraction manifest")
    entry = config["cells"][cell]
    if manifest["tracks"][cell]["source_index"] != entry["track_index"]:
        raise ValueError("Wrong CAGE target")
    if file_hash(data / "sequences_enformer.bed") != manifest["sequences_bed_sha256"]:
        raise ValueError("Sequence BED changed after extraction")
    if file_hash(entry["pretrained_checkpoint"]) != entry["pretrained_checkpoint_sha256"]:
        raise ValueError("Pretraining checkpoint changed")
    for path in [entry["dnase_npz"], config["genome_seq"]]:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    group = zarr.open_group(str(data / f"{cell}_CAGE.zarr"), mode="r")
    for split, info in manifest["splits"].items():
        array = group[split]
        if list(array.shape) != info["output_shape"] or np.dtype(array.dtype) != np.dtype("float32"):
            raise ValueError(f"Unexpected label dimensions/dtype for {cell}/{split}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("cell", choices=["CD4", "NK"])
    parser.add_argument("phase", choices=["smoke", "train", "config"])
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--resume", action="store_true",
                        help="Restore full training state from this cell's train/checkpoints/last.ckpt")
    args = parser.parse_args()
    if args.resume and args.phase == "smoke":
        parser.error("--resume is supported only for train or config")
    config = json.loads(args.config.read_text())
    phase = "train" if args.phase == "config" else args.phase
    overrides = make_overrides(config, args.cell, phase, args.workers, args.resume)
    resume_info = validate_resume(config, args.cell) if args.resume else None
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    if args.phase == "config":
        overrides += ["+train.print_config_only=true", "train.gpu_mem=80",
                      f"hydra.run.dir={json.dumps(str(Path(config['run_dir']) / args.cell / ('resume_config_check' if args.resume else 'config_check')))}"]
    else:
        validate_inputs(config, args.cell)
        if phase == "train":
            smoke_passed = Path(config["run_dir"]) / args.cell / "SMOKE_PASSED"
            if not smoke_passed.exists():
                raise RuntimeError("End-to-end GPU smoke test must pass before full training")
        output = Path(config["run_dir"]) / args.cell / phase
        if not args.resume and (output / "checkpoints").exists():
            raise FileExistsError(f"Refusing to resume or overwrite existing training: {output}")
    if rank == 0:
        record = Path(config["run_dir"]) / args.cell
        record.mkdir(parents=True, exist_ok=True)
        if args.resume and args.phase == "train":
            attempt = os.environ.get("SLURM_JOB_ID", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
            record = record / f"resume_{attempt}"
            record.mkdir(exist_ok=False)
            # Preserve the previous config and exact restart checkpoint before
            # Hydra and future epoch saves update the existing training folder.
            shutil.copytree(output / ".hydra", record / "previous_hydra")
            shutil.copy2(resume_info["checkpoint"], record / "starting.ckpt")
            resume_info["recorded_utc"] = datetime.now(timezone.utc).isoformat()
            (record / "resume.json").write_text(json.dumps(resume_info, indent=2) + "\n")
        suffix = "_resume" if args.resume else ""
        (record / f"{args.phase}{suffix}_overrides.json").write_text(json.dumps(overrides, indent=2) + "\n")
    repo = Path(config["caduceus_repo"])
    os.chdir(repo)
    command = [str(repo / ".pixi/envs/default/bin/python"), "-B", "-u", "-m", "train", *overrides]
    print(f"Starting {args.cell}/{args.phase}; original CAGE track={config['cells'][args.cell]['track_index']}", flush=True)
    if resume_info:
        print(f"Resuming complete CAGE training state: {json.dumps(resume_info)}", flush=True)
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
