#!/usr/bin/env python3
"""CD4/NK copies of the October 7 wide256 recipe, with ordinary dated run paths.

Usage (from the repository's pixi Python environment):
  python slurm_scripts/cismamba_cd4_nk_wide256.py prepare --bundle /absolute/path
  python slurm_scripts/cismamba_cd4_nk_wide256.py audit --manifest /path/manifest.json
  python slurm_scripts/cismamba_cd4_nk_wide256.py submit --manifest /path/manifest.json

Each Slurm stage first checks real-data training and complete-state resume in an
isolated smoke directory. Production checkpoints live directly in
outputs/YYYY-MM-DD/HH-MM-SS-microseconds/checkpoints/. Requeued stages reuse their
reserved output directory and restore the newest epoch/recovery checkpoint.
"""
import argparse
from collections import Counter
import contextlib
import copy
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.cismamba_controls import check_checkpoint, write_json

REFERENCES = {
    'pretrain': ROOT / 'outputs/cismamba_tuning_20260928/training/cosine/config.json',
    'finetune': ROOT / 'tests/results/cismamba_rna_tuning_20261005/rna_wd001/planned_config.json',
}
REFERENCE_MANIFEST = ROOT / 'outputs/cismamba_rna_tuning_20261005/manifest.json'
CELL_MANIFEST = ROOT / 'outputs/cage_finetuning/20261003_cd4_nk/experiment.json'
BUDGETS = {'pretrain': 30, 'finetune': 15}
STAGES = tuple(BUDGETS)
CELLS = ('CD4', 'NK')


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def reserve_output():
    while True:
        now = datetime.now()
        path = ROOT / 'outputs' / now.strftime('%Y-%m-%d') / now.strftime('%H-%M-%S-%f')
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue


def check_sources(manifest):
    for name, expected in manifest['source_sha256'].items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f'Source changed since preparation: {name}')
    for record in manifest['references'].values():
        assert sha256(record['snapshot']) == record['sha256'], record
    for name, expected in manifest['data_files'].items():
        stat = Path(name).stat()
        assert [stat.st_size, stat.st_mtime_ns] == expected, f'Data file changed: {name}'


def prepare(bundle):
    bundle = bundle.resolve()
    bundle.mkdir(parents=True, exist_ok=False)
    old = read_json(CELL_MANIFEST)
    recorded = read_json(REFERENCE_MANIFEST)['sources_sha256']
    core = {k: v for k, v in recorded.items() if k.startswith('src/') or k == 'train.py'}
    for name, expected in core.items():
        assert sha256(ROOT / name) == expected, f'Current source differs from recommended run: {name}'
    files = set(core) | {
        'src/dataloaders/datasets/general_dataset.py', 'src/dataloaders/genomics.py',
        'src/dataloaders/utils/mask_seq.py', 'src/tasks/tasks.py', 'src/tasks/metrics.py',
        'src/utils/registry.py', 'tests/cismamba_controls.py', 'tests/bench_models.py',
        'tests/bench_utils.py', 'evals/evals_utils_joint.py', 'pixi.lock',
        'slurm_scripts/cismamba_cd4_nk_wide256.py',
        'slurm_scripts/cismamba_cd4_nk_wide256.sh',
    }
    manifest = {
        'created_utc': utc_now(), 'bundle': str(bundle),
        'recommendation': str(ROOT / 'claude_summaries/cismamba_rna_tuning_20261005.md'),
        'source_sha256': {name: sha256(ROOT / name) for name in sorted(files)},
        'reference_core_hashes_match': True, 'references': {}, 'cells': {},
        'data_dir': old['data_dir'], 'genome_seq': old['genome_seq'],
        'sequences_bed': old['sequences_bed'], 'source_zarr': old['source_zarr'],
        'budgets': BUDGETS, 'data_files': {},
        'selection': 'minimum filename validation loss within budget; earlier epoch on ties',
    }
    for stage, path in REFERENCES.items():
        snapshot = bundle / 'references' / f'{stage}.json'
        write_json(snapshot, read_json(path))
        manifest['references'][stage] = {
            'original': str(path), 'snapshot': str(snapshot), 'sha256': sha256(snapshot),
        }
    paths = [old['genome_seq'], old['sequences_bed'],
             str(Path(old['data_dir']) / 'extraction_manifest.json')]
    for cell in CELLS:
        entry = old['cells'][cell]
        paths.append(entry['dnase_npz'])
        runs = {stage: str(reserve_output()) for stage in STAGES}
        manifest['cells'][cell] = {
            'accessibility': entry['dnase_npz'], 'assay': 'DNase',
            'cage': str(Path(old['data_dir']) / f'{cell}_CAGE.zarr'),
            'source_track': entry['track_index'], 'track_identifier': entry['track_identifier'],
            'runs': runs,
        }
        for stage, output in runs.items():
            write_json(Path(output) / 'run_info.json', {
                'cell': cell, 'stage': stage, 'manifest': str(bundle / 'manifest.json'),
                'reference': manifest['references'][stage]['original'],
                'label': f'cismamba_wide256_{cell.lower()}_{stage}',
                'accessibility_assay': 'DNase',
            })
    for name in paths:
        stat = Path(name).stat()
        manifest['data_files'][name] = [stat.st_size, stat.st_mtime_ns]
    write_json(bundle / 'manifest.json', manifest)
    for cell in CELLS:
        for stage in STAGES:
            cfg = configuration(manifest, cell, stage, resume=False)
            write_json(Path(manifest['cells'][cell]['runs'][stage]) / 'planned_config.json', cfg)
    write_index(manifest, {})
    print(bundle / 'manifest.json', flush=True)


def configuration(manifest, cell, stage, *, output=None, smoke_epochs=None, resume=True):
    cfg = read_json(manifest['references'][stage]['snapshot'])
    entry = manifest['cells'][cell]
    out = Path(output or entry['runs'][stage])
    cfg['dataset']['data_path'] = entry['accessibility']
    if stage == 'finetune':
        cfg['dataset'].update(additional_data=entry['cage'], additional_data_idxs=0)
    cfg['trainer'].update(max_epochs=BUDGETS[stage], enable_progress_bar=False)
    cfg['train']['pretrained_model_path'] = (
        str(Path(entry['runs']['pretrain']) / 'checkpoints/best.ckpt')
        if stage == 'finetune' else None)
    # Use the fresh-transfer hook from the audited planned RNA configuration.
    cfg['train']['pretrained_model_state_hook'] = (
        copy.deepcopy(read_json(manifest['references']['finetune']['snapshot'])['train']['pretrained_model_state_hook'])
        if stage == 'finetune' else {'_name_': None})
    cfg['train']['ckpt'] = str(out / 'checkpoints/last.ckpt')
    candidates = [p for p in (out / 'checkpoints/last.ckpt', out / 'recovery/last.ckpt') if p.exists()]
    if resume and candidates:
        cfg['train']['ckpt'] = str(max(candidates, key=lambda p: p.stat().st_mtime_ns))
        cfg['train']['pretrained_model_path'] = None
        cfg['train']['pretrained_model_state_hook'] = {'_name_': None}
    if smoke_epochs is not None:
        cfg['trainer'].update(max_epochs=smoke_epochs, limit_train_batches=16, limit_val_batches=2)
        cfg['callbacks']['model_checkpoint_every_n_steps']['every_n_train_steps'] = 1
    cfg['hydra'] = {'run': {'dir': str(out)}, 'job': {'chdir': True}}
    return cfg


def config_diff(before, after, prefix=''):
    changes = {}
    for key in sorted(before.keys() | after.keys()):
        name = f'{prefix}.{key}' if prefix else key
        if key not in before or key not in after:
            changes[name] = [before.get(key), after.get(key)]
        elif isinstance(before[key], dict) and isinstance(after[key], dict):
            changes.update(config_diff(before[key], after[key], name))
        elif before[key] != after[key]:
            changes[name] = [before[key], after[key]]
    return changes


def validate_recipe(manifest, cell, stage, cfg):
    base = read_json(manifest['references'][stage]['snapshot'])
    changes = config_diff(base, {k: v for k, v in cfg.items() if k != 'hydra'})
    allowed = {'dataset.data_path', 'train.ckpt', 'train.pretrained_model_path',
               'trainer.enable_progress_bar'}
    if stage == 'finetune':
        allowed |= {'dataset.additional_data', 'dataset.additional_data_idxs'}
    assert set(changes) <= allowed, changes
    assert cfg['model']['config']['d_in'] == cfg['model']['config']['d_model'] == 256
    assert cfg['dataset']['batch_size'] == cfg['trainer']['devices'] == 1
    assert cfg['trainer']['accumulate_grad_batches'] == 8
    assert cfg['trainer']['max_epochs'] == BUDGETS[stage]
    assert cfg['dataset']['data_path'] == manifest['cells'][cell]['accessibility']
    if stage == 'finetune':
        assert cfg['dataset']['additional_data'] == manifest['cells'][cell]['cage']
        assert cfg['train']['pretrained_model_state_hook'] == {
            '_name_': 'load_full_model', 'freeze_backbone': False,
            'load_encoder': True, 'load_decoder': False,
        }
    else:
        assert cfg['train']['pretrained_model_path'] is None
    out = Path(cfg['hydra']['run']['dir']).relative_to(ROOT / 'outputs')
    assert len(out.parts) == 2
    assert re.fullmatch(r'\d{4}-\d{2}-\d{2}/\d{2}-\d{2}-\d{2}-\d{6}', str(out))
    return changes


def validate_data(manifest):
    import numpy as np
    import zarr
    import zipfile
    data = Path(manifest['data_dir'])
    assert (data / 'SUCCESS').is_file()
    extraction = read_json(data / 'extraction_manifest.json')
    assert extraction['status'] == 'complete'
    assert extraction['source_zarr'] == manifest['source_zarr']
    assert sha256(manifest['sequences_bed']) == extraction['sequences_bed_sha256']
    assert sha256(data / 'sequences_enformer.bed') == extraction['sequences_bed_sha256']
    splits = Counter(line.split()[3] for line in Path(manifest['sequences_bed']).read_text().splitlines())
    assert splits == {'train': 34021, 'valid': 2213, 'test': 1937}, splits
    updates = math.ceil(splits['train'] / 8)
    assert read_json(manifest['references']['pretrain']['snapshot'])['scheduler']['num_training_steps'] == 30 * updates
    for cell, entry in manifest['cells'].items():
        print(f'Checking {cell} CAGE labels and accessibility headers...', flush=True)
        track = extraction['tracks'][cell]
        assert (track['source_index'], track['identifier'], track['output_index']) == (
            entry['source_track'], entry['track_identifier'], 0)
        group = zarr.open_group(entry['cage'], mode='r')
        for split, info in extraction['splits'].items():
            array = group[split]
            assert list(array.shape) == info['output_shape'] and array.dtype == np.dtype('float32')
            sample = np.asarray(array[:2])
            assert np.isfinite(sample).all() and (sample >= 0).all()
        # Inspect only NPZ headers, avoiding multi-GB chromosome reads on the login node.
        with zipfile.ZipFile(entry['accessibility']) as archive:
            assert 'chr1.npy' in archive.namelist()
            for name in archive.namelist():
                with archive.open(name) as handle:
                    version = np.lib.format.read_magic(handle)
                    reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                              else np.lib.format.read_array_header_2_0)
                    shape, _, dtype = reader(handle)
                    assert len(shape) == 2 and shape[0] == 1 and dtype.kind == 'f', (name, shape, dtype)
    return {'splits': dict(splits), 'updates_per_epoch': updates, 'cosine_horizon': 30 * updates}


def build_model(cfg, source=None):
    import torch
    from omegaconf import OmegaConf
    from train import SequenceLightningModule
    local = copy.deepcopy(cfg)
    local.pop('hydra', None)
    local['train']['disable_dataset'] = True
    torch.manual_seed(2222)
    model = SequenceLightningModule(OmegaConf.create(local))
    if source is not None:
        fresh = {k: v.clone() for k, v in model.state_dict().items() if k.startswith('decoder.')}
        checkpoint = torch.load(source, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        for key, value in model.state_dict().items():
            expected = fresh[key] if key.startswith('decoder.') else checkpoint['state_dict'][key]
            assert torch.equal(value, expected), key
        omitted = set(checkpoint['state_dict']) - set(model.state_dict())
        assert any(k.startswith('model.backbone.global_up.') for k in omitted)
        assert all(k.startswith(('model.backbone.global_up.', 'decoder.')) for k in omitted)
    assert all(p.requires_grad for p in model.parameters())
    return model


def check_modules(model, stage):
    import torch
    from src.models.nn.sampling import ConvBlock
    from src.tasks.decoders import EnformerDecoder
    backbone = model.model.backbone
    assert len(backbone.hydra_blocks) == len(backbone.transformer_blocks) == 8
    assert backbone.global_pooling == 128 and backbone.transformer_pooling == 1
    assert all(b.mixer.d_state == b.mixer.headdim == 64 for b in backbone.hydra_blocks)
    assert all(b.attn.use_gating and b.attn.use_rope and b.attn.use_flash_attn
               and b.attn.num_heads == 4 and b.attn.multiplier == 4 for b in backbone.transformer_blocks)
    assert {type(m.norm).__name__ for m in backbone.modules() if isinstance(m, ConvBlock)} == {'RMSBatchNorm1d'}
    if stage == 'finetune':
        head = next(m for m in model.decoder.modules() if isinstance(m, EnformerDecoder))
        assert backbone.global_up is None and backbone.global_down.skip_count == 0
        assert backbone.output_resolution == head.downsampled == 128
        assert head.bin_size == 1 and head.yshape == 896 and head.output_transform.out_features == 1
    else:
        assert backbone.global_up is not None and backbone.output_resolution == 1
    length = 896 if stage == 'finetune' else 128
    with torch.no_grad():
        prediction, _ = model.decoder(torch.zeros(1, length, 256))
    if stage == 'pretrain':
        prediction = prediction[1]
    assert prediction.shape == (1, length, 1) and torch.isfinite(prediction).all()


def save_yaml(path, cfg):
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def train_command(path):
    return [sys.executable, '-B', '-u', '-m', 'train', '--config-path', str(path.parent), '--config-name', path.stem]


def preflight_modules():
    # train registers OmegaConf resolvers unconditionally; Evals registers them
    # defensively. Import in this order in both the CPU audit and GPU preflight.
    import train
    from tests.bench_models import make_joint_batch
    from src.tasks.metrics import joint_loss
    from evals.evals_utils_joint import Evals
    return make_joint_batch, joint_loss, Evals


def audit(manifest):
    import yaml
    check_sources(manifest)
    bundle = Path(manifest['bundle'])
    data = validate_data(manifest)
    preflight_modules()
    print('Data and split checks passed; resolving the four training configurations.', flush=True)
    rows = {}
    for cell in CELLS:
        for stage in STAGES:
            cfg = configuration(manifest, cell, stage, resume=False)
            changes = validate_recipe(manifest, cell, stage, cfg)
            path = save_yaml(bundle / 'checks' / f'{cell}_{stage}.yaml', cfg)
            result = subprocess.run(train_command(path) + ['--cfg', 'job', '--resolve'],
                                    cwd=ROOT, text=True, capture_output=True, check=True, timeout=180)
            resolved = yaml.safe_load(result.stdout)
            assert resolved == {k: v for k, v in cfg.items() if k != 'hydra'}
            rows[f'{cell}/{stage}'] = changes
            print(f'{cell}/{stage}: resolved configuration matches the reference.', flush=True)
    for stage in STAGES:
        cfg = configuration(manifest, 'CD4', stage, resume=False)
        source = read_json(manifest['references']['finetune']['snapshot'])['train']['pretrained_model_path'] if stage == 'finetune' else None
        with (bundle / 'checks' / f'{stage}_module_audit.log').open('w') as handle, contextlib.redirect_stdout(handle):
            model = build_model(cfg, source)
            check_modules(model, stage)
        del model
        gc.collect()
    write_json(bundle / 'CPU_AUDIT_PASSED.json', {
        'status': 'passed', 'checked_utc': utc_now(), 'reference_config_differences': rows,
        'model_encoder_decoder_optimizer_scheduler_loss_match_references': True,
        'module_defaults_match': True, 'pooled_transfer_exact': True, **data,
    })
    print('All four resolved configs, data/splits, module settings, and pooled transfer passed.', flush=True)


def run_fit(manifest, cell, stage, cfg, label, log=None):
    check_sources(manifest)
    attempt = f'{os.environ.get("SLURM_JOB_ID", "local")}_{os.environ.get("SLURM_RESTART_COUNT", "0")}'
    path = save_yaml(Path(manifest['bundle']) / 'configs' / cell / stage / f'{attempt}_{label}.yaml', cfg)
    if log:
        with log.open('w') as handle:
            subprocess.run(train_command(path), cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                           check=True, timeout=1200)
    else:
        subprocess.run(train_command(path), cwd=ROOT, check=True)


def check_optimizer(checkpoint, stage, steps):
    import torch
    state = torch.load(checkpoint, map_location='cpu')
    assert state['global_step'] == steps
    groups = state['optimizer_states'][0]['param_groups']
    lr = 1e-3 * steps / 1000 if stage == 'pretrain' else 1e-4
    wd = 0.1 if stage == 'pretrain' else 0.01
    assert all(abs(g['lr'] - lr) < 1e-12 for g in groups)
    decays = {g['weight_decay'] for g in groups}
    assert wd in decays and decays <= {0., wd}
    if stage == 'pretrain':
        assert state['lr_schedulers'][0]['last_epoch'] == steps
    assert max(float(s['step']) for s in state['optimizer_states'][0]['state'].values()) == steps
    assert state['loops'] and state['lr_schedulers']
    return state


def preflight(manifest, cell, stage):
    import torch
    make_joint_batch, joint_loss, Evals = preflight_modules()
    bundle = Path(manifest['bundle'])
    proof = bundle / 'checks' / cell / stage
    proof.mkdir(parents=True, exist_ok=True)
    marker = proof / 'GPU_PREFLIGHT_PASSED.json'
    if marker.exists():
        assert read_json(marker)['status'] == 'passed'
        return
    props = torch.cuda.get_device_properties(0)
    assert 'A100' in props.name and props.total_memory > 70 * 1024**3, props
    cfg = configuration(manifest, cell, stage, resume=False)
    source = cfg['train']['pretrained_model_path']
    with (proof / 'initialization.log').open('w') as handle, contextlib.redirect_stdout(handle):
        model = build_model(cfg, source)
        check_modules(model, stage)
    model = model.cuda().train()
    batch = make_joint_batch(524288, acc_mask_frac=0.25 if stage == 'pretrain' else 0)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        hidden, intermediate = model.encoder(*batch[:2])
        hidden, _ = model.model(hidden)
        if stage == 'pretrain':
            prediction, _ = model.decoder(hidden, intermediates=intermediate)
            assert prediction[1].shape == (1, 524288, 1)
            loss = joint_loss(prediction, batch[2:], task1='mlm', task2='reg')
        else:
            prediction, _ = model.decoder(hidden, intermediates=intermediate)
            assert prediction.shape == (1, 896, 1)
            loss = model.loss(prediction, torch.rand_like(prediction))
    assert torch.isfinite(loss).all()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    del model, batch, hidden, intermediate, prediction, loss
    gc.collect()
    torch.cuda.empty_cache()
    attempt = f'{os.environ.get("SLURM_JOB_ID", "local")}_{os.environ.get("SLURM_RESTART_COUNT", "0")}'
    smoke = bundle / 'smoke' / cell / stage / attempt
    assert not smoke.exists(), f'Smoke attempt already exists: {smoke}'
    for epochs in (1, 2):
        cfg = configuration(manifest, cell, stage, output=smoke, smoke_epochs=epochs)
        if epochs == 2:
            assert cfg['train']['pretrained_model_path'] is None
            assert cfg['train']['pretrained_model_state_hook'] == {'_name_': None}
            saved = torch.load(cfg['train']['ckpt'], map_location='cpu')
            with (proof / 'resume_load.log').open('w') as handle, contextlib.redirect_stdout(handle):
                resumed = build_model(cfg)
                resumed.load_state_dict(saved['state_dict'], strict=True)
            assert all(torch.equal(v, saved['state_dict'][k]) for k, v in resumed.state_dict().items())
            del saved, resumed
            gc.collect()
        run_fit(manifest, cell, stage, cfg, f'smoke_e{epochs}', proof / f'smoke_e{epochs}_{attempt}.log')
        check_checkpoint(smoke / 'checkpoints/last.ckpt', epochs * 2)
        check_optimizer(smoke / 'checkpoints/last.ckpt', stage, epochs * 2)
    evaluator = Evals(str(smoke / 'checkpoints/last.ckpt'), dataset=object(), device='cuda')
    batch = make_joint_batch(524288, acc_mask_frac=0.25 if stage == 'pretrain' else 0)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        hidden, intermediate = evaluator.encoder(*batch[:2])
        hidden, _ = evaluator.backbone(hidden)
        if stage == 'pretrain':
            prediction = evaluator.decoder(hidden)[1]
        else:
            prediction = evaluator.decoder(hidden, intermediates=intermediate)
        assert prediction.shape == (1, 524288 if stage == 'pretrain' else 896, 1)
        assert torch.isfinite(prediction).all()
    del evaluator, batch, hidden, intermediate, prediction
    gc.collect()
    torch.cuda.empty_cache()
    write_json(marker, {'status': 'passed', 'checked_utc': utc_now(), 'source_checkpoint': source,
                        'full_context_finite_gradients': True, 'strict_evaluator_load': True,
                        'all_resume_tensors_exact': True, 'optimizer_updates': [2, 4],
                        'optimizer_recipe_verified': True, 'smoke_output': str(smoke)})
    print(f'{cell}/{stage}: GPU preflight and complete-state resume passed.', flush=True)


def select_checkpoint(output, budget):
    rows = []
    for path in (Path(output) / 'checkpoints').glob('*-val_loss=*.ckpt'):
        match = re.fullmatch(r'(\d+)-val_loss=(.+)\.ckpt', path.name)
        if match and int(match[1]) < budget:
            loss = float(match[2])
            assert math.isfinite(loss), path
            rows.append((loss, int(match[1]), path))
    assert len(rows) == budget and {r[1] for r in rows} == set(range(budget)), (budget, rows)
    loss, epoch, path = min(rows)
    return path, {'checkpoint': str(path), 'val_loss': loss, 'epoch_zero_based': epoch,
                  'budget_epochs': budget, 'selection': 'minimum filename val_loss; earlier epoch on ties'}


def stage_run(manifest, cell, stage):
    check_sources(manifest)
    bundle = Path(manifest['bundle'])
    assert read_json(bundle / 'CPU_AUDIT_PASSED.json')['status'] == 'passed'
    validate_data(manifest)
    output = Path(manifest['cells'][cell]['runs'][stage])
    owner = read_json(output / 'run_info.json')
    assert (owner['cell'], owner['stage'], owner['manifest']) == (cell, stage, str(bundle / 'manifest.json'))
    if (output / 'COMPLETED.json').exists():
        print(f'{cell}/{stage} already completed: {output}', flush=True)
        return
    if stage == 'finetune':
        parent = Path(manifest['cells'][cell]['runs']['pretrain'])
        done = read_json(parent / 'COMPLETED.json')
        assert done['status'] == 'completed' and done['budget_epochs'] == 30
        assert sha256(parent / 'checkpoints/best.ckpt') == done['selected_checkpoint_sha256']
    preflight(manifest, cell, stage)
    cfg = configuration(manifest, cell, stage)
    if Path(cfg['train']['ckpt']).exists():
        check_checkpoint(cfg['train']['ckpt'], 1)
        print(f'Resuming {cell}/{stage}: {cfg["train"]["ckpt"]}', flush=True)
    else:
        assert not (output / 'checkpoints').exists(), f'Refusing ambiguous production directory: {output}'
    run_fit(manifest, cell, stage, cfg, 'production')
    completed = check_checkpoint(output / 'checkpoints/last.ckpt', BUDGETS[stage] * 4253)
    assert completed['epoch'] + 1 >= BUDGETS[stage]
    actual = read_json(output / 'config.json')
    for key in ('model', 'encoder', 'decoder', 'dataset', 'optimizer', 'scheduler', 'loader', 'task'):
        assert actual[key] == cfg[key], key
    selected, record = select_checkpoint(output, BUDGETS[stage])
    check_checkpoint(selected, 1)
    write_json(output / 'selected_checkpoint.json', record)
    link = output / 'checkpoints/best.ckpt'
    if link.is_symlink():
        assert link.resolve() == selected.resolve()
    else:
        assert not link.exists()
        link.symlink_to(selected.name)
    write_json(output / 'COMPLETED.json', {
        **completed, **record, 'status': 'completed', 'completed_utc': utc_now(),
        'selected_checkpoint_sha256': sha256(selected), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
    })
    print(f'{cell}/{stage} completed. Selected checkpoint: {selected}', flush=True)


def write_index(manifest, jobs):
    with (Path(manifest['bundle']) / 'runs.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['cell', 'stage', 'output dir', 'label', 'job_id', 'pretrained run'])
        for cell in CELLS:
            for stage in STAGES:
                entry = manifest['cells'][cell]
                writer.writerow([cell, stage, str(Path(entry['runs'][stage]).relative_to(ROOT)),
                                 f'cismamba_wide256_{cell.lower()}_{stage}',
                                 jobs.get(f'{cell}/{stage}', {}).get('job_id', ''),
                                 str(Path(entry['runs']['pretrain']).relative_to(ROOT)) if stage == 'finetune' else ''])


def submit(manifest):
    check_sources(manifest)
    bundle = Path(manifest['bundle'])
    assert read_json(bundle / 'CPU_AUDIT_PASSED.json')['status'] == 'passed'
    path = bundle / 'submission.json'
    jobs = read_json(path) if path.exists() else {}
    (bundle / 'logs').mkdir(exist_ok=True)
    for cell in CELLS:
        for stage in STAGES:
            key = f'{cell}/{stage}'
            if key in jobs:
                print(f'Already submitted {key}: {jobs[key]["job_id"]}', flush=True)
                continue
            cmd = ['sbatch', '--parsable', f'--job-name=wide256_{cell.lower()}_{stage}',
                   f'--output={bundle}/logs/{cell}_{stage}_%j.out']
            if stage == 'finetune':
                cmd += [f'--dependency=afterok:{jobs[f"{cell}/pretrain"]["job_id"]}', '--kill-on-invalid-dep=yes']
            cmd += [str(ROOT / 'slurm_scripts/cismamba_cd4_nk_wide256.sh'),
                    str(bundle / 'manifest.json'), cell, stage]
            result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
            job = result.stdout.strip().split(';')[0]
            assert job.isdigit(), result.stdout
            jobs[key] = {'job_id': job, 'submitted_utc': utc_now(), 'command': cmd,
                         'output': manifest['cells'][cell]['runs'][stage]}
            write_json(path, jobs)
            write_index(manifest, jobs)
            print(f'Submitted {key}: {job} -> {jobs[key]["output"]}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'audit', 'submit', 'stage'])
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--cell', choices=CELLS)
    parser.add_argument('--stage', choices=STAGES)
    args = parser.parse_args()
    if args.action == 'prepare':
        if args.bundle is None:
            parser.error('prepare requires --bundle')
        prepare(args.bundle)
        return
    if args.manifest is None:
        parser.error('--manifest is required')
    manifest = read_json(args.manifest.resolve())
    if args.action == 'audit':
        audit(manifest)
    elif args.action == 'submit':
        submit(manifest)
    else:
        if args.cell is None or args.stage is None:
            parser.error('stage requires --cell and --stage')
        stage_run(manifest, args.cell, args.stage)


if __name__ == '__main__':
    main()
