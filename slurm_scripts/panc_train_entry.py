"""Entry point for panc_pretrain.sh arms that need the sequence-only callback.

Registers tests/cismamba_sequence_only_protocol.SequenceOnlyProtocol (100% accessibility input
masking with an independent 25% / 500-bp loss mask), then runs train.main() with the same
command-line Hydra overrides as `python -m train`.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)) #repo root, so train and tests import after hydra changes cwd

import train
from src.utils import registry

registry.callbacks['sequence_only_protocol'] = 'tests.cismamba_sequence_only_protocol.SequenceOnlyProtocol'

if __name__ == '__main__':
    #imported (not run as a script), hydra would look for configs as a python module, so point it at the directory
    sys.argv[1:1] = ['--config-path', str(ROOT / 'configs'), '--config-name', 'config']
    train.main()
