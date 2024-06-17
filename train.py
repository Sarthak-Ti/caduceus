import copy
import os
import random
import time
from functools import partial, wraps
from typing import Callable, List, Sequence

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn
from pytorch_lightning.strategies.ddp import DDPStrategy
from tqdm.auto import tqdm
# from pytorch_lightning.strategies.ddp import DDPStrategy #literally just repeated 2 lines above

import src.models.nn.utils as U
import src.utils as utils
import src.utils.train
from src.dataloaders import SequenceDataset  # TODO make registry
from src.tasks import decoders, encoders, tasks
from src.utils import registry
from src.utils.optim_groups import add_optimizer_hooks

from icecream import ic

log = src.utils.train.get_logger(__name__)

# Turn on TensorFloat32 (speeds up large model training substantially)
import torch.backends
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

OmegaConf.register_new_resolver('eval', eval)
OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)

#and make pretty errors work
# import pretty_errors

# Lots of annoying hacks to get WandbLogger to continuously retry on failure
class DummyExperiment:
    """Dummy experiment."""

    def nop(self, *args, **kw):
        pass

    def __getattr__(self, _):
        return self.nop

    def __getitem__(self, idx) -> "DummyExperiment":
        # enables self.logger.experiment[0].add_image(...)
        return self

    def __setitem__(self, *args, **kwargs) -> None:
        pass


def rank_zero_experiment(fn: Callable) -> Callable:
    """Returns the real experiment on rank 0 and otherwise the DummyExperiment."""

    @wraps(fn)
    def experiment(self):
        @rank_zero_only
        def get_experiment():
            return fn(self)

        return get_experiment() or DummyExperiment()

    return experiment


class CustomWandbLogger(WandbLogger):

    def __init__(self, *args, **kwargs):
        """Modified logger that insists on a wandb.init() call and catches wandb's error if thrown."""

        super().__init__(*args, **kwargs)

    @property
    @rank_zero_experiment
    def experiment(self):
        r"""
        Actual wandb object. To use wandb features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.
        Example::
        .. code-block:: python
            self.logger.experiment.some_wandb_function()
        """
        if self._experiment is None:
            if self._offline:
                os.environ["WANDB_MODE"] = "dryrun"

            attach_id = getattr(self, "_attach_id", None)
            if wandb.run is not None:
                # wandb process already created in this instance
                rank_zero_warn(
                    "There is a wandb run already in progress and newly created instances of `WandbLogger` will reuse"
                    " this run. If this is not desired, call `wandb.finish()` before instantiating `WandbLogger`."
                )
                self._experiment = wandb.run
            elif attach_id is not None and hasattr(wandb, "_attach"):
                # attach to wandb process referenced
                self._experiment = wandb._attach(attach_id)
            else:
                # create new wandb process
                while True:
                    try:
                        self._experiment = wandb.init(**self._wandb_init)
                        break
                    except Exception as e:
                        print("wandb Exception:\n", e)
                        t = random.randint(30, 60)
                        print(f"Sleeping for {t} seconds")
                        time.sleep(t)

                # define default x-axis
                if getattr(self._experiment, "define_metric", None):
                    self._experiment.define_metric("trainer/global_step")
                    self._experiment.define_metric("*", step_metric="trainer/global_step", step_sync=True)

        return self._experiment


class SequenceLightningModule(pl.LightningModule):
    def __init__(self, config):
        # Disable profiling executor. This reduces memory and increases speed.
        try:
            torch._C._jit_set_profiling_executor(False)
            torch._C._jit_set_profiling_mode(False)
        except AttributeError:
            pass

        super().__init__()
        # Passing in config expands it one level, so can access by self.hparams.train instead of self.hparams.config.train
        self.save_hyperparameters(config, logger=False)

        #here we check if our parameters say to load a bias model
        if config.train.get("bias_model_path", None) is not None:
            self.bias_model = torch.load(config.train.bias_model_path)
            #freeze the weights
            for param in self.bias_model.parameters():
                param.requires_grad = False

        # Dataset arguments
        # print(self.hparams.dataset._name_) is cCRE or DNase depending on what you need
        # print(SequenceDataset.registry)
        self.dataset = SequenceDataset.registry[self.hparams.dataset._name_](
            **self.hparams.dataset
        )

        # Check hparams
        self._check_config()

        # PL has some bugs, so add hooks and make sure they're only called once
        self._has_setup = False

        self.setup()  ## Added by KS

    def setup(self, stage=None):
        if not self.hparams.train.disable_dataset:
            self.dataset.setup()

        # We need to set up the model in setup() because for some reason when training with DDP, one GPU uses much more memory than the others
        # In order to not overwrite the model multiple times during different stages, we need this hack
        # TODO PL 1.5 seems to have an option to skip hooks to avoid this
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/5410#issuecomment-762257024
        if self._has_setup:
            return
        else:
            self._has_setup = True

        # Convenience feature: if model specifies encoder, combine it with main encoder
        encoder_cfg = utils.to_list(self.hparams.encoder) + utils.to_list(
            self.hparams.model.pop("encoder", None)
        )
        decoder_cfg = utils.to_list(
            self.hparams.model.pop("decoder", None)
        ) + utils.to_list(self.hparams.decoder)

        # Instantiate model
        self.model = utils.instantiate(registry.model, self.hparams.model)
        if (name := self.hparams.train.post_init_hook['_name_']) is not None:
            kwargs = self.hparams.train.post_init_hook.copy()
            del kwargs['_name_']
            for module in self.modules():
                if hasattr(module, name):
                    getattr(module, name)(**kwargs)

        # Instantiate the task
        self.task = utils.instantiate(
            tasks.registry, self.hparams.task, dataset=self.dataset, model=self.model
        )
        # print("Task:", self.task)
        # import sys
        # #and close out the file
        # sys.exit()

        # Create encoders and decoders
        encoder = encoders.instantiate(
            encoder_cfg, dataset=self.dataset, model=self.model
        )
        decoder = decoders.instantiate(
            decoder_cfg, model=self.model, dataset=self.dataset
        )

        # Extract the modules so they show up in the top level parameter count
        self.encoder = U.PassthroughSequential(self.task.encoder, encoder)
        self.decoder = U.PassthroughSequential(decoder, self.task.decoder)
        self.loss = self.task.loss
        self.loss_val = self.task.loss
        if hasattr(self.task, 'loss_val'):
            self.loss_val = self.task.loss_val
        self.metrics = self.task.metrics
        self.train_torchmetrics = self.task.train_torchmetrics
        self.val_torchmetrics = self.task.val_torchmetrics
        self.test_torchmetrics = self.task.test_torchmetrics

    def load_state_dict(self, state_dict, strict=False):
        if self.hparams.train.pretrained_model_state_hook['_name_'] is not None:
            model_state_hook = utils.instantiate(
                registry.model_state_hook,
                self.hparams.train.pretrained_model_state_hook.copy(),
                partial=True,
            )
            state_dict = model_state_hook(self.model, state_dict)
        # print(self.model.lm_head.weight.shape) #this is the wrong shape, could find some way to modify it here
        print("Custom load_state_dict function is running.")
        # print("State dict keys:", state_dict.keys())
        # print(state_dict['model.backbone.embeddings.word_embeddings.weight'].shape)
        # print(state_dict['model.lm_head.weight'].shape)
        # print(self.model) #all have an embedding size 20x128
        # strict==True will require all modules to match
        # strict==False can allow encoder/decoder to be loaded from scratch too
        return super().load_state_dict(state_dict, strict=strict)

    def _check_config(self):
        assert self.hparams.train.state.mode in [None, "none", "null", "reset", "bptt", "tbptt"]
        assert (
            (n := self.hparams.train.state.n_context) is None
            or isinstance(n, int)
            and n >= 0
        )
        assert (
            (n := self.hparams.train.state.n_context_eval) is None
            or isinstance(n, int)
            and n >= 0
        )

    def _initialize_state(self):
        """Called at model setup and start of epoch to completely reset state"""
        self._state = None
        self._memory_chunks = []

    def _reset_state(self, batch, device=None):
        """Called to construct default_state when necessary, e.g. during BPTT"""
        device = device or batch[0].device
        self._state = self.model.default_state(*batch[0].shape[:1], device=device)

    def _detach_state(self, state):
        if isinstance(state, torch.Tensor):
            return state.detach()
        elif isinstance(state, tuple):
            return tuple(self._detach_state(s) for s in state)
        elif isinstance(state, list):
            return [self._detach_state(s) for s in state]
        elif isinstance(state, dict):
            return {k: self._detach_state(v) for k, v in state.items()}
        elif state is None:
            return None
        else:
            raise NotImplementedError

    def _process_state(self, batch, batch_idx, train=True):
        """Handle logic for state context."""
        # Number of context steps
        key = "n_context" if train else "n_context_eval"
        n_context = self.hparams.train.state.get(key)

        # Don't need to do anything if 0 context steps. Make sure there is no state
        if n_context == 0 and self.hparams.train.state.mode not in ['tbptt']:
            self._initialize_state()
            return

        # Reset state if needed
        if self.hparams.train.state.mode == "reset":
            if batch_idx % (n_context + 1) == 0:
                self._reset_state(batch)

        # Pass through memory chunks
        elif self.hparams.train.state.mode == "bptt":
            self._reset_state(batch)
            with torch.no_grad():  # should be unnecessary because individual modules should handle this
                for _batch in self._memory_chunks:
                    self.forward(_batch)
            # Prepare for next step
            self._memory_chunks.append(batch)
            self._memory_chunks = self._memory_chunks[-n_context:]

        elif self.hparams.train.state.mode == 'tbptt':
            _, _, z = batch
            reset = z["reset"]
            if reset:
                self._reset_state(batch)
            else:
                self._state = self._detach_state(self._state)

    # def forward(self, batch):
    #     """Passes a batch through the encoder, backbone, and decoder"""
    #     # z holds arguments such as sequence length
    #     x, y, *z = batch # z holds extra dataloader info such as resolution
    #     if len(z) == 0:
    #         z = {}
    #     else:
    #         assert len(z) == 1 and isinstance(z[0], dict), "Dataloader must return dictionary of extra arguments"
    #         z = z[0]

    #     x, w = self.encoder(x, **z) # w can model-specific constructions such as key_padding_mask for transformers or state for RNNs
    #     x, state = self.model(x, **w, state=self._state)
    #     self._state = state
    #     x, w = self.decoder(x, state=state, **z)
    #     return x, y, w

    def forward(self, batch):
        # print(self.encoder)
        # print(self.model)
        # print(self.decoder)
        # import sys
        # sys.exit()
        # if hasattr(self, 'bias_model'):
        #     return self.task.forward(batch, self.encoder, self.model, self.decoder, self._state, self.bias_model) #only for profile task
        return self.task.forward(batch, self.encoder, self.model, self.decoder, self._state)

    def step(self, x_t):
        x_t, *_ = self.encoder(x_t) # Potential edge case for encoders that expect (B, L, H)?
        x_t, state = self.model.step(x_t, state=self._state)
        self._state = state
        # x_t = x_t[:, None, ...] # Dummy length
        # x_t, *_ = self.decoder(x_t, state=state)
        # x_t = x_t[:, 0, ...]
        x_t, *_ = self.decoder.step(x_t, state=state)
        return x_t

    def _shared_step(self, batch, batch_idx, prefix="train"):

        self._process_state(batch, batch_idx, train=(prefix == "train")) #does some state stuff
        x, y, w = self.forward(batch) #here the forward is gone through, x is actually y hat, and y is the output, w is the parameters
        #expect x to be long x 16 since it's the logits
        # print("x shape", x.shape) #it is correct, as we expected
        # print("y shape", y.shape)
        # print("w shape", w) #just empty dict with state none
        # import sys
        # sys.exit()
        # Loss
        # print(self.loss) #both are the same of <function discard_kwargs.<locals>.f_ at 0x2aba86612c00>
        #this is located at src.models.nn.utils
        # print(self.loss_val)
        # import sys
        # sys.exit()
        if prefix == 'train':
            loss = self.loss(x, y, **w)
        else:
            loss = self.loss_val(x, y, **w)

        # Metrics
        metrics = self.metrics(x, y, **w)
        metrics["loss"] = loss
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        # Calculate torchmetrics
        torchmetrics = getattr(self, f'{prefix}_torchmetrics')
        torchmetrics(x, y, loss=loss)
        
        log_on_step = 'eval' in self.hparams and self.hparams.eval.get('log_on_step', False) and prefix == 'train'

        self.log_dict(
            metrics,
            on_step=log_on_step,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )

        # log the whole dict, otherwise lightning takes the mean to reduce it
        # https://pytorch-lightning.readthedocs.io/en/stable/visualize/logging_advanced.html#enable-metrics-for-distributed-training
        self.log_dict(
            torchmetrics,
            on_step=log_on_step,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )
        return loss

    def on_train_epoch_start(self):
        # Reset training torchmetrics
        self.task._reset_torchmetrics("train")

    # def training_epoch_end(self, outputs):
    #     # Log training torchmetrics
    #     super().training_epoch_end(outputs)
    #     #inherits the training_epoch_end of the parent class, which is the basic pl.lightningmodule.
    #     #we will just comment it out since it doesn't seem to do anything using the basic values, doesn't seem to be called anywhere

    def on_validation_epoch_start(self):
        # Reset all validation torchmetrics
        for name in self.val_loader_names:
            self.task._reset_torchmetrics(name)

    # def validation_epoch_end(self, outputs):
    #     # Log all validation torchmetrics
    #     super().validation_epoch_end(outputs)

    def on_test_epoch_start(self):
        # Reset all test torchmetrics
        for name in self.test_loader_names:
            self.task._reset_torchmetrics(name)

    # def test_epoch_end(self, outputs):
    #     # Log all test torchmetrics
    #     super().test_epoch_end(outputs)

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self._shared_step(batch, batch_idx, prefix="train")

        # Log the loss explicitly so it shows up in WandB
        # Note that this currently runs into a bug in the progress bar with ddp (as of 1.4.6)
        # https://github.com/PyTorchLightning/pytorch-lightning/pull/9142
        # We additionally log the epochs under 'trainer' to get a consistent prefix with 'global_step'
        loss_epoch = {"trainer/loss": loss, "trainer/epoch": self.current_epoch}
        self.log_dict(
            loss_epoch,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            add_dataloader_idx=False,
            sync_dist=True,
        )

        # Log any extra info that the models want to expose (e.g. output norms)
        metrics = {}
        for module in list(self.modules())[1:]:
            if hasattr(module, "metrics"):
                metrics.update(module.metrics)

        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            add_dataloader_idx=False,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        ema = (
            self.val_loader_names[dataloader_idx].endswith("/ema")
            and self.optimizers().optimizer.stepped
        )  # There's a bit of an annoying edge case with the first (0-th) epoch; it has to be excluded due to the initial sanity check
        if ema:
            self.optimizers().swap_ema()
        loss = self._shared_step(
            batch, batch_idx, prefix=self.val_loader_names[dataloader_idx]
        )
        if ema:
            self.optimizers().swap_ema()

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_step(
            batch, batch_idx, prefix=self.test_loader_names[dataloader_idx]
        )

    def configure_optimizers(self):
        # Set zero weight decay for some params
        if 'optimizer_param_grouping' in self.hparams.train:
            add_optimizer_hooks(self.model, **self.hparams.train.optimizer_param_grouping)

        # Normal parameters
        all_params = list(self.parameters())
        params = [p for p in all_params if not hasattr(p, "_optim")]

        optimizer = utils.instantiate(registry.optimizer, self.hparams.optimizer, params)

        del self.hparams.optimizer._name_

        # Add parameters with special hyperparameters
        hps = [getattr(p, "_optim") for p in all_params if hasattr(p, "_optim")]
        hps = [
            # dict(s) for s in set(frozenset(hp.items()) for hp in hps)
            dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
            # dict(s) for s in dict.fromkeys(frozenset(hp.items()) for hp in hps)
        ]  # Unique dicts
        print("Hyperparameter groups", hps)
        for hp in hps:
            params = [p for p in all_params if getattr(p, "_optim", None) == hp]
            optimizer.add_param_group(
                {"params": params, **self.hparams.optimizer, **hp}
            )

        ### Layer Decay ###

        if self.hparams.train.layer_decay['_name_'] is not None:
            get_num_layer = utils.instantiate(
                registry.layer_decay,
                self.hparams.train.layer_decay['_name_'],
                partial=True,
            )

            # Go through all parameters and get num layer
            layer_wise_groups = {}
            num_max_layers = 0
            for name, p in self.named_parameters():
                # Get layer id for each parameter in the model
                layer_id = get_num_layer(name)

                # Add to layer wise group
                if layer_id not in layer_wise_groups:
                    layer_wise_groups[layer_id] = {
                        'params': [],
                        'lr': None,
                        'weight_decay': self.hparams.optimizer.weight_decay
                    }
                layer_wise_groups[layer_id]['params'].append(p)

                if layer_id > num_max_layers: num_max_layers = layer_id

            # Update lr for each layer
            for layer_id, group in layer_wise_groups.items():
                group['lr'] = self.hparams.optimizer.lr * (self.hparams.train.layer_decay.decay ** (num_max_layers - layer_id))

            # Reset the torch optimizer's param groups
            optimizer.param_groups = []
            for layer_id, group in layer_wise_groups.items():
                optimizer.add_param_group(group)

        # Print optimizer info for debugging
        keys = set([k for hp in hps for k in hp.keys()])  # Special hparams
        utils.train.log_optimizer(log, optimizer, keys)
        # Configure scheduler
        if "scheduler" not in self.hparams:
            return optimizer
        lr_scheduler = utils.instantiate(
            registry.scheduler, self.hparams.scheduler, optimizer
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": self.hparams.train.interval,  # 'epoch' or 'step'
            "monitor": self.hparams.train.monitor,
            "name": "trainer/lr",  # default is e.g. 'lr-AdamW'
        }
        # See documentation for how to configure the return
        # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self.dataset.train_dataloader(**self.hparams.loader)

    def _eval_dataloaders_names(self, loaders, prefix):
        """Process loaders into a list of names and loaders"""
        if utils.is_dict(loaders):
            return [
                f"{prefix}/{k}" if k is not None else prefix for k in loaders.keys()
            ], list(loaders.values())
        elif utils.is_list(loaders):
            return [f"{prefix}/{i}" for i in range(len(loaders))], loaders
        else:
            return [prefix], [loaders]

    def _eval_dataloaders(self):
        # Return all val + test loaders
        val_loaders = self.dataset.val_dataloader(**self.hparams.loader)
        test_loaders = self.dataset.test_dataloader(**self.hparams.loader)
        val_loader_names, val_loaders = self._eval_dataloaders_names(val_loaders, "val")
        test_loader_names, test_loaders = self._eval_dataloaders_names(
            test_loaders, "test"
        )

        # Duplicate datasets for ema
        if self.hparams.train.ema > 0.0:
            val_loader_names += [name + "/ema" for name in val_loader_names]
            val_loaders = val_loaders + val_loaders
            test_loader_names += [name + "/ema" for name in test_loader_names]
            test_loaders = test_loaders + test_loaders

        # adding option to only have val loader at eval (eg if test is duplicate)
        if self.hparams.train.get("remove_test_loader_in_eval", False):
            return val_loader_names, val_loaders
        # adding option to only have test loader at eval
        elif self.hparams.train.get("remove_val_loader_in_eval", False):
            return test_loader_names, test_loaders
        # default behavior is to add test loaders in eval
        else:
            return val_loader_names + test_loader_names, val_loaders + test_loaders

    def val_dataloader(self):
        val_loader_names, val_loaders = self._eval_dataloaders()
        self.val_loader_names = val_loader_names
        return val_loaders

    def test_dataloader(self):
        test_loader_names, test_loaders = self._eval_dataloaders()
        self.test_loader_names = ["final/" + name for name in test_loader_names]
        return test_loaders

    # for debugging
    # def on_after_backward(self):
    #     # if self.hparams.train.get("clip_grad_norm", None) is not None:
    #     #     torch.nn.utils.clip_grad_norm_(
    #     #         self.parameters(), self.hparams.train.clip_grad_norm
    #     #     )
    #     super().on_after_backward()  # Ensure the original behavior is preserved
    #     unused_params = []
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             unused_params.append(name)
    #     if unused_params:
    #         print("Unused Parameters:", unused_params)
    #     import sys
    #     sys.exit()
    
    #save model after training
    # def on_train_end()


### pytorch-lightning utils and entrypoint ###

def create_trainer(config, **kwargs):
    callbacks: List[pl.Callback] = []
    logger = None

    # WandB Logging
    if config.get("wandb") is not None:
        # Pass in wandb.init(config=) argument to get the nice 'x.y.0.z' hparams logged
        # Can pass in config_exclude_keys='wandb' to remove certain groups
        import wandb

        logger = CustomWandbLogger(
            config=utils.to_dict(config, recursive=True),
            settings=wandb.Settings(start_method="fork"),
            **config.wandb,
        )

    # Lightning callbacks
    if "callbacks" in config:
        for _name_, callback in config.callbacks.items():
            if config.get("wandb") is None and _name_ in ["learning_rate_monitor"]:
                continue
            log.info(f"Instantiating callback <{registry.callbacks[_name_]}>")
            callback._name_ = _name_
            callbacks.append(utils.instantiate(registry.callbacks, callback))

    # Add ProgressiveResizing callback
    if config.callbacks.get("progressive_resizing", None) is not None:
        num_stages = len(config.callbacks.progressive_resizing.stage_params)
        print(f"Progressive Resizing: {num_stages} stages")
        for i, e in enumerate(config.callbacks.progressive_resizing.stage_params):
            # Stage params are resolution and epochs, pretty print
            print(f"\tStage {i}: {e['resolution']} @ {e['epochs']} epochs")

    # Configure ddp automatically
    n_devices = config.trainer.get('devices', 1)
    if isinstance(n_devices, Sequence):  # trainer.devices could be [1, 3] for example
        n_devices = len(n_devices)
    if n_devices > 1 and config.trainer.get('strategy', None) is None:
        config.trainer.strategy = dict(
            _target_='pytorch_lightning.strategies.DDPStrategy',
            find_unused_parameters=False,
            gradient_as_bucket_view=True,  # https://pytorch-lightning.readthedocs.io/en/stable/advanced/advanced_gpu.html#ddp-optimizations
        )

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    # special processing for seqlen warmup scheduler (reload)
    if config.callbacks.get("seqlen_warmup_reload", None) is not None:
        # print('IT is the IF')
        # we need to instantiate manually instead of with hydra, since it expects a dict instead of a hydra config for the accumulate_grad_batches
        # so we convert everything to dicts (from hydra configs)
        trainer_config_dict = dict(config.trainer)
        epochs_cume = 0  # track cumulative epochs
        accumulate_grad_schedule = {}  # contains the accumulate_grad_batches schedule to init the trainer

        for stage in config.callbacks.seqlen_warmup_reload.stage_params:
            batch_size = stage['batch_size']  # curr batch size at this stage
            grad_accum_factor = config.train.global_batch_size // batch_size  # grad accum factor for this stage
            accumulate_grad_schedule[epochs_cume] = grad_accum_factor  # set the grad accum factor for this stage
            epochs_cume += stage['epochs']  # increment epochs_cume for next stage
        trainer_config_dict['accumulate_grad_batches'] = accumulate_grad_schedule  # set the accumulate_grad_batches schedule
        trainer_config_dict.pop('_target_')  # only hydra uses this to instantiate
        # Set DDPStrategy to work with pl.Trainer
        config.trainer.pop('strategy')
        trainer_config_dict['strategy'] = DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True)
        trainer = pl.Trainer(**trainer_config_dict, callbacks=callbacks, logger=logger)
    else:
        # print('IT is the ELSE') #this is what runs
        trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=logger)    

    return trainer

# import subprocess

# def check_gpu_status():
#     try:
#         # Execute the nvidia-smi command and get JSON output
#         result = subprocess.run(['nvidia-smi', '-q', '-x'], capture_output=True, text=True)
#         if result.returncode != 0:
#             print("Failed to execute nvidia-smi")
#             return
        
#         # Convert XML output to JSON for easier parsing
#         # Using xml.etree.ElementTree to parse XML
#         import xml.etree.ElementTree as ET
#         root = ET.fromstring(result.stdout)
        
#         # Parse and print GPU information
#         for gpu in root.findall('gpu'):
#             gpu_id = gpu.find('minor_number').text
#             gpu_name = gpu.find('product_name').text
#             gpu_util = gpu.find('utilization').find('gpu_util').text
#             memory_total = gpu.find('fb_memory_usage').find('total').text
#             memory_used = gpu.find('fb_memory_usage').find('used').text
#             memory_free = gpu.find('fb_memory_usage').find('free').text
#             print(f"GPU {gpu_id} - {gpu_name}: Utilization {gpu_util}, Memory Usage {memory_used}/{memory_total} ({memory_free} free)")
#     except Exception as e:
#         print(f"An error occurred: {e}")

def train(config):
    if config.train.seed is not None:
        pl.seed_everything(config.train.seed, workers=True)
    trainer = create_trainer(config)
    model = SequenceLightningModule(config)
    # print(model)
    # print(model.model.backbone.load_old_embedding,'\n','\n','\n','\n','\n','\n','\n','\n','\n','\n','\n','\n','\n','\n')
    #this is fine now
    # check_gpu_status()
    
    #check which device the model is on
    # param = next(model.parameters())
    # print(param.device)
    # if isinstance(model, torch.nn.DataParallel):
    #     print(model.device_ids)
    #it appears to be cpu and not data parallel yet
    # model = model.to('cuda:0')

    # Load pretrained_model if specified
    if config.train.get("pretrained_model_path", None) is not None:
        # PTL style.  Note, method returns a new model object, and need to pass config.
        # print(config)
        #first check if ddp
        #the model doesn't like having the load_old_embeddings for some reason, so let's remove it
        # temp_config = config
        # if temp_config.train.pretrained_model_state_hook.get('load_old_embedding', None) is not None:
        #     temp_config.train.pretrained_model_state_hook.pop('load_old_embedding')
        model = SequenceLightningModule.load_from_checkpoint(
            config.train.pretrained_model_path,
            config=config,
            strict=config.train.pretrained_model_strict_load,
            map_location='cpu',
        )
        # print(model)
        #let's just go ahead and find the weights and save the embeddings
        # torch.save(ic(model.model.backbone.embeddings.word_embeddings), '/data/leslie/sarthak/data/og_embeddings.pt')
        # torch.save(ic(model.model.backbone.new_embeddings.word_embeddings), '/data/leslie/sarthak/data/new_embeddings.pt')
        # try:
        #     load_old_embeddings = config.train['pretrained_model_state_hook']['load_old_embedding'] #a better hack to load the old embeddings
        # except:
        #     load_old_embeddings = False

        #a better method than the try except is
        load_old_embeddings =  config.train['pretrained_model_state_hook'].get('load_old_embedding', False)
            # load_old_embeddings = config.train['pretrained_model_state_hook']['load_old_embedding']
        #much better than the one below
        if load_old_embeddings: #for all intents and purposes, is 12!
            #now we replace the embeddings with the old ones
            old_embeddings = model.model.backbone.embeddings.word_embeddings
            new_embeddings = model.model.backbone.new_embeddings.word_embeddings
            new_embeddings.weight.data[:load_old_embeddings] = old_embeddings.weight.data[:load_old_embeddings] #this should be the important ones
            model.model.backbone.embeddings.word_embeddings = new_embeddings
            model.model.lm_head = new_embeddings #so we can load the model, we never use this for downstream finetuned tasks
            #now let's set the unused embeddings to None
            model.model.backbone.new_embeddings = None #doesn't affect the rest now new_embeddings references None, they reference old new_embedding
            # old_embeddings = None
            #the other issue is that old_embeddings 
            
        print('model successfully loaded!!')
        # print(model)
        # print(model.model.backbone.embeddings.word_embeddings)
    # print(model)
    #below is a hack to access the embeddings, we will save it out once, then can randomly access it
    #set seed, so should be repeatable, and honestly not the worst thing, just grabbing them all
    #you set the ignore_embeddings to true, and then you can access the embeddings and save them out
    # embeddings = model.model.backbone.embeddings.word_embeddings
    # #and save them out
    # torch.save(embeddings, '/data/leslie/sarthak/data/saved_embeddings.pt')
    # import sys
    # sys.exit()
    
                
    # word_embeddings_layer = self.model.backbone.embeddings.word_embeddings #this accesses the embeddings
    # print(self.hparams.model) #dict of the stuff under model in the experiment
    # print(self.hparams.train) # this is what we need to decide to add embeddings or not
    # print("Model:", self.model)
    # print(word_embeddings_layer) #class Embedding(20,128), we know how to modify that and access the embeddings

    #note that this approach is not great, as it just uses the already saved embeddings. terrible coding practice, not flexible at all
    # '''A hack to ensure that we can add embeddings to the modle while keeping the old embeddings
    # This approach is useful to increase the vocabulary size of the model
    # No longer used, as the embeddings don't really matter, can be retrained, but the option remains
    # Requres saving out the old embeddings manually, then it loads them in here
    # Also, we have a better method below, where we don't need this anymore
    # Actually this method is wrong, so we will be taking it out and commenting it, not useful!!'''
    # if config.train['pretrained_model_state_hook']['add_embeddings']:
    #     # keep = config.train['pretrained_model_state_hook']['add_embeddings']
    #     #we add the number of embeddings equal to the model
    #     original_embeddings = model.model.backbone.embeddings.word_embeddings
    #     # new_vocab_size = original_embeddings.num_embeddings + add_embeddings
    #     # new_embedding_layer = torch.nn.Embedding(new_vocab_size, original_embeddings.embedding_dim)
    #     # nn.init.normal_(module.weight, std=initializer_range) #no need to use since we grabbed their values
    #     saved_embeddings = torch.load('/data/leslie/sarthak/data/saved_embeddings.pt')
    #     saved_embeddings.weight.data[:12] = original_embeddings.weight.data[:12] #because it's set by what was saved
    #     model.model.backbone.embeddings.word_embeddings = saved_embeddings
    #     print(saved_embeddings)
    
    # import sys
    # sys.exit()
    #we will now test to see if our data matches
    # ic(model.model.backbone.embeddings.word_embeddings)
    #now load in embeddings with torch
    # old = torch.load('/data/leslie/sarthak/data/saved_embeddings.pt') #based on wrong ones so does give false, but ours is correct
    # ic(old)
    #and now make sure that all of it is the same
    # weights1 = model.model.backbone.embeddings.word_embeddings.weight.data[:12]
    # weights2 = old.weight.data[:12]
    # a = ic(torch.allclose(weights1.detach().cpu(), weights2.detach().cpu()))
    # ic(a)
    #save out the new embeddings
    # torch.save(model.model.backbone.embeddings.word_embeddings, '/data/leslie/sarthak/data/saved_embeddings_new.pt')
    # import sys
    # sys.exit()
    

    # Run initial validation epoch (useful for debugging, finetuning)
    if config.train.validate_at_start:
        print("Running validation before training")
        trainer.validate(model)

    if config.train.ckpt is not None:
        trainer.fit(model, ckpt_path=config.train.ckpt)
    else:
        trainer.fit(model)
    if config.train.test:
        trainer.test(model)




@hydra.main(config_path="configs", config_name="config.yaml")
def main(config: OmegaConf):

    # Process config:
    # - register evaluation resolver
    # - filter out keys used only for interpolation
    # - optional hooks, including disabling python warnings or debug friendly configuration
    config = utils.train.process_config(config)

    # Pretty print config using Rich library
    utils.train.print_config(config, resolve=True)

    train(config)
    print('model trained')


if __name__ == "__main__":
    main()