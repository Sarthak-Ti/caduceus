import inspect
from typing import List

import torch.nn as nn
from einops import rearrange

import torch
import src.models.nn.utils as U
import src.tasks.metrics as M
import torchmetrics as tm
from src.models.nn.adaptive_softmax import AdaptiveEmbedding, ProjectedAdaptiveLogSoftmax
from src.tasks.torchmetrics import torchmetric_fns as tm_mine
from src.utils.config import to_list, instantiate
from torchmetrics import MetricCollection


class BackboneDecoder(nn.Module):
    def __init__(self, model1, model2):
        super(BackboneDecoder, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, x):
        x,_ = self.model1(x)
        x = self.model2(x) #because state is also passed, but this is to ignore it
        return x

class BaseTask:
    """ Abstract class that takes care of:
    - loss function
    - arbitrary metrics
    - forward pass
    - (optional) encoder module that interfaces with dataset (inputs) and model
    - (optional) decoder module that interfaces with dataset (targets) and model
    """
    encoder = None
    decoder = None

    def __init__(self, dataset=None, model=None, loss=None, loss_val=None, metrics=None, torchmetrics=None, bias_model=None):
        """ This class is allowed to grab attributes directly off a constructed dataset and model object """
        self.dataset = dataset
        self.model = model
        if metrics is None: metrics = []
        self.metric_names = to_list(metrics)

        if torchmetrics is None: torchmetrics = []
        self.torchmetric_names = to_list(torchmetrics)
        self._tracked_torchmetrics = {}

        # The decoder might pass through arguments that the loss needs (e.g. sequence lengths)
        # but might also pass through extraneous arguments (e.g. sampling rate)
        # Wrap loss and metrics so that they accept kwargs and

        # Create loss function
        # print(loss)
        # import sys
        # sys.exit()
        self.loss = instantiate(M.output_metric_fns, loss, partial=True)
        self.loss = U.discard_kwargs(self.loss)
        if loss_val is not None:
            self.loss_val = instantiate(M.output_metric_fns, loss_val, partial=True)
            self.loss_val = U.discard_kwargs(self.loss_val)
        torchmetrics = MetricCollection(self._init_torchmetrics())
        self.train_torchmetrics = torchmetrics.clone(prefix='train/')
        self.val_torchmetrics = torchmetrics.clone(prefix='val/')
        self.test_torchmetrics = torchmetrics.clone(prefix='test/')

    def _init_torchmetrics(self):
        """
        Instantiate torchmetrics.
        """
        tracked_torchmetrics = {}

        for name in self.torchmetric_names:
            if name in tm_mine:
                tracked_torchmetrics[name] = tm_mine[name]()
            elif name in ['AUROC', 'StatScores', 'Precision', 'Recall', 'F1', 'F1Score']:
                tracked_torchmetrics[name] = getattr(tm, name)(
                    average='macro', num_classes=self.dataset.d_output, compute_on_step=False
                )
            elif name in ['MultilabelAUROC', 'MultilabelAveragePrecision']:
                tracked_torchmetrics[name] = getattr(tm, name)(
                    average='macro', num_labels=self.dataset.d_output
                )
            elif '@' in name:
                k = int(name.split('@')[1])
                mname = name.split('@')[0]
                tracked_torchmetrics[name] = getattr(tm, mname)(
                    average='macro', num_classes=self.dataset.d_output, compute_on_step=False, top_k=k
                )
            else:
                tracked_torchmetrics[name] = getattr(tm, name)(compute_on_step=False)

        return tracked_torchmetrics

    def _reset_torchmetrics(self, prefix=None):
        """
        Reset torchmetrics for a prefix
        associated with a particular dataloader (e.g. train, val, test).

        Generally do this at the start of an epoch.
        """
        all_prefixes = [prefix] if prefix is not None else self._tracked_torchmetrics

        for prefix in all_prefixes:
            if prefix in self._tracked_torchmetrics:
                self._tracked_torchmetrics[prefix].reset()

    def get_torchmetrics(self, prefix):
        """
        Compute torchmetrics for a prefix associated with
        a particular dataloader (e.g. train, val, test).

        Generally do this at the end of an epoch.
        """
        return {name: self._tracked_torchmetrics[prefix][name].compute() for name in self.torchmetric_names}

    def torchmetrics(self, x, y, prefix, loss=None):
        """
        Update torchmetrics with new x, y .
        Prefix corresponds to a particular dataloader (e.g. train, val, test).

        Generally call this every batch.
        """
        if prefix not in self._tracked_torchmetrics:
            self._init_torchmetrics(prefix)
        self._tracked_torchmetrics[prefix](x, y, loss=loss)

        # for name in self.torchmetric_names:
        #     if name.startswith('Accuracy'):
        #         if len(x.shape) > 2:
        #             # Multi-dimensional, multi-class
        #             self._tracked_torchmetrics[prefix][name].update(x.transpose(1, 2), y.squeeze())
        #             continue
        #     self._tracked_torchmetrics[prefix][name].update(x, y)

    def get_torchmetrics(self, prefix):
        return self._tracked_torchmetrics[prefix]

    def metrics(self, x, y, **kwargs):
        """
        Metrics are just functions
        output metrics are a function of output and target
        loss metrics are a function of loss (e.g. perplexity)
        """
        output_metrics = {
            name: U.discard_kwargs(M.output_metric_fns[name])(x, y, **kwargs)
            for name in self.metric_names if name in M.output_metric_fns
        }
        loss_metrics = {
            name: U.discard_kwargs(M.loss_metric_fns[name])(x, y, self.loss, **kwargs)
            for name in self.metric_names if name in M.loss_metric_fns
        }
        return {**output_metrics, **loss_metrics}

    def forward(self, batch, encoder, model, decoder, _state):
        """Passes a batch through the encoder, backbone, and decoder"""
        # z holds arguments such as sequence length
        x, y, *z = batch # z holds extra dataloader info such as resolution
        if len(z) == 0:
            z = {}
        else:
            assert len(z) == 1 and isinstance(z[0], dict), "Dataloader must return dictionary of extra arguments"
            z = z[0]

        # w can model-specific constructions, such as key_padding_mask for transformers or state for RNNs
        x, w = encoder(x, **z)
        x, state = model(x, **w, state=_state)
        self._state = state
        x, w = decoder(x, state=state, **z)
        return x, y, w


class Scalar(nn.Module):
    def __init__(self, c=1):
        super().__init__()
        self.c = c

    def forward(self, x):
        return x * self.c


class LMTask(BaseTask):
    def forward(self, batch, encoder, model, decoder, _state):
        """Passes a batch through the encoder, backbone, and decoder"""
        # z holds arguments such as sequence length
        x, y, *z = batch # z holds extra dataloader info such as resolution
        # print("x shape: ", x.shape)
        # print("y shape: ", y.shape) #x and y shape seems to be 512 x 1023, not sure why since batch size is 256
        # print("z shape: ", z) #is an empty dict for now, because not doing transformer
        # print("x one sample", x[0,:])
        # print("y one sample", y[0,:])
        # print('test pad', x[0,0:25])
        if len(z) == 0:
            z = {} #sets to an exmpty dict, seems dataloader can have some other options, only for encoder and decoder
        else:
            assert len(z) == 1 and isinstance(z[0], dict), "Dataloader must return dictionary of extra arguments"
            z = z[0]
        x, w = encoder(x, **z) # w can model-specific constructions such as key_padding_mask for transformers or state for RNNs
        #the x output should be identical, not sure what w is, probably some options
        # print('w: ', w)
        # print("state before model: ", _state)
        if "state" in inspect.signature(model.forward).parameters.keys():
            x, state = model(x, **w, state=_state)
        else:
            x = model(x, **w)
            state = None
        self._state = state
        # print("state after model: ", state) #both are none
        x, w = decoder(x, state=state, **z)
        # print(torch.all(x.eq(w)))
        # print("w after decoder: ", w) # just a dict with key state and value none
        # print("x after decoder: ", x) #this is now a dict
        if hasattr(x, 'logits'):
            x = x.logits
        x = rearrange(x, '... C -> (...) C')
        y = rearrange(y, '... -> (...)')
        # print("x shape after logit: ", x.shape) #is (512 x 1023) x 16 which doesn't make sense, but that is the network output
        #it is 523776 x 16, and then y is just the labels, so it is 523776 x 1
        
        #let's visualize the output before rearranging
        # print(x.shape) #is now 512 x 1023 x 16 for the logits. batch size, then for each of the values the prediction for the 16 classes
        # print(y.shape)
        # print('example 15')
        # print(x[15,100:102,:])
        # print(y[15, 100:102]) #the actual values
        # print('example 16')
        # print(x[16,100:102,:])
        # print(y[16, 100:102]) #the actual values
        #looks as you expect, the predicted logits and the actual values
        
               
        # print("x shape after rearrange: ", x.shape)
        # print("y shape before rearrange: ", y.shape)
        # print("y shape after rearrange: ", y.shape)
        # import sys
        # sys.exit()
        return x, y, w


class Regression(BaseTask): #my custom defined task
    # def __init__(self, batch, encoder, model, decoder, _state):
    #     super().__init__(dataset, model, loss, loss_val, metrics, torchmetrics)
    #     self.loss = nn.MSELoss()
    def forward(self, batch, encoder, model, decoder, _state, minimum=-10):
        """Passes a batch through the encoder, backbone, and decoder"""
        # z holds arguments such as sequence length
        x, y, *z = batch
        
        # ic(x.shape)
        # # ic(y.shape)
        # ic(y[0].shape)
        # ic(y[1].shape)
        # ic(y)
        # ic(x)
        # ic(z)
        
        if len(z) == 0:
            z = {}
        else:
            assert len(z) == 1 and isinstance(z[0], dict), "Dataloader must return dictionary of extra arguments"
            z = z[0]
            
        x, w = encoder(x) # w can model-specific constructions such as key_padding_mask for transformers or state for RNNs
        # ic(x)
        # ic(x.shape)
        # print(model)
        x, state = model(x)
        self._state = state
        x, w = decoder(x, state=state, **z)
        # print(x.shape, y.shape, w) #with d_output=1 we find that x is 1024 x 1, y is 1024 x 1, and w is none. This is perfect and what we need!
        # import sys
        # sys.exit()
        
        #what we will do is now clamp the data to at minimum be -10
        x = torch.clamp(x, min=minimum)
        # ic(x)
        # import sys
        # sys.exit()
        # ic(y)
        # ic(w)
        return x, y, w


    # def metrics(self, x, y, **kwargs):
    #     output_metrics = {
    #         name: U.discard_kwargs(M.output_metric_fns[name])(x, y, **kwargs)
    #         for name in self.metric_names if name in M.output_metric_fns
    #     }
    #     loss_metrics = {
    #         name: U.discard_kwargs(M.loss_metric_fns[name])(x, y, self.loss, **kwargs)
    #         for name in self.metric_names if name in M.loss_metric_fns
    #     }
    #     return {**output_metrics, **loss_metrics}

class RegClass(BaseTask):
    #this class will separately calculate the loss for regression and classification
    #note that since it's just the loss calculation, we can just use the same forward pass as the regression task
    def forward(self, batch, encoder, model, decoder, _state, minimum=-10):
        """Passes a batch through the encoder, backbone, and decoder"""
        # z holds arguments such as sequence length
        x, y, *z = batch
        
        # print("x shape: ", x.shape) # 1024 x 1023 as we expect
        # print(x[:,0:7]) #it seems to just work... damn!
        # # print("y shape: ", y.shape) #1024 x 1 again as expected!
        # # print("z shape: ", z) #is empty
        # import sys
        # sys.exit()
        # print('decoder:',decoder) #was identity until we set loose load, now it seems we can have a decoder, let's try again
        #just set d_output and it's good
        
        if len(z) == 0:
            z = {}
        else:
            assert len(z) == 1 and isinstance(z[0], dict), "Dataloader must return dictionary of extra arguments"
            z = z[0]
            
        x, w = encoder(x) # w can model-specific constructions such as key_padding_mask for transformers or state for RNNs
        x, state = model(x)
        self._state = state
        x, w = decoder(x, state=state, **z)
        # print(x.shape, y.shape, w) #with d_output=1 we find that x is 1024 x 1, y is 1024 x 1, and w is none. This is perfect and what we need!
        # import sys
        # sys.exit()
        
        #what we will do is now clamp the data to at minimum be -10
        x[1] = torch.clamp(x[1], min=minimum)
        # print()
        
        return x, y, w

class ProfileClass(BaseTask):
    def __init__(self, dataset=None, model=None, loss=None, loss_val=None, metrics=None, torchmetrics=None, bias_model=None):
        super().__init__(dataset, model, loss, loss_val, metrics, torchmetrics)
        self.bias_model = None
        self.backbone = None
        self.decoder_bias = None
        self.bias_model_pt = None
        if bias_model.endswith('.h5'):
        # if bias_model == '/data/leslie/sarthak/data/chrombpnet_test/chrombpnet_model_1000/models/bias_model_scaled.h5':
            self.bias_model = self.load_bias_modeltf(bias_model)
            #freeze bias model
            for param in self.bias_model.parameters():
                param.requires_grad = False
            # device = next(self.model.parameters()).device
            self.bias_model.to('cuda:0')
            print('using tensorflow bias model')
        elif bias_model.endswith('.ckpt') and bias_model.startswith('hs_'):
            #it's called hs_path when want to add it to hidden states or just path for standard chrombpnet way
            bias_model = bias_model[3:]
            #load it via torch and it runs through 2 
            print('using hyena bias model with hidden states')
            self.backbone, self.decoder_bias = self.load_bias_modelpt(bias_model)
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.decoder_bias.parameters():
                param.requires_grad = False
            self.backbone.to('cuda:0').eval()
            self.decoder_bias.to('cuda:0').eval()
        elif bias_model.endswith('.ckpt'):
            print('using hyena bias model on output')
            backbone, decoder_bias = self.load_bias_modelpt(bias_model)
            for param in backbone.parameters():
                param.requires_grad = False
            for param in decoder_bias.parameters():
                param.requires_grad = False
            self.bias_model_pt = BackboneDecoder(backbone,decoder_bias) #this will make it so that the output of the backbone is fed into the decoder
            self.bias_model_pt.to('cuda:0').eval()
        else:
            print('no bias model')
            
    def load_bias_modeltf(self, path):
        #we will load it as a pytorch model using jacob schrieber's code
        import sys
        sys.path.append('/data/leslie/sarthak/chrombpnet/')
        from bpnetlite.bpnet import BPNet
        #the location is /data/leslie/sarthak/chrombpnet/bpnetlite/bpnet.py
        model = BPNet.from_chrombpnet(path,trimming=(1024-800)//2)
        return model

    def load_bias_modelpt(self, path):
        #load the pytorch model, a bit harder
        import yaml
        from src.models.sequence.dna_embedding import DNAEmbeddingModel
        from src.tasks.decoders import ProfileDecoder
        cfg = '/data/leslie/sarthak/hyena/hyena-dna/configs/evals/profile.yaml'
        cfg = yaml.load(open(cfg, 'r'), Loader=yaml.FullLoader)
        train_cfg = cfg['train']
        model_cfg = cfg['model_config']
        d_output = train_cfg['d_output']
        backbone = DNAEmbeddingModel(**model_cfg)
        decoder = ProfileDecoder(model_cfg['d_model'], d_output=d_output, l_output=0, mode='pool')
        state_dict = torch.load(path, map_location='cpu')
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            state_dict["state_dict"], "model."
        )
        model_state_dict = state_dict["state_dict"]
        # need to remove torchmetrics. to remove keys, need to convert to list first
        for key in list(model_state_dict.keys()):
            if "torchmetrics" in key:
                model_state_dict.pop(key)
        # the state_dict keys slightly mismatch from Lightning..., so we fix it here
        decoder_state_dict = {}
        decoder_state_dict['output_transform_counts.weight'] = model_state_dict.pop('decoder.0.output_transform_counts.weight')
        decoder_state_dict['output_transform_counts.bias'] = model_state_dict.pop('decoder.0.output_transform_counts.bias')
        decoder_state_dict['output_transform_profile.weight'] = model_state_dict.pop('decoder.0.output_transform_profile.weight')
        decoder_state_dict['output_transform_profile.bias'] = model_state_dict.pop('decoder.0.output_transform_profile.bias')
        # now actually load the state dict to the decoder and backbone separately
        decoder.load_state_dict(decoder_state_dict, strict=True)
        backbone.load_state_dict(model_state_dict, strict=True)
        return backbone, decoder
    
    def forward(self,batch,encoder,model,decoder,_state):
        x, y, *z = batch
        #check if y is a tensor or a tuple
        #this is primarily for the onehot encoding and bias later on
        onehot_x = x[1]
        x = x[0]
        x_og = x #cuz it's batch x len, but test it
        true_counts = y[0]
        # else:
        #     true_counts = y[:,:-1]
        if len(z) == 0:
            z = {}
        else:
            assert len(z) == 1 and isinstance(z[0], dict), "Dataloader must return dictionary of extra arguments"
            z = z[0]
        x, w = encoder(x)
        x, state = model(x) #check what this state is and if we need to add it
        # if self.reverse: #doesn't really work, because idea makes no sense, only works for RC!
        #     x2, _ = encoder(x_reversed)
        #     x2, _ = model(x2)
        #     x = (x+x2)/2
        self._state = state #check state, check the backbone and decodder are correct
        #so state is none, backbone and decoder are correct when we have the hyena bias model
        if self.backbone is not None:
            x_backbone, _ = self.backbone(x_og)
            x = x + x_backbone
        x, w = decoder(x, state=state, **z)
        #now we need to center x[0] and get the middle 800
        profile_out = x[0]
        count_out = x[1]
        if self.bias_model_pt is not None: #added here for the cropping
            bias_output = self.bias_model_pt(x_og)
            profile_out = profile_out + bias_output[0]
            count_out = torch.logsumexp(torch.cat([x[1], bias_output[1]], dim=1), dim=1, keepdim=True)
        # true_counts = y[0]
        if profile_out.shape[1] > true_counts.shape[1]:
            #then we cut off from both ends until we get the same size
            diff = profile_out.shape[1] - true_counts.shape[1]
            start = diff // 2
            end = start + true_counts.shape[1]
            profile_out = profile_out[:, start:end].squeeze()
        # global bias_model
        # print(self.bias_model)
        # if self.bias_model is None:
            # import sys
            # print('this shit sucks')
            # sys.exit()
        if self.bias_model is not None: #added here because it's after cropping
            # print('bias model worked!')
            # import sys
            # sys.exit()
            # else:
            # print('didn\'t work')
            # import sys
            # sys.exit()
            #need to now separate between bias models
            bias_output = self.bias_model(onehot_x)
            profile_out = profile_out + bias_output[0].squeeze()
            #do log sum exp for count out
            count_out = torch.logsumexp(torch.cat([count_out, bias_output[1]], dim=1), dim=1, keepdim=True) #need to test this shapes
            #nn.Linear always preserves the last dimension, so we can just concatenate the two tensors and then do logsumexp
        x = (profile_out, count_out)
        return x, y, w

class EnformerTask(BaseTask):
    #This task is very basic but requires inputting the desired output length into the decoder
    def forward(self, batch, encoder, model, decoder, _state, minimum=-10):
        """Passes a batch through the encoder, backbone, and decoder"""
        # z holds arguments such as sequence length
        x, y, *z = batch
        
        if len(z) == 0:
            z = {}
        else:
            assert len(z) == 1 and isinstance(z[0], dict), "Dataloader must return dictionary of extra arguments"
            z = z[0]
            
        x, w = encoder(x) # w can model-specific constructions such as key_padding_mask for transformers or state for RNNs
        x, state = model(x) #this part can be quite slow on the cpu especially!
        #x shape after model is batch x seqlen x d_model in this case d_model is 256
        #y shape is batch x 896 x 5313 where the 896 corresponds to 128*896=114688 which is the number of nucleotides we predict over
        self._state = state
        x, w = decoder(x, state=state, **z)
        
        return x, y, w

class MultiClass(BaseTask):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.continual_metrics = {}
        for name in self.metric_names:
            if name.endswith('_per_class'):
                for spec_idx, spec in enumerate(self.dataset.species):
                    self.continual_metrics[name + '_' + spec] = M.output_metric_fns[name](spec_idx)
            elif name in ['precision_species', 'recall_species']:
                self.continual_metrics[name] = M.output_metric_fns[name](num_classes=len(self.dataset.species))

    def metrics(self, x, y, **kwargs):
        output_metrics = {}
        for name in self.metric_names:
            if name in M.output_metric_fns:
                if name.endswith('_per_class'):
                    for spec_idx, spec in enumerate(self.dataset.species):
                        self.continual_metrics[name + '_' + spec] = self.continual_metrics[name + '_' + spec].to(
                            x.device)
                        self.continual_metrics[name + '_' + spec].update(x, y)
                        output_metrics[name + '_' + spec] = self.continual_metrics[name + '_' + spec].compute()
                elif name in ['precision_species', 'recall_species']:
                    self.continual_metrics[name] = self.continual_metrics[name].to(x.device)
                    metrics = self.continual_metrics[name](x, y)
                    for spec_idx, spec in enumerate(self.dataset.species):
                        output_metrics[name[:-7] + spec] = metrics[spec_idx]
                else:
                    output_metrics[name] = U.discard_kwargs(M.output_metric_fns[name])(x, y, **kwargs)

        loss_metrics = {
            name: U.discard_kwargs(M.loss_metric_fns[name])(x, y, self.loss, **kwargs)
            for name in self.metric_names if name in M.loss_metric_fns
        }

        return {**output_metrics, **loss_metrics}

    def _reset_torchmetrics(self, prefix=None):
        super()._reset_torchmetrics(prefix)
        for name in self.metric_names:
            if name.endswith('_per_class'):
                for spec_idx, spec in enumerate(self.dataset.species):
                    self.continual_metrics[name + '_' + spec].reset()


class MaskedMultiClass(MultiClass):

    def forward(self, batch, encoder, model, decoder, _state):
        """Passes a batch through the encoder, backbone, and decoder"""

        # z holds arguments such as sequence length
        x, y, *z = batch # z holds extra dataloader info such as resolution
        if len(z) == 0:
            z = {}
        else:
            assert len(z) == 1 and isinstance(z[0], dict), "Dataloader must return dictionary of extra arguments"
            z = z[0]

        x, w = encoder(x) # w can model-specific constructions such as key_padding_mask for transformers or state for RNNs
        x, state = model(x)
        self._state = state
        x, w = decoder(x, state=state, **z)
        return x, y, w
        

class HG38Task(LMTask):

    def __init__(self, dataset=None, model=None, loss=None, loss_val=None, metrics=None, torchmetrics=None, last_k_ppl=None, per_token_ppl=None):
        """ Extending LMTask to add custom metrics for HG38 task 
        
        last_k_ppl: config for custom ppl, with hparams to pass with it

        per_token_ppl: config for per token ppl calc, with list of k (ppls) to track

        """
        self.dataset = dataset
        self.model = model
        if metrics is None:
            metrics = []
        self.metric_names = to_list(metrics)
        self.last_k_ppl = last_k_ppl
        self.per_token_ppl = per_token_ppl

        if torchmetrics is None:
            torchmetrics = []
        self.torchmetric_names = to_list(torchmetrics)
        self._tracked_torchmetrics = {}

        # The decoder might pass through arguments that the loss needs (e.g. sequence lengths)
        # but might also pass through extraneous arguments (e.g. sampling rate)
        # Wrap loss and metrics so that they accept kwargs and

        # Create loss function
        self.loss = instantiate(M.output_metric_fns, loss, partial=True)
        self.loss = U.discard_kwargs(self.loss)
        if loss_val is not None:
            self.loss_val = instantiate(M.output_metric_fns, loss_val, partial=True)
            self.loss_val = U.discard_kwargs(self.loss_val)
        torchmetrics = MetricCollection(self._init_torchmetrics())
        self.train_torchmetrics = torchmetrics.clone(prefix='train/')
        self.val_torchmetrics = torchmetrics.clone(prefix='val/')
        self.test_torchmetrics = torchmetrics.clone(prefix='test/')

        # Create custom metrics for last k ppl
        # last_k_ppl is a list of dicts (configs), so loop through them
        if self.last_k_ppl is not None:
            self.custom_ppl_dict = {}
            for k in self.last_k_ppl:
                key_name = "last_" + str(k) + "_ppl"
                # create config
                custom_ppl_config = {"_name_": "last_k_ppl", "k": k, "seq_len": self.dataset.max_length}
                k_ppl_fn = instantiate(M.output_metric_fns, custom_ppl_config, partial=True)
                k_ppl_fn = U.discard_kwargs(k_ppl_fn)
                self.custom_ppl_dict[key_name] = k_ppl_fn

        # Create custom metric for per token ppl
        if self.per_token_ppl is not None:
            per_token_ppl_config = {"_name_": "per_token_ppl", "ks": self.per_token_ppl["ks"],
                                    "seq_len": self.dataset.max_length}
            per_token_fn = instantiate(M.output_metric_fns, per_token_ppl_config, partial=True)
            per_token_fn = U.discard_kwargs(per_token_fn)
            self.per_token_fn = per_token_fn

    def metrics(self, x, y, **kwargs):
        """
        Need to modify metrics to include custom metrics
        """
        output_metrics = {
            name: U.discard_kwargs(M.output_metric_fns[name])(x, y, **kwargs)
            for name in self.metric_names if name in M.output_metric_fns
        }
        loss_metrics = {
            name: U.discard_kwargs(M.loss_metric_fns[name])(x, y, self.loss, **kwargs)
            for name in self.metric_names if name in M.loss_metric_fns
        }

        # loop through all custom ppls and add them to output_metrics
        if self.last_k_ppl is not None:
            for key_name, k_ppl_fn in self.custom_ppl_dict.items():
                output_metrics[key_name] = k_ppl_fn(x, y, **kwargs)

        # loop through all custom ppls and add them to output_metrics
        if self.per_token_ppl is not None:
            # returns k ppl values, (averaged over batch)
            per_k_ppl = self.per_token_fn(x, y, **kwargs)

            # loop over ks to log metric
            for ind, k in enumerate(self.per_token_ppl["ks"]):
                key_name = "ppl_at_{}".format(k)
                # k = k-1  # 0 index in the background #commented out by caduceus
                output_metrics[key_name] = per_k_ppl[ind]  # should be in order

        return {**output_metrics, **loss_metrics}


class AdaptiveLMTask(BaseTask):
    def __init__(
            self,
            div_val,
            cutoffs: List[int],
            tie_weights: bool,
            tie_projs: List[bool],
            init_scale=1.0,
            bias_scale=0.0,
            dropemb=0.0,
            dropsoft=0.0,
            **kwargs,
    ):
        super().__init__(**kwargs)
        n_tokens = self.dataset.n_tokens
        d_model = self.model.d_model
        d_output = self.model.d_output

        encoder = AdaptiveEmbedding(
            n_tokens,
            d_model,
            d_model,
            cutoffs=cutoffs,
            div_val=div_val,
            init_scale=init_scale,
            dropout=dropemb,
        )

        if tie_weights:
            assert d_model == d_output
            emb_layers = [i.weight for i in encoder.emb_layers]
        else:
            emb_layers = None

        # Construct decoder/loss
        emb_projs = encoder.emb_projs
        loss = ProjectedAdaptiveLogSoftmax(
            n_tokens, d_output, d_output,
            cutoffs, div_val=div_val,
            tie_projs=tie_projs,
            out_projs=emb_projs,
            out_layers_weights=emb_layers,
            bias_scale=bias_scale,
            dropout=dropsoft,
        )

        self.encoder = encoder
        self.loss = loss


registry = {
    'base': BaseTask,
    'multiclass': MultiClass,
    'lm': LMTask,
    'hg38': HG38Task,
    "masked_multiclass": MaskedMultiClass,
    'regression': Regression,
    'regclass': RegClass,
    'profileclass': ProfileClass,
    'enformer': EnformerTask,
}
