#we will create a utility function for shap called shap_utils.py
#you define several attributes, and then it will load the data to be able to access the shap values

import torch 
import sys
import yaml 
sys.path.append('/data/leslie/sarthak/hyena/hyena-dna/')
from src.tasks.decoders import SequenceDecoder
from src.models.sequence.dna_embedding import DNAEmbeddingModel
from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer
import shap
import numpy as np

class ShapUtils():
    def __init__(self, model_type, ckpt_path, cfg = None, split = 'train', filter=True, ignore_embed = True, percentage_background = 0.01):
        type_list = ['ccre', 'DNase_ctst', 'DNase_allcelltypes', 'DNase']
        if model_type not in type_list:
            raise ValueError('Model type not recognized')
        self.mtype = model_type
        
        #check to see the type, and then load the right tokenizer, class and cfg
        if self.mtype == 'DNase':
            from src.dataloaders.datasets.DNase_dataset import DNaseDataset as DatasetClass
            self.tokenizer = CharacterTokenizer( #make sure to fix the tokenizer too
                characters=['A', 'C', 'G', 'T', 'N', 'S', 'U', 'V', 'W', 'X', 'Y', 'Z'],
                model_max_length=1024 + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side='left'
            )
            if cfg is None:
                cfg = '/data/leslie/sarthak/hyena/hyena-dna/configs/evals/DNase.yaml'
                
        elif self.mtype == 'DNase_allcelltypes':
            from src.dataloaders.datasets.DNase_allcelltypes import DNaseAllCellTypeDataset as DatasetClass
            self.tokenizer = CharacterTokenizer( #make sure to fix the tokenizer too
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=1024 + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side='left'
            )
            if cfg is None:
                cfg = '/data/leslie/sarthak/hyena/hyena-dna/configs/evals/DNase_allcelltypes.yaml'

        elif self.mtype == 'Dnase_ctst':
            from src.dataloaders.datasets.DNase_ctst_dataset import DNaseCtstDataset as DatasetClass
            self.tokenizer = CharacterTokenizer( #make sure to fix the tokenizer too
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=1024 + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side='left'
            )
            if cfg is None:
                cfg = '/data/leslie/sarthak/hyena/hyena-dna/configs/evals/DNase_ctst.yaml'

        else:
            raise ValueError('Model type not recognized')

        #now we load the model and dataset

        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset = DatasetClass(max_length = 1024, split = split, tokenizer=self.tokenizer, rc_aug = False, tokenizer_name='char', add_eos='True', filter = filter)
        cfg = yaml.load(open(cfg, 'r'), Loader=yaml.FullLoader)
        
        train_cfg = cfg['train']  # grab section `train` section of config
        model_cfg = cfg['model_config']  # grab the `model` section of config
        d_output = train_cfg['d_output']
        backbone = DNAEmbeddingModel(**model_cfg)
        # backbone_skip = DNAEmbeddingModel(skip_embedding=True, **model_cfg)
        decoder = SequenceDecoder(model_cfg['d_model'], d_output=d_output, l_output=0, mode='pool')
        state_dict = torch.load(ckpt_path, map_location='cpu')  # has both backbone and decoder
        
        # loads model from ddp by removing prexix to single if necessary
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
        decoder_state_dict['output_transform.weight'] = model_state_dict.pop('decoder.0.output_transform.weight')
        decoder_state_dict['output_transform.bias'] = model_state_dict.pop('decoder.0.output_transform.bias')

        #now adjust the backbone if needed
        if self.mtype == 'DNase':
            embedding1 = torch.nn.Embedding(20, 128)
            # embedding2 = torch.nn.Embedding(20, 128)
            backbone.backbone.embeddings.word_embeddings = embedding1 #again a hack
            # backbone_skip.backbone.embeddings.word_embeddings = embedding2 #again a hack

        # now actually load the state dict to the decoder and backbone separately
        decoder.load_state_dict(decoder_state_dict, strict=True)
        backbone.load_state_dict(model_state_dict, strict=True)
        # backbone_skip.load_state_dict(model_state_dict, strict=True)

        # decoder = decoder.to(self.device)
        # backbone = backbone.to(sdevice)

        self.backbone = backbone.eval()
        self.decoder = decoder.eval()
        # self.backbone_skip = backbone_skip.eval()

        self.create_model(ignore_embed)
        # Net(backbone, decoder)

        #we also can assign the background and test input
        self.assign_train(1/percentage_background)

        #also assign this class that finds the filtered data for the variance
        # filtered_idx_list = []
        # for key in self.dataset.filtered_indices:
        #     # print(self.dataset.filtered_indices[key])
        #     filtered_idx_list.append(self.dataset.filtered_indices[key])
        # self.dnase_filtered = self.dataset.cell_dnase_levels[:,filtered_idx_list]
        #we just added this in the class itself


        #now the last step is to load the data, but that should be done externally from the class
        

    def create_model(self, ignore_embed = False):
        #here we can define the model that will be used for the SHAP analysis
        if ignore_embed:
            self.model = NetNoEmbed(self.backbone, self.decoder).eval() #surprisingly enough, this sets it all to eval!!!!
            #so now self.backbone_skip and self.decoder will be in eval mode! It's incredibly interesting!!
            #recurses through and sets all the submodules to eval!!
        else:
            self.model = Net(self.backbone, self.decoder).eval()

    def assign_train(self, percentage_background = 100):
        #find the length of the dataset, use that to assign the train and test
        if self.mtype == 'DNase_allcelltypes':
            self.length = int(len(self.dataset))
            indices = np.arange(int(len(self.dataset)))
        else:
            self.length = int(len(self.dataset)/self.dataset.cell_types)
            indices = np.arange(int(len(self.dataset)/self.dataset.cell_types))
        np.random.seed(420)
        np.random.shuffle(indices)
        self.background_indices = indices[:int(len(indices)/percentage_background)]
        self.test_input_indices = indices[int(len(indices)/percentage_background):]
        

    def load_data(self, batch_size = 110, remainder = 10): #this class is for a basic test that lets you do a single batch
        loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        batch = next(iter(loader))
        inputs, targets = batch
        self.background = inputs[:batch_size - remainder]
        self.test_input = inputs[batch_size-remainder:]
        self.background_embed = self.backbone.backbone.embeddings.word_embeddings(self.background)
        self.test_input_embed = self.backbone.backbone.embeddings.word_embeddings(self.test_input)

    def load_from_indices(self):
        if self.mtype == 'DNase_allcelltypes': #the easy one wher eit loads it for all the cell types
            background = torch.zeros((len(self.background_indices), 1023))
            # test_input = torch.zeros((len(self.test_input_indices), 1023, 161))
            for i, idx in enumerate(self.background_indices):
                background[i], _ = self.dataset[idx]
            # for i, idx in enumerate(self.test_input_indices):
            #     test_input[i], _ = self.dataset[idx]
        else:
            background = torch.zeros((len(self.background_indices)*161, 1023))
            # test_input = torch.zeros((len(self.test_input_indices)*161, 1023))
            for i, idx in enumerate(self.background_indices): #now we get an index, but this is the cCRE index, need to then loop through all of the cell types
                for j in range(161):
                    background[i*161 + j], _ = self.dataset[idx*161 + j]
            # for i, idx in enumerate(self.test_input_indices):
            #     for j in range(161):
            #         test_input[i*161 + j], _ = self.dataset[idx*161 + j]
        #now we need to embed the background and test input
        self.background = background.long()
        self.background_embed = self.backbone.backbone.embeddings.word_embeddings(self.background)

    def shap_values(self, background = None, test_input = None):
        if background is None:
            background = self.background_embed
        if test_input is None:
            test_input = self.test_input_embed
        self.explainer = shap.DeepExplainer(self.model, background)
        shap_values = self.explainer.shap_values(test_input)
        return shap_values


    def var(self, idx):
        # idx = idx * util.dataset.cell_types #takes into account the fact that we need to find ccre, but we're given cell type
        # seq_idx = int(idx/self.dataset.cell_types)
        #but we are given ccre id so it's easy
        seq_idx = idx
        cCRE_id = self.dataset.array[seq_idx][0] #get the id from the array
        row = self.dataset.cCRE_dict[cCRE_id]
        #now we can calculate the variance using this data
        # print(np.var(dnase_filtered[row,:])) #identical
        
        return np.var(self.dataset.cell_dnase_levels[row,:])


    
         
''' The NetNoEmbed is actually just wrong, we should never use this, the backbone_skip is it instead!!
To run through, start from the data
a,_ = util.dataset[i]. Then b,_ = util.backbone(a) and util.decoder(b) gives output
or we embed it manually and then run through the backbone_skip and decoder isntead!

Actually that all causes a gradient issue, we just had this tiny bug with this, let's redo everything and try again
Issue with NetNoEmbed was what I was passing to the decoder, it was literally just the hidden states, not the output of the backbone
That makes it so that it doesn't even go through the model, so of course the values are the same!!!'''
class NetNoEmbed(torch.nn.Module):
    def __init__(self, backbone, decoder):
        super(NetNoEmbed, self).__init__()
        self.backbone = backbone  # Your pre-defined backbone
        self.decoder = decoder    # Your pre-defined decoder

    def forward(self, x):
        residual = None
        backbone = self.backbone
        # a_embed = backbone.backbone.embeddings.word_embeddings(x)
        a_embed = x #since we will embed it manually
        for layer in backbone.backbone.layers:
            a_embed, residual = layer(a_embed, residual)
        dropped = backbone.backbone.drop_f(a_embed)
        residual = (dropped + residual) if residual is not None else dropped
        hidden_states = backbone.backbone.ln_f(residual.to(dtype=backbone.backbone.ln_f.weight.dtype))
        
        # x = self.backbone(x)  # Pass input through backbone
        x = self.decoder(hidden_states)   # Pass backbone's output through decoder
        return x
    

#let's define a class that takes both elements
class Net(torch.nn.Module):
    def __init__(self, backbone, decoder):
        super(Net, self).__init__()
        self.backbone = backbone  # Your pre-defined backbone
        self.decoder = decoder    # Your pre-defined decoder

    def forward(self, x):
        x, _ = self.backbone(x)  # Pass input through backbone
        x = self.decoder(x)   # Pass backbone's output through decoder
        return x