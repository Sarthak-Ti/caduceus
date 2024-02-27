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
            embedding = torch.nn.Embedding(20, 128)
            backbone.backbone.embeddings.word_embeddings = embedding #again a hack

        # now actually load the state dict to the decoder and backbone separately
        decoder.load_state_dict(decoder_state_dict, strict=True)
        backbone.load_state_dict(model_state_dict, strict=True)

        # decoder = decoder.to(self.device)
        # backbone = backbone.to(sdevice)

        self.backbone = backbone
        self.decoder = decoder

        self.create_model(ignore_embed)
        # Net(backbone, decoder)

        #we also can assign the background and test input
        self.assign_train(1/percentage_background)

        #now the last step is to load the data, but that should be done externally from the class
        

    def create_model(self, ignore_embed = False):
        #here we can define the model that will be used for the SHAP analysis
        if ignore_embed:
            self.model = NetNoEmbed(self.backbone, self.decoder).eval()
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
        x = self.decoder(x)   # Pass backbone's output through decoder
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