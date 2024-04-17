#so loads this in properly
#we will create a utility function for shap called shap_utils.py
#you define several attributes, and then it will load the data to be able to access the shap values

import torch 
import sys
import yaml 
sys.path.append('/data/leslie/sarthak/hyena/hyena-dna/')
from src.tasks.decoders import SequenceDecoder
from src.models.sequence.dna_embedding import DNAEmbeddingModel
from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer
# import shap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logomaker
from tqdm import tqdm

#TODO there are many NotImplementedErrors that need to be fixed, search for self.dataset to see where it's used and fix it
#TODO there is one difference based on the opposite signs function that is in shap_testing5, but not in the actual ism_utils.py, copy paste it

class ISMUtils():
    def __init__(self, model_type, ckpt_path, cfg = None, split = 'train', filter=True, classification = False):
        type_list = ['ccre', 'DNase_ctst', 'DNase_allcelltypes', 'DNase']
        if model_type not in type_list:
            raise ValueError('Model type not recognized')
        self.mtype = model_type
        if cfg is not None:
            cfg = '/data/leslie/sarthak/hyena/hyena-dna/configs/evals/'+cfg
        self.classification = classification
        
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
                if self.classification:
                    cfg = '/data/leslie/sarthak/hyena/hyena-dna/configs/evals/DNase_allcelltypes_classification.yaml'
                else:
                    cfg = '/data/leslie/sarthak/hyena/hyena-dna/configs/evals/DNase_allcelltypes.yaml'
                 

        elif self.mtype == 'DNase_ctst':
            from src.dataloaders.datasets.DNase_ctst_dataset import DNaseCtstDataset as DatasetClass
            self.tokenizer = CharacterTokenizer( #make sure to fix the tokenizer too
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=1024 + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side='left'
            )
            if cfg is None:
                if self.classification:
                    cfg = '/data/leslie/sarthak/hyena/hyena-dna/configs/evals/DNase_ctst_classification.yaml'
                else:
                    cfg = '/data/leslie/sarthak/hyena/hyena-dna/configs/evals/DNase_ctst.yaml'

        else:
            raise ValueError('Model type not recognized')

        #now we load the model and dataset

        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset = DatasetClass(max_length = 1024, split = split, tokenizer=self.tokenizer, rc_aug = False, tokenizer_name='char', add_eos='True', filter = filter, classification=self.classification)
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

        self.backbone = backbone.eval()
        self.decoder = decoder.eval()

        self.bed = pd.read_csv('/data/leslie/sarthak/data/GRCh38-cCREs.bed', header=None, delimiter='\t').to_numpy()
        self.middle = 1023//2 #just used for the logo plots
        self.ccre_id_dict = None

    def calculate_ISM(self,ccre, cuda = False, return_out = False, progress_bar = False, stop=False, b_size = 8):
        #does ISM for that ccre, based on the model type
        #Does what the ISM.py and ISM_allcelltypes.py files do but for one ccre at a time
        #ccre should be the index of the ccre, not the index to get that dataset
        # start = time.time()
        device = "cuda:0" if cuda else "cpu"
        backbone = self.backbone.to(device)
        decoder = self.decoder.to(device)
        #first have to load in the batch
        #if it's the different groups have very different dataset classes
        if self.mtype == 'DNase' or self.mtype == 'DNase_ctst':
            ccre = 161*ccre
            ccre_list = []
            # out_list = []
            # class_list = []
            for i in range(161):
                a,b = self.dataset[ccre+i]
                ccre_list.append(a)
                # if self.classification:
                    # raise NotImplementedError('This is not implemented yet, need to consider how to do ISM with the classification model')
                    # class_list.append(b[0].item())
                    # b = b[1] #just ignore the classification stuff?
                # out_list.append(b.item())
            ccre_list = torch.stack(ccre_list)
        else:
            ccre_list = self.dataset[ccre][0].unsqueeze(0)
            # class_list = self.dataset[ccre][1][0] #the classification stuff
        # print(class_list)
        temp,_ = backbone(ccre_list.to(device)) #pass whole ccre as a batch
        out = decoder(temp)
        seqlen = ccre_list.shape[1] #because it's batch x seq
        dim_out = out.shape[1]//2
        #also a dimension for ccre_list.shape[0]
        dim_celltypes = ccre_list.shape[0]

        if self.classification:
            out_class = out[:,:dim_out]
            out = out[:,dim_out:] #just ignore the classification stuff? for now maybe and we just look at the regression outputs?
            #but if it thinks that the thing is closed, we should keep it as 0... or maybe set it to some other value? idk
            #i say keep it but we know that we can mask it, no let's mask for reg, but still need to use for class...
            #and let's keep the class out
            #we can mask out the reg if needed using the actual outputs, if it's -10, ignore it maybe?
            out_class_gt = out_class.detach().reshape(1,1,161)
            ISM_class = torch.ones((4,seqlen,161)).to(device)*out_class_gt
            out_class_gt_np = out_class.detach().cpu().numpy().reshape(1,1,161)
            
        out_gt = out.detach().reshape(1,1,161)
        ISM_results = torch.ones((4,seqlen,161)).to(device)*out_gt
        out_gt_np = out.detach().cpu().numpy().reshape(1, 1, 161) #the initial output of the model before ism
        # if self.classification:
        #     mask_idx = np.array(class_list) == 1 #if it's open
        #     #now we make this a matrix of the same size as the output which will be 3x1x161 or just 3x161
        #     mask = np.zeros((1,1,161))
        #     mask[:,:,mask_idx] = 1
        # print(mask)
        if self.mtype == 'DNase' or self.mtype == 'DNase_ctst': #we just need the first sequence, don't care about the useless stuff
            seq = ccre_list[0]
        else:
            seq = ccre_list.squeeze() #get rid f the embedding dimension
        token_list = [7,8,9,10] #tokenize this list first
        mutations = {
            7: np.array([8,9,10]),
            8: np.array([7,9,10]),
            9: np.array([7,8,10]),
            10: np.array([7,8,9])
        }
        ccre_list_gpu = ccre_list.to(device)

        if progress_bar:
            iterator = tqdm(enumerate(seq), total = len(seq))
        else:
            iterator = enumerate(seq)

        with torch.no_grad():
            for idx, nucleotide in iterator: 
                if nucleotide not in token_list:
                    continue #basically skips this and none of it is updated, because is a weird tooken, whether permutation, ctst or something else
                temp_token_list = mutations[nucleotide.item()]
                class_list = []
                results_list = []
                for idx2,j in enumerate(temp_token_list):
                    temp_seq = ccre_list_gpu.clone() #already put on the gpu
                    # if self.mtype in ['DNase', 'DNase_ctst']:
                    #     temp_seq[:,idx] = j #should be 161x1024 or 1023. 
                    # else:
                    #     temp_seq[idx] = j #should be 1x1023 because we unsqueezed
                    temp_seq[:,idx] = j #now that it's unsqueezed, we can just do this, affects all the sequences or just the one sequence
                    a,_ = backbone(temp_seq)
                    out = decoder(a)
                    if self.classification:
                        class_list.append(out[:,:dim_out].detach()) #161 x 1 if ctst, 1x161 if multitasking, squeeze fixes it
                        results_list.append(out[:,dim_out:].detach()) 
                    else:
                        results_list.append(out.detach())

                ISM_results[temp_token_list-7,idx,:] = torch.stack(results_list).squeeze().to(device) #baasically turns to 3 x 1 x 161 or 3 x 161 x 1 then squeezes it
                if self.classification:
                    ISM_class[temp_token_list-7,idx,:] = torch.stack(class_list).squeeze().to(device)
        ISM_results_normalized = (ISM_results - out_gt).detach().cpu().numpy() #is mutated - reference
        if self.classification:
            #first we do sigmoid then subtraction
            ISM_class = torch.sigmoid(ISM_class)
            out_class_gt = torch.sigmoid(out_class_gt)
            ISM_class_normalized = (ISM_class-out_class_gt).detach().cpu().numpy()
            # ISM_class_normalized = ISM_class.detach().cpu().numpy() - out_class_gt_np
            # ISM_results_normalized *= mask
        if self.classification and return_out:
            return ISM_results_normalized, ISM_class_normalized, out_gt_np, out_class_gt_np
        elif self.classification:
            return ISM_results_normalized, ISM_class_normalized 
        elif return_out: #but this is optional
            return ISM_results_normalized, out_gt_np
        return ISM_results_normalized

    def var(self, idx):
        #This is the idx for the ccre, so if you want it for ccre 2 it's just idx == 2. Not 161*idx
        seq_idx = idx
        cCRE_id = self.dataset.array[seq_idx][0] #get the id from the array
        row = self.dataset.cCRE_dict[cCRE_id]
        #now we can calculate the variance using this data
        # print(np.var(dnase_filtered[row,:])) #identical
        
        return np.var(self.dataset.cell_dnase_levels[row,:])

    def output(self,idx):
        #given the index, this finds the associated output (single value)
        #this literally just puts it thorugh the model, but it's like 2 lines of code and really easy to do yourself
        a,b = self.dataset[idx]
        temp,_ = self.backbone(a.unsqueeze(0))
        out = self.decoder(temp)

        if self.mtype == 'DNase':
            print(f'predicted output: {out}, actual output: {b}')
            return b, out

        elif self.mtype == 'DNase_allcelltypes':
            print(f'predicted output: {out[0,0]}, actual output: {b[0]}')
            return b[0], out[0,0]

    def output_all(self,idx, print_out = False):
        #given the index, this finds the associated output (all values averaged across the cell types for a singel ccre)
        #if it's DNase, then we need to do it for all the cell types, here the input idx should not be 161 times any number, rather just the ccre number
        if self.mtype == 'DNase' or self.mtype == 'DNase_ctst':
            idx = 161*idx
            out_list = []
            target_list = []
            for i in range(161):
                a,b = self.dataset[idx+i]
                temp,_ = self.backbone(a.unsqueeze(0))
                out = self.decoder(temp)
                out_list.append(out.item())
                if isinstance(b,tuple):
                    raise NotImplementedError('This is not implemented yet, need to consider how to do handle these outputs and what we want to display')
                target_list.append(b.item())
            out = np.mean(out_list)
            if print_out:
                print(f'predicted output mean: {out}, actual output mean: {np.mean(target_list)}')
            #but return the whole lists
            return np.array(target_list), np.array(out_list)

        if self.mtype == 'DNase_allcelltypes':
            a,b = self.dataset[idx]
            temp,_ = self.backbone(a.unsqueeze(0))
            out = self.decoder(temp)
            print(f'predicted output mean: {torch.mean(out)}, actual output mean: {torch.mean(b)}')
            return b.detach().numpy(), out.detach().numpy()

    def plot_singlecelltype(self, data, celltype = 0):
        #given the index, this finds the associated output (single value)
        #we can either take in the actual data, or the ccre number
        #actual data must be formatted the same as the saved data
        if type(data) == int:
            data_loaded = np.load(f'/data/leslie/sarthak/hyena/hyena-dna/shap_analysis/saved_ISM/ISM_run1_normalized_{data}.npy')
        elif type(data) == np.ndarray: #you can also just give it the data and then it doesn't need to load it
            data_loaded = data
        fig, ax = plt.subplots(1,1, figsize = (20,10))
        sns.heatmap(data_loaded[:,:,celltype], cmap = 'seismic', center = 0)
        plt.yticks([0.5,1.5,2.5,3.5],['A','C','G','T'])
        plt.xlabel('sequence position')
        plt.title(f'ISM scores for the {celltype} cell type of the {data} ccre')
        plt.show()

    def plot_all(self,data):
        #once again can give the ccre or the actual data
        if type(data) == int:
            data_loaded = np.load(f'/data/leslie/sarthak/hyena/hyena-dna/shap_analysis/saved_ISM/ISM_run1_normalized_{data}.npy')
        elif type(data) == np.ndarray:
            data_loaded = data
        fig, ax = plt.subplots(1,1, figsize = (20,10))
        sns.heatmap(data_loaded.sum(axis = 0).T, cmap = 'seismic', center = 0)
        plt.ylabel('celltype')
        plt.xlabel('sequence position')
        plt.title('ISM scores for all cell types single ccre')
        plt.show()

    def default_heights(self,data):
        #it will sum across the nucelotides and average over the celltypes
        if type(data) == int:
            data_loaded = np.load(f'/data/leslie/sarthak/hyena/hyena-dna/shap_analysis/saved_ISM/ISM_run1_normalized_{data}.npy')
        elif type(data) == np.ndarray:
            data_loaded = data
        return data_loaded.sum(axis = 0).mean(axis = 1)

    def logoplot(self,ccre,heights = None, startend=None,flip_heights = False): #potentially we add the title as an input into the model
        #so given the ccre number and the heights, this will plot the logo plot
        #note ccre should be the ccre number regardless of the model type
        #heights must be provided, is usually the sum across the axis=1 for the data that is used for plot_all, uses same start and end
        if self.mtype == 'DNase' or self.mtype == 'DNase_ctst':
            ccre = 161*ccre
        a,b = self.dataset[ccre]
        seq=self.dataset.tokenizer.decode(a)
        if startend is None and self.mtype == 'DNase':
            start = self.middle-50+4
            end = self.middle+50+4
        elif startend is None and self.mtype == 'DNase_allcelltypes':
            start = self.middle-50
            end = self.middle+50
        elif self.mtype == 'DNase': #if specify it, add 4 to the dnase, just so aligns with the multitasking model
            start = startend[0]+4
            end = startend[1]+4
        elif self.mtype == 'DNase_ctst':
            start = startend[0]+1
            end = startend[1]+1
        else:
            start = startend[0]
            end = startend[1]

        #now we can define cut_seq
        cut_seq = seq[start:end]
                    
        if heights is None:
            raise ValueError('Need to provide heights')

        #now we define cut heights
        heights = heights[start:end]
        if flip_heights:
            heights = -heights
        logo_df = pd.DataFrame(0, index=np.arange(len(cut_seq)), columns=list(set(cut_seq)), dtype=float)

        # Fill the DataFrame with heights, converting heights to float if necessary
        for i, symbol in enumerate(cut_seq):
            logo_df.loc[i, symbol] = heights[i]

        # Generate the sequence logo
        # fig, ax = 
        # fig.set_size_inches(20, 3)  # Set the desired width and height here        
        logo = logomaker.Logo(logo_df, color_scheme='classic', flip_below = True, figsize = (20,3))
        plt.title('Sequence logo')
        # fig.show()
    
    def find_ccre_type(self, idx):
        #this function will use the ccre id to find the specific type
        #uses the ccre id, not the dataset index, just like for the var
        #first get the id
        #actually, first it generates the dictionary of ids
        if self.ccre_id_dict is None: #checks to see if the dictionary is none
            self.ccre_id_dict = {}
            for i,item in enumerate(self.bed[:,3]): #goes through 3rd column of bed file which is the ccre id, now instead of finding that index, goes into dict
                self.ccre_id_dict[item] = i #now uses the ccre_id to find the line
        ccre_id = self.dataset.array[idx][0]
        # line = np.where(self.bed[:,3] == ccre_id)
        line = self.ccre_id_dict[ccre_id]
        return self.bed[line, -1]

    def onehot_ism(self, idx, name = None, ism=None, split = 'train'):
        #this function takes in an index, the index of the cCRE, not the dataset index
        #also takes in either the name or the actual ism results that have been loaded, depend son if you use the same naming scheme or not
        #It outputs the ISM results in one hot encoded format, tthe one hot encoded results, and the sequencing
        seq_idx = idx #there's no difference, it's based on the ccre that's it! same self.dataset.array between them, idx is the ccre
        if self.mtype == 'DNase' or self.mtype == 'DNase_ctst':
            # seq_idx = int(idx/161)
            start = 1
        else:
            # seq_idx = idx
            start = 0
        seq = self.dataset.array[seq_idx,2]
        seq = seq[:-1].upper() #because it'ts length 1024, but we use 1023
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        num_seq = np.array([mapping[nuc] for nuc in seq])
        one_hot = np.zeros((len(seq), 4))
        one_hot[np.arange(len(seq)), num_seq] = 1
        one_hot_expanded = np.repeat(one_hot[np.newaxis,:,:], 161, axis=0)
        if ism is None:
            ism = np.load(f'/data/leslie/sarthak/hyena/hyena-dna/shap_analysis/most_variable_cCREs2/{split}/{name}_{idx}_reg.npy')
        ism = ism[:,start:,:]
        ismsum = np.sum(ism, axis=0)
        ismsumt = ismsum.T
        ismsumt = ismsumt[:,:, np.newaxis]
        final = one_hot_expanded * ismsumt
        #we now have to transpose both matrices to work with modiscolite, the above code makes it work for normal modisco which is one_hot last
        #modiscolite is length last
        final = np.transpose(final,(0,2,1))
        one_hot_expanded = np.transpose(one_hot_expanded,(0,2,1))
        return final, one_hot_expanded, seq

def multi_cluster_difference(utils_list, results_list, true_values, name_list, cluster_index = 0, return_vals = False):
    #utils_list is the list of utilities
    #results_list is the list of the results, likely the loading of the .npy files
    #look at shap_analysis5.ipynb for some examples at the bottom
    #true values is the true values, likely the output of the multitasking dataset or output_all function
    #and the name list is the list of the names of the models that you want in the plot
    #here cluster index is used to find which one clusters but also which one we find the difference compared to
    #so like if 0, that one will be all zeros and is the one that is used for clustering
    
    global_min = min([i.min() for i in results_list])
    global_max = max([i.max() for i in results_list])
    #find the maximum
    max1 = global_max - global_min
    max2 = np.abs(global_min-global_max)
    max3 = max(max1,max2)

    #now we set it 
    global_min = -max3
    global_max = max3

    #uses index to determine which model in results list is used for clustering, then we align the sequence accordingly
    i = results_list[cluster_index]
    temp_values = i.sum(axis=0).T
    if utils_list[cluster_index].mtype == 'DNase':
        temp_values = temp_values[:,7:]
    elif utils_list[cluster_index].mtype == 'DNase_allcelltypes':
        temp_values = temp_values[:,3:-4]
    elif utils_list[cluster_index].mtype == 'DNase_ctst':
        temp_values = temp_values[:,4:-4]
    
    g = sns.clustermap(temp_values, cmap = 'seismic', center = 0,col_cluster=False, vmin=global_min, vmax=global_max, figsize=(20,5), cbar_pos = (.02,.1,.05,.8))
    row_order = g.dendrogram_row.reordered_ind
    real = temp_values[row_order]
    plt.close(g.figure)

    fig = plt.figure(figsize=(20, 5*len(results_list)))
    # from matplotlib.colors import TwoSlopeNorm

    # Dimensions for main and secondary heatmaps
    main_heatmap_width = 0.85
    secondary_heatmap_width = 0.025
    colorbar_width = .025
    gap_between_heatmaps = 0.01
    height = 0.8/len(results_list)  # Adjust based on your preference for the subplot height
    vertical_gap = 0.2/len(results_list)  # Gap between rows

    main_left = 0.05  # Starting position of the main heatmap (left)
    secondary_left = main_left + main_heatmap_width + gap_between_heatmaps  # Starting position of secondary heatmap (left)
    tertiary_left = secondary_left + colorbar_width + gap_between_heatmaps  # Starting position of the colorbars (left)
    total_vertical_space_per_heatmap = height + vertical_gap
    temp_true = true_values[row_order].unsqueeze(1)

    difference_list = []

    for j, i in enumerate(results_list):
        # Calculate positions of main and secondary heatmaps    
        bottom = 0.1 + (height + vertical_gap) * (len(results_list) - 1 - j)  # Starting position from bottom, adjust for each row
        # bottom = 1 - (vertical_gap + (j+1) * total_vertical_space_per_heatmap) #doesn't work, let's figure out why

        # Create axes for main heatmap
        ax_main = fig.add_axes([main_left, bottom, main_heatmap_width, height])
        ax_secondary = fig.add_axes([secondary_left, bottom, secondary_heatmap_width, height])

        temp_values = i.sum(axis=0).T
        #now let's align them all, required to subtract them
        if utils_list[j].mtype == 'DNase':
            temp_values = temp_values[:,7:]
        elif utils_list[j].mtype == 'DNase_allcelltypes':
            temp_values = temp_values[:,3:-4]
        elif utils_list[j].mtype == 'DNase_ctst':
            temp_values = temp_values[:,4:-4]
        
        if j == len(results_list)-1:
            continue
        sns.heatmap(temp_values[row_order]-real, cmap='seismic', center=0, ax=ax_main, cbar=False, vmin=global_min, vmax=global_max)
        difference_list.append(temp_values[row_order]-real)
        
        ax_main.set_xlabel('sequence position')
        ax_main.set_title(f'{name_list[j]} model minus {name_list[cluster_index]}')
        ax_main.set_yticks(range(0,161,5))
        ax_main.set_yticklabels(row_order[::5])

        # Create axes for secondary heatmap (true_values)
        #now we can also organize the true_values
        
        sns.heatmap(temp_true, cmap='viridis', center=0, ax=ax_secondary, cbar=False, vmin=min(true_values), vmax=max(true_values))
        # Optional: Remove y-axis labels for the secondary heatmap if they're redundant
        ax_secondary.set_yticks([])
        ax_secondary.set_xticks([])
        ax_secondary.set_xlabel('accessibility')

    fig.subplots_adjust(right=0.9)
    cbar_ax1 = fig.add_axes([tertiary_left, 0.1, 0.025, 0.45])  # Adjust the dimensions as needed
    # sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(true_values), vmax=max(true_values)))
    # fig.colorbar(sm, cax=cbar_ax1)
    cbar_ax = fig.add_axes([tertiary_left, 0.6, 0.025, 0.45])  # Adjust the dimensions as needed
    
    sns.heatmap(temp_values[row_order]-real, cmap='seismic', center=0, ax=ax_main, vmin=global_min, vmax=global_max, cbar_ax = cbar_ax)
    difference_list.append(temp_values[row_order]-real)
    ax_main.set_xlabel('sequence position')
    ax_main.set_title(f'{name_list[j]} model minus {name_list[cluster_index]}')    #now set the ytick labels to row_order
    ax_main.set_yticks(range(0,161,5))
    ax_main.set_yticklabels(row_order[::5])

    sns.heatmap(temp_true, cmap='viridis', center=0, ax=ax_secondary, vmin=min(true_values), vmax=max(true_values), cbar_ax = cbar_ax1)
    ax_secondary.set_yticks([])
    ax_secondary.set_xticks([])
    ax_secondary.set_xlabel('accessibility')
    if return_vals:
        return difference_list


def multi_ism(results_list, true_values, name_list, utils_list = None, align_indices = True):
    #now make it work with the heatmaps, we will maek it thinner and add the colorbars
    fig = plt.figure(figsize=(20, 5*len(results_list)))
    # from matplotlib.colors import TwoSlopeNorm

    # Dimensions for main and secondary heatmaps
    main_heatmap_width = 0.85
    secondary_heatmap_width = 0.025
    colorbar_width = .025
    gap_between_heatmaps = 0.01
    height = 0.8/len(results_list)  # Adjust based on your preference for the subplot height
    vertical_gap = 0.2/len(results_list)  # Gap between rows
    global_min = min([i.min() for i in results_list])
    global_max = max([i.max() for i in results_list])

    

    main_left = 0.05  # Starting position of the main heatmap (left)
    secondary_left = main_left + main_heatmap_width + gap_between_heatmaps  # Starting position of secondary heatmap (left)
    tertiary_left = secondary_left + colorbar_width + gap_between_heatmaps  # Starting position of the colorbars (left)
    total_vertical_space_per_heatmap = height + vertical_gap
    # temp_true = true_values[row_order].unsqueeze(1)
    for j, i in enumerate(results_list):
        # Calculate positions of main and secondary heatmaps    
        bottom = 0.1 + (height + vertical_gap) * (len(results_list) - 1 - j)  # Starting position from bottom, adjust for each row
        # bottom = 1 - (vertical_gap + (j+1) * total_vertical_space_per_heatmap) #doesn't work, let's figure out why

        # Create axes for main heatmap
        ax_main = fig.add_axes([main_left, bottom, main_heatmap_width, height])
        ax_secondary = fig.add_axes([secondary_left, bottom, secondary_heatmap_width, height])

        temp_values = i.sum(axis=0).T
        #now let's align them all
        if align_indices:
            if utils_list[j].mtype == 'DNase':
                temp_values = temp_values[:,7:]
            elif utils_list[j].mtype == 'DNase_allcelltypes':
                temp_values = temp_values[:,3:-4]
            elif utils_list[j].mtype == 'DNase_ctst':
                temp_values = temp_values[:,4:-4]

        if j == len(results_list)-1:
            continue

        sns.heatmap(temp_values, cmap='seismic', center=0, ax=ax_main, cbar=False, vmin=global_min, vmax=global_max)
        ax_main.set_xlabel('sequence position')
        ax_main.set_title(f'{name_list[j]} model')

        # Create axes for secondary heatmap (true_values)
        sns.heatmap(true_values.unsqueeze(1), cmap='viridis', center=0, ax=ax_secondary, cbar=False, vmin=min(true_values), vmax=max(true_values))
        # Optional: Remove y-axis labels for the secondary heatmap if they're redundant
        ax_secondary.set_yticks([])
        ax_secondary.set_xticks([])
        ax_secondary.set_xlabel('accessibility')

    fig.subplots_adjust(right=0.9)
    cbar_ax1 = fig.add_axes([tertiary_left, 0.1, 0.025, 0.45])  # Adjust the dimensions as needed
    # sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(true_values), vmax=max(true_values)))
    # fig.colorbar(sm, cax=cbar_ax1)
    cbar_ax = fig.add_axes([tertiary_left, 0.6, 0.025, 0.45])  # Adjust the dimensions as needed
    # sm = plt.cm.ScalarMappable(cmap='seismic', norm=plt.Normalize(vmin=global_min, vmax=global_max), center=0)
    # norm = TwoSlopeNorm(vmin=global_min, vcenter=0, vmax=global_max)
    # sm = plt.cm.ScalarMappable(cmap='seismic', norm=norm)
    # fig.colorbar(sm, cax=cbar_ax)

    #let's redraw the last one so we can easily draw the colorbars
    sns.heatmap(temp_values, cmap='seismic', center=0, ax=ax_main, vmin=global_min, vmax=global_max, cbar_ax = cbar_ax)
    ax_main.set_xlabel('sequence position')
    ax_main.set_title(f'{name_list[j]} model')

    sns.heatmap(true_values.unsqueeze(1), cmap='viridis', center=0, ax=ax_secondary, vmin=min(true_values), vmax=max(true_values), cbar_ax = cbar_ax1)
    ax_secondary.set_yticks([])
    ax_secondary.set_xticks([])
    ax_secondary.set_xlabel('accessibility')


def multi_cluster(results_list, true_values, name_list, utils_list = None, cluster_index = 0, align_indices = True):
    global_min = min([i.min() for i in results_list])
    global_max = max([i.max() for i in results_list])

    #uses index to determine which model in results list is used for clustering
    i = results_list[cluster_index]
    g = sns.clustermap(i.sum(axis=0).T, cmap = 'seismic', center = 0,col_cluster=False, vmin=global_min, vmax=global_max, figsize=(20,5), cbar_pos = (.02,.1,.05,.8))
    row_order = g.dendrogram_row.reordered_ind
    plt.close(g.figure)

    fig = plt.figure(figsize=(20, 5*len(results_list)))
    # from matplotlib.colors import TwoSlopeNorm

    # Dimensions for main and secondary heatmaps
    main_heatmap_width = 0.85
    secondary_heatmap_width = 0.025
    colorbar_width = .025
    gap_between_heatmaps = 0.01
    height = 0.8/len(results_list)  # Adjust based on your preference for the subplot height
    vertical_gap = 0.2/len(results_list)  # Gap between rows
    global_min = min([i.min() for i in results_list])
    global_max = max([i.max() for i in results_list])

    main_left = 0.05  # Starting position of the main heatmap (left)
    secondary_left = main_left + main_heatmap_width + gap_between_heatmaps  # Starting position of secondary heatmap (left)
    tertiary_left = secondary_left + colorbar_width + gap_between_heatmaps  # Starting position of the colorbars (left)
    total_vertical_space_per_heatmap = height + vertical_gap
    temp_true = true_values[row_order].unsqueeze(1)

    for j, i in enumerate(results_list):
        # Calculate positions of main and secondary heatmaps    
        bottom = 0.1 + (height + vertical_gap) * (len(results_list) - 1 - j)  # Starting position from bottom, adjust for each row
        # bottom = 1 - (vertical_gap + (j+1) * total_vertical_space_per_heatmap) #doesn't work, let's figure out why

        # Create axes for main heatmap
        ax_main = fig.add_axes([main_left, bottom, main_heatmap_width, height])
        ax_secondary = fig.add_axes([secondary_left, bottom, secondary_heatmap_width, height])

        temp_values = i.sum(axis=0).T
        if align_indices:
            if utils_list[j].mtype == 'DNase':
                temp_values = temp_values[:,7:]
            elif utils_list[j].mtype == 'DNase_allcelltypes':
                temp_values = temp_values[:,3:-4]
            elif utils_list[j].mtype == 'DNase_ctst':
                temp_values = temp_values[:,4:-4]

        if j == len(results_list)-1:
            continue

        sns.heatmap(temp_values[row_order], cmap='seismic', center=0, ax=ax_main, cbar=False, vmin=global_min, vmax=global_max)
        ax_main.set_xlabel('sequence position')
        ax_main.set_title(f'{name_list[j]} model')
        ax_main.set_yticks(range(0,161,5))
        ax_main.set_yticklabels(row_order[::5])

        # Create axes for secondary heatmap (true_values)
        # temp_true = true_values[row_order].unsqueeze(1)
        sns.heatmap(temp_true, cmap='viridis', center=0, ax=ax_secondary, cbar=False, vmin=min(true_values), vmax=max(true_values))
        # Optional: Remove y-axis labels for the secondary heatmap if they're redundant
        ax_secondary.set_yticks([])
        ax_secondary.set_xticks([])
        ax_secondary.set_xlabel('accessibility')

    fig.subplots_adjust(right=0.9)
    cbar_ax1 = fig.add_axes([tertiary_left, 0.1, 0.025, 0.45])  
    cbar_ax = fig.add_axes([tertiary_left, 0.6, 0.025, 0.45])  

    #let's redraw the last one so we can easily draw the colorbars
    sns.heatmap(temp_values[row_order], cmap='seismic', center=0, ax=ax_main, vmin=global_min, vmax=global_max, cbar_ax = cbar_ax)
    ax_main.set_xlabel('sequence position')
    ax_main.set_title(f'{name_list[j]} model')
    #now set the ytick labels to row_order
    ax_main.set_yticks(range(0,161,5))
    ax_main.set_yticklabels(row_order[::5])

    sns.heatmap(temp_true, cmap='viridis', center=0, ax=ax_secondary, vmin=min(true_values), vmax=max(true_values), cbar_ax = cbar_ax1)
    ax_secondary.set_yticks([])
    ax_secondary.set_xticks([])
    ax_secondary.set_xlabel('accessibility')


def multi_logo(utils_list,title_list, ccre, startend=None, heights_list = None, results_list = None):
    # utils list should be a list of the ISMUtils class
    # title list should be a list of the titles you want
    # ccre is the ccre number of that split
    # startend is the start and end of the sequence, if none, will use the middle 100. Note that it adjusts DNase and DNase_ctst slightly, it aligns to the multitasking model!
    # heights list is the list of heights, if none, will use the default heights which is just the mean over all celltypes of the ISM scores
    # results list is a list of results, and that is used as the heights if no heights are provided, but one must be given
    fig, ax = plt.subplots(len(utils_list),1, figsize = (20,len(utils_list)*3))
    tempccre = ccre
    for j,i in enumerate(utils_list):
        if i.mtype == 'DNase' or i.mtype == 'DNase_ctst':
            tempccre = 161*ccre
        else:
            tempccre = ccre
        a,_ = i.dataset[tempccre]

        #now we will check to see if it's none, if so we will use the middle
        if startend is None:
            start = i.middle-50
            end = i.middle+50
        else:
            start = startend[0]
            end = startend[1]
        
        #now we can adjust the start and end
        #note we could cut the sequence, but this is literally just identical, it doesn't matter at all
        if i.mtype == 'DNase': #if specify it, add 4 to the dnase, just so aligns with the multitasking model
            start = start+4
            end = end+4
        try:
            seq=i.dataset.tokenizer.decode(a)
        except:
            seq=i.dataset.tokenizer.decode(a[1:]) #hack to deal with DNase_ctst which is same sequence as multitasking but with an extra token for the celltype

        #now we can define cut_seq
        cut_seq = seq[start:end]
        if heights_list is None:
            heights = i.default_heights(results_list[j])
        else:
            heights = heights_list[j]

        #now we define cut heights
        heights = heights[start:end]
        if True:
            heights = -heights
        logo_df = pd.DataFrame(0, index=np.arange(len(cut_seq)), columns=list(set(cut_seq)), dtype=float)

        # Fill the DataFrame with heights, converting heights to float if necessary
        for i, symbol in enumerate(cut_seq):
            logo_df.loc[i, symbol] = heights[i]

        # Generate the sequence logo
        logo = logomaker.Logo(logo_df, color_scheme='classic', flip_below = True, ax = ax.flatten()[j])
        #now set the title
        ax.flatten()[j].set_title(f'{title_list[j]} model')
        ax.flatten()[j].set_xlim(start, end)
    plt.tight_layout()
    return fig, ax

def celltype_logo(utils, ccre, celltype_idx, heights_all, startend=None, true_values = None):
    # utils is the single utility for the one we want
    # ccre is the ccre number of that split
    # startend is the start and end of the sequence, if none, will use the middle 100. Note that it adjusts DNase and DNase_ctst slightly, it aligns to the multitasking model!
    # heights_all is basically just the saved array of ISM values or can be something else
    fig, ax = plt.subplots(len(celltype_idx),1, figsize = (15,len(celltype_idx)*3))
    celltypesfile = '/data/leslie/sarthak/data/cCRE_celltype_matrices/cell_types_filtered.txt'
    # if true_values is None:
    #     utils.dataset[ccre][1]
    #now load in the celltypes
    celltypes = []
    with open(celltypesfile) as f:
        for line in f:
            celltypes.append(line.strip())
    #first find global min and max
    global_min = np.max(heights_all) #because positive and negative get flipped!!
    global_max = np.min(heights_all)
    for j,i in enumerate(celltype_idx):
        #what we do is first get the sequence
        if utils.mtype == 'DNase' or utils.mtype == 'DNase_ctst':
            tempccre = 161*ccre
        else:
            tempccre = ccre
        a,_ = utils.dataset[tempccre]
        #now we will check to see if it's none, if so we will use the middle
        if startend is None:
            start = utils.middle-50
            end = utils.middle+50
        else:
            start = startend[0]
            end = startend[1]

        if utils.mtype == 'DNase':
            start = start+4
            end = end+4
        try:
            seq=utils.dataset.tokenizer.decode(a)
        except:
            seq=utils.dataset.tokenizer.decode(a[1:])
        cut_seq = seq[start:end]
        #in this function heights is required, generally just the results used for ism

        #now we define cut heights
        heights = heights_all[start:end,i] #will be a matrix of values, we take just the row of the provided celltype_idx
        if True:
            heights = -heights
        logo_df = pd.DataFrame(0, index=np.arange(len(cut_seq)), columns=list(set(cut_seq)), dtype=float)
        # print(heights)

        # Fill the DataFrame with heights, converting heights to float if necessary
        
        for k, symbol in enumerate(cut_seq):
            logo_df.loc[k, symbol] = heights[k]

        # Generate the sequence logo
        logo = logomaker.Logo(logo_df, color_scheme='classic', flip_below = True, ax = ax.flatten()[j])
        #now set the title
        try:
            ax.flatten()[j].set_title(f'{celltypes[i]}, true={true_values[i]}') #if you provide true values then it works
        except:
            ax.flatten()[j].set_title(f'{celltypes[i]}')
        ax.flatten()[j].set_xticks(np.arange(0,end-start,int((end-start)/5)))
        ax.flatten()[j].set_xticklabels(np.arange(start,end,int((end-start)/5)))
        ax.flatten()[j].set_ylim(global_min,global_max)
    plt.tight_layout()
    return fig, ax

#new function to compare them
def celltype_compare_logo(utils_list, ccre, celltype_idx, heights_all_list, startend=None, true_values = None):
    # utils is the list of utilities in order multitasking, ctst
    # ccre is the ccre number of that split
    # startend is the start and end of the sequence, if none, will use the middle 100. Note that it adjusts DNase and DNase_ctst slightly, it aligns to the multitasking model!
    # heights_all is basically just the saved array of ISM values or can be something else
    fig, ax = plt.subplots(len(celltype_idx),2, figsize = (15,len(celltype_idx)*3))
    celltypesfile = '/data/leslie/sarthak/data/cCRE_celltype_matrices/cell_types_filtered.txt'
    # if true_values is None:
    #     utils.dataset[ccre][1]
    #now load in the celltypes
    celltypes = []
    with open(celltypesfile) as f:
        for line in f:
            celltypes.append(line.strip())
    #first find global min and max
    heights_all = np.concatenate(heights_all_list, axis=0)
    global_min = max(i.max() for i in heights_all_list) #because positive and negative get flipped!!
    global_max = min(i.min() for i in heights_all_list)
    # print(global_min, global_max)
    for util_counter,utils in enumerate(utils_list):
        heights_all = heights_all_list[util_counter]
        for j,i in enumerate(celltype_idx):
            #what we do is first get the sequence
            if utils.mtype == 'DNase' or utils.mtype == 'DNase_ctst':
                tempccre = 161*ccre
            else:
                tempccre = ccre
            a,_ = utils.dataset[tempccre]
            #now we will check to see if it's none, if so we will use the middle
            if startend is None:
                start = utils.middle-50
                end = utils.middle+50
            else:
                start = startend[0]
                end = startend[1]

            if utils.mtype == 'DNase':
                start = start+4
                end = end+4
            try:
                seq=utils.dataset.tokenizer.decode(a)
            except:
                seq=utils.dataset.tokenizer.decode(a[1:])
            cut_seq = seq[start:end]
            #in this function heights is required, generally just the results used for ism

            #now we define cut heights
            heights = heights_all[start:end,i] #will be a matrix of values, we take just the row of the provided celltype_idx
            heights = -heights
            logo_df = pd.DataFrame(0, index=np.arange(len(cut_seq)), columns=list(set(cut_seq)), dtype=float)
            # print(heights)

            # Fill the DataFrame with heights, converting heights to float if necessary
            
            for k, symbol in enumerate(cut_seq):
                logo_df.loc[k, symbol] = heights[k]

            # Generate the sequence logo
            logo = logomaker.Logo(logo_df, color_scheme='classic', flip_below = True, ax = ax[j, util_counter])
            #now set the title
            try:
                ax[j,util_counter].set_title(f'{celltypes[i]}, true={true_values[i]}') #if you provide true values then it works
            except:
                ax[j,util_counter].set_title(f'{celltypes[i]}')
            ax[j,util_counter].set_xticks(np.arange(0,end-start,int((end-start)/5)))
            ax[j,util_counter].set_xticklabels(np.arange(start,end,int((end-start)/5)))
            ax[j,util_counter].set_ylim(global_max,global_min)
    plt.tight_layout()
    return fig, ax