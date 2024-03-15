#we will test multiprocessing
#we see that it seems to be the same, i'm not sure if it's basically doing it sequentially, let's test it!
#let's try running from scratch, seems passing global variables is actually somehow slower, but not a huge memory usage?
from shap_utils import ShapUtils
import shap
import numpy as np

ckpt_path = '/data/leslie/sarthak/hyena/hyena-dna/outputs/2024-02-09/17-38-16-568113/checkpoints/last.ckpt'
util = ShapUtils('DNase', ckpt_path, percentage_background = 1/500000) #can take some values as global variables
util.load_from_indices()
explainer = shap.DeepExplainer(util.model, util.background_embed)
#first list the number of cores again
# Let's try to use multiprocessing on 2 cores
import os

# Example for SLURM
num_cores = int(os.environ.get('LSB_DJOB_NUMPROC', 1))  # Default to 1 if not set
print(f'Number of CPU cores available: {num_cores}') #that slurm cpus per task isn't set, changed it to lsf
from tqdm import tqdm
def process_index(i):
    templist = []
    for j in tqdm(range(15)):
        a,_ = util.dataset[i*161+j]
        a_embed = util.backbone.backbone.embeddings.word_embeddings(a.unsqueeze(0))
        shap_values = explainer.shap_values(a_embed)
        templist.append(shap_values)
        # Assuming shap_values is a numpy array or can be converted into one
    return np.concatenate(templist, axis=0) #oops, this should be templist, have to modify it since was shap_values
    # a, _ = util.dataset[i]
    # a_embed = util.backbone.backbone.embeddings.word_embeddings(a.unsqueeze(0))
    # shap_values = explainer.shap_values(a_embed)
    # # Assuming shap_values is a numpy array or can be converted into one
    # return output_all

from concurrent.futures import ProcessPoolExecutor

indices_to_process = [14, 15, 16, 17]  # The indices you want to process

# Assuming 'util' and 'explainer' are defined and ready to use

import time

start = time.time()

# Parallel execution
with ProcessPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_index, i) for i in indices_to_process]
    results = [future.result() for future in futures]

# Assuming you want to concatenate the results from both processes into one array
final_output = np.concatenate(results, axis=0)
print(final_output.shape)

end = time.time()
print(f'Time taken: {end - start:.2f}s')

#and do it sequentially
start = time.time()
#and let's do it with a single core
out_list = []
for i in indices_to_process:
    templist = []
    for j in tqdm(range(15)):
        a,_ = util.dataset[i*161+j]
        a_embed = util.backbone.backbone.embeddings.word_embeddings(a.unsqueeze(0))
        shap_values = explainer.shap_values(a_embed)
        templist.append(shap_values)
    out_list.append(np.concatenate(templist,axis=0))
#actually half the time per iteration... might take the same amoutn of time... and it did, 2 seconds faster even
end = time.time()
print(f'Time taken: {end - start:.2f}s')