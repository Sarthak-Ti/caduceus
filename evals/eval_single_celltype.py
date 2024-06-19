#details can be seen in evaluation 7

import torch
from evals_utils import Evals


predictions_reg = torch.zeros([105252, 161])
predictions_class = torch.zeros([105252, 161])
targets_reg = torch.zeros([105252, 161])
targets_class = torch.zeros([105252, 161])

import yaml

def find_key_recursively(data, target_key):
    if isinstance(data, dict):
        for key, value in data.items():
            if key == target_key:
                return value
            found = find_key_recursively(value, target_key)
            if found is not None:
                return found
    elif isinstance(data, list):
        for item in data:
            found = find_key_recursively(item, target_key)
            if found is not None:
                return found
    return None

def get_single_cell_type(yaml_file, target_key='single_cell_type'):
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
    return find_key_recursively(data, target_key)

path = '/data/leslie/sarthak/hyena/hyena-dna/outputs'
import os
items = sorted(os.listdir(path))
main_dirs = [item for item in items if os.path.isdir(os.path.join(path, item))]
main_dirs = main_dirs[-10:-1]

cell_type_list = []

for date_dir in main_dirs:
    subdirs = sorted(os.listdir(os.path.join(path, date_dir)))
    for subdir in subdirs:
        #now we first check to see if there is a checkpoitns folder
        if not os.path.isdir(os.path.join(path, date_dir, subdir, 'checkpoints')):
            continue
        #now we check if there are 26 checkpoints
        checkpoints = sorted(os.listdir(os.path.join(path, date_dir, subdir, 'checkpoints')))
        if len(checkpoints) != 26:
            continue
        #now this means we are good, so let's extract the single cell type
        yaml_file = os.path.join(path, date_dir, subdir, '.hydra', 'config.yaml')
        single_cell_type_value = get_single_cell_type(yaml_file)
        model_path = os.path.join(path, date_dir, subdir, 'checkpoints', checkpoints[-1])
        eval_ctst = Evals('DNase_ctst',model_path, classification=True, single_cell_type=single_cell_type_value)
        targets,predicts = eval_ctst.evaluate(num_workers = 1, batch_size = 512)
        #now separate out into regression and classification
        predictions_reg[:,single_cell_type_value] = predicts[1].squeeze()
        predictions_class[:,single_cell_type_value] = predicts[0].squeeze()
        targets_reg[:,single_cell_type_value] = targets[1].squeeze()
        targets_class[:,single_cell_type_value] = targets[0].squeeze()

#now we save it out in this format
#torch.save(targets_class,'/data/leslie/sarthak/hyena/hyena-dna/evals/results/multitasking_bestepoch_targets_class.pt')
torch.save(targets_class,'/data/leslie/sarthak/hyena/hyena-dna/evals/results/sct_bestepoch_targets_class.pt')
torch.save(targets_reg,'/data/leslie/sarthak/hyena/hyena-dna/evals/results/sct_bestepoch_targets_reg.pt')
torch.save(predictions_class,'/data/leslie/sarthak/hyena/hyena-dna/evals/results/sct_bestepoch_predicts_class.pt')
torch.save(predictions_reg,'/data/leslie/sarthak/hyena/hyena-dna/evals/results/sct_bestepoch_predicts_reg.pt')