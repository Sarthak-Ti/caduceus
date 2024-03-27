#using the vals utils, evaluates all of them and saves them out
#final test with the normal DNase model
# import pretty_errors
import torch
from evals_utils import Evals

multitasking_path = '/data/leslie/sarthak/hyena/hyena-dna/outputs/2024-03-20/22-26-03-326937/checkpoints/138-val_loss=0.37743.ckpt'

# #let's try the ctst path
# eval_cts = Evals('DNase',cts_path, cfg = 'DNase_full.yaml')
# #now let's do evals
# targets,predicts = eval_cts.evaluate(num_workers = 4, batch_size = 4096)

# #now let's save them out
# torch.save(targets,'/data/leslie/sarthak/hyena/hyena-dna/evals/results/cts_targets.pt')
# torch.save(predicts,'/data/leslie/sarthak/hyena/hyena-dna/evals/results/cts_predicts.pt')

# del eval_cts

#now let's try the multitasking path
eval_multitasking = Evals('DNase_allcelltypes',multitasking_path, classification=True)
targets,predicts = eval_multitasking.evaluate(num_workers = 2, batch_size = 512)
targets_class,targets_reg = targets
predicts_class,predicts_reg = predicts

torch.save(targets_class,'/data/leslie/sarthak/hyena/hyena-dna/evals/results/multitasking_targets_class.pt')
torch.save(targets_reg,'/data/leslie/sarthak/hyena/hyena-dna/evals/results/multitasking_targets_reg.pt')
torch.save(predicts_class,'/data/leslie/sarthak/hyena/hyena-dna/evals/results/multitasking_predicts_class.pt')
torch.save(predicts_reg,'/data/leslie/sarthak/hyena/hyena-dna/evals/results/multitasking_predicts_reg.pt')

# del eval_multitasking

# #now let's try the cts2 path
# eval_cts2 = Evals('DNase',cts_path2)
# targets,predicts = eval_cts2.evaluate(num_workers = 4, batch_size = 4096)

# torch.save(targets,'/data/leslie/sarthak/hyena/hyena-dna/evals/results/cts2_targets.pt')
# torch.save(predicts,'/data/leslie/sarthak/hyena/hyena-dna/evals/results/cts2_predicts.pt')

# del eval_cts2

# #now let's try the ctst path
# eval_ctst = Evals('DNase_ctst',ctst_path)
# targets,predicts = eval_ctst.evaluate(num_workers = 4, batch_size = 4096)

# torch.save(targets,'/data/leslie/sarthak/hyena/hyena-dna/evals/results/ctst_targets.pt')
# torch.save(predicts,'/data/leslie/sarthak/hyena/hyena-dna/evals/results/ctst_predicts.pt')