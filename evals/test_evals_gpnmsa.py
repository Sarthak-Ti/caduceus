#simply need to run this script and then print out the mean

#had issue with gpn msa and rc aug, so hopefully it's fixed now?

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import evals_utils_enformer as e
output_dir = 'outputs/2025-02-05/12-33-11-311701'
ckpt_path = f'/data1/lesliec/sarthak/caduceus/{output_dir}/checkpoints/last.ckpt'
evals = e.Evals(ckpt_path, dataset_class='GPNMSA')
zarr_name = '/data1/lesliec/sarthak/data/borzoi/model_outputs/gpnmsa_basic_ohe_2.zarr'
evals.evaluate_zarr(zarr_name)
print('done evaluating')
evals.correlate(zarr_name, 0)
print('done correlating')

output_dir = 'outputs/2025-02-05/12-33-11-305656'
ckpt_path = f'/data1/lesliec/sarthak/caduceus/{output_dir}/checkpoints/last.ckpt'
evals = e.Evals(ckpt_path, dataset_class='GPNMSA')
zarr_name = '/data1/lesliec/sarthak/data/borzoi/model_outputs/gpnmsa_basic_cnn_2.zarr'
evals.evaluate_zarr(zarr_name)
print('done evaluating')
evals.correlate(zarr_name, 0)
print('done correlating')