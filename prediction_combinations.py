import os, sys, numpy as np, SimpleITK as sitk
from os.path import join
NUM_PROCESSES=min(3, os.cpu_count())
# nnUNet
sys.path.append('sources')
from sources.nnunet.inference.ensemble_predictions import merge
nnUNet_OUTPUT_DIR = 'tmp/outputTs'


def single(predict, model_index):
    return predict(model_index)

def ensemble_max(predict, model_index, output_dirname='max_ensembled'): # UNION
    final_output_dir = join(nnUNet_OUTPUT_DIR, output_dirname)
    os.makedirs(final_output_dir, exist_ok=True)
    recursive_output_dirs = []

    # Recursive prediction
    for model_idx in model_index:
        if type(model_idx) == dict:
            for k, v in model_idx.items():
                recursive_output_dirs.append(k(predict, v))
        else:
            recursive_output_dirs.append(predict(model_idx))
    
    # Ensemble
    assert len(recursive_output_dirs) == 2, "Only support two model outputs"
    prediction_1 = sitk.GetArrayFromImage(sitk.ReadImage(f'{recursive_output_dirs[0]}/sample.nii.gz'))
    prediction_2 = sitk.GetArrayFromImage(sitk.ReadImage(f'{recursive_output_dirs[1]}/sample.nii.gz'))
    union = np.squeeze(np.squeeze(np.logical_or(prediction_1, prediction_2).astype(np.int8))) # Union

    # Create new image for return
    union_raw = sitk.GetImageFromArray(union)
    union_raw.CopyInformation(sitk.ReadImage(f'{recursive_output_dirs[0]}/sample.nii.gz'))
    
    sitk.WriteImage(union_raw, join(final_output_dir, 'sample.nii.gz'))
    return final_output_dir

def ensemble_mean(predict, model_index, output_dirname='mean_ensembled'):
    final_output_dir = join(nnUNet_OUTPUT_DIR, output_dirname)
    os.makedirs(final_output_dir, exist_ok=True)
    recursive_output_dirs = []
    
    # Recursive prediction
    for model_idx in model_index:
        if type(model_idx) == dict:
            for k, v in model_idx.items():
                recursive_output_dirs.append(k(predict, v))
        else:
            recursive_output_dirs.append(predict(model_idx))
    
    merge(recursive_output_dirs, final_output_dir, NUM_PROCESSES, override=True, postprocessing_file=None, store_npz=True)
    return final_output_dir