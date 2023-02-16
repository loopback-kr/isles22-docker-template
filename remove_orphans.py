# Python Built-in libraries
import os, sys, json, csv, re, random
from os.path import join, basename, exists, splitext, dirname, isdir
from shutil import copy, copytree, rmtree
from glob import glob, iglob

import process

print('Remove unused model files...')

# Remove unused models in build context
buildctx_model_dirs = [path for path in sorted(os.listdir('models'))]
compiling_model_dirs = [model_dir.split('/')[1] for model_dir in process.MODEL_DIRS]

for model_dir in buildctx_model_dirs:
    if model_dir in compiling_model_dirs:
        pass
    else:
        rmtree(join('models', model_dir))

# Remove unused files in build context
for model_dir in process.MODEL_DIRS:
    for model_file_path in sorted(list(iglob(join(model_dir, '**', '*'), recursive=True))):
        if basename(model_file_path) == process.nnUNet_CHKPOINT + '.model':
            continue
        elif basename(model_file_path) == process.nnUNet_CHKPOINT + '.model.pkl':
            continue
        elif basename(model_file_path) == 'postprocessing.json':
            continue
        elif basename(model_file_path) == 'plans.pkl':
            continue
        elif isdir(model_file_path):
            continue
        else:
            os.remove(model_file_path)