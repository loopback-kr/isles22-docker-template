# Python Built-in libraries
import os, sys, importlib, json, csv, re, random, numpy as np, SimpleITK
from os.path import join, basename, exists, splitext, dirname, isdir
from pathlib import Path
from shutil import copy, copytree, rmtree
from glob import glob, iglob
from tqdm import tqdm, trange
from tqdm.contrib import tzip
# Medical imaging libraries
import nibabel as nib, SimpleITK as sitk
# nnUNet
sys.path.append('sources')
from sources.nnunet.inference.predict import predict_from_folder
from prediction_combinations import *


MODEL_DIRS = {
    0: 'models/Task511_ISLES_d_m_div2_dwionly/nnUNetTrainerV2__nnUNetPlans_pretrained_nnUNetData_plans_v2.1',
    1: 'models/Task762_ISLES22_prep_ADCDWI-FINETUNE-BraTS2021_T1T1_Necrotic(Multitask_sumloss)/nnUNetTrainerV2__nnUNetPlans_pretrained_nnUNetData_plans_v2.1',
    2: 'models/Task763_ISLES22_prep_ADCDWI-FINETUNE-BraTS2021_T1CET1CE_Necrotic(Multitask_sumloss)/nnUNetTrainerV2__nnUNetPlans_pretrained_nnUNetData_plans_v2.1',
    3: 'models/Task764_ISLES22_prep_ADCDWI-FINETUNE-BraTS2021_T2T2_Necrotic(Multitask_sumloss)/nnUNetTrainerV2__nnUNetPlans_pretrained_nnUNetData_plans_v2.1',
    4: 'models/Task765_ISLES22_prep_ADCDWI-FINETUNE-BraTS2021_FLAIRFLAIR_Necrotic(Multitask_sumloss)/nnUNetTrainerV2__nnUNetPlans_pretrained_nnUNetData_plans_v2.1',
}
MODALITIES_PER_MODEL = [
    {0: 'DWI',},
    {0: 'ADC', 1: 'DWI',},
    {0: 'ADC', 1: 'DWI',},
    {0: 'ADC', 1: 'DWI',},
    {0: 'ADC', 1: 'DWI',},
]
MODEL_COMBINATION = {ensemble_max: [{ensemble_mean: [1, 2, 3, 4]}, 0]} # TODO: ensemble(ensemble(0, 1), ensemble(2, 3)) 이런식이면 덮여씌워져서 안됨. 폴더 다르게 해야함. 같은 nested layer단에서 같은 combination을 하면 안됨!(같은 저장 폴더를 가진채로)


# nnUNet_CHKPOINT='model_best'
nnUNet_CHKPOINT='model_final_checkpoint'
DEFAULT_INPUT_PATH = Path("/input")
DEFAULT_ALGORITHM_OUTPUT_IMAGES_PATH = Path("/output/images/")
DEFAULT_ALGORITHM_OUTPUT_FILE_PATH = Path("/output/results.json")
MODALITIES_PER_MODEL = { i : modals for i, modals in enumerate(MODALITIES_PER_MODEL)}
NUM_PROCESSES=min(3, os.cpu_count())
nnUNet_INPUT_DIR = 'tmp/imagesTs'
nnUNet_OUTPUT_DIR = 'tmp/outputTs'


def nnunet_inference(model_idx: int):
    model_dir = MODEL_DIRS[model_idx]
    model_name = model_dir.split('/')[1]
    os.makedirs(str(Path(nnUNet_OUTPUT_DIR, model_name)), exist_ok=True)
    copy(Path(model_dir, 'fold_0', 'postprocessing.json'), Path(model_dir, 'postprocessing.json'))
    predict_from_folder(
        model=model_dir,
        input_folder=str(Path(nnUNet_INPUT_DIR, model_name)),
        output_folder=str(Path(nnUNet_OUTPUT_DIR, model_name)),
        folds=None,
        save_npz=True,
        num_threads_preprocessing=NUM_PROCESSES,
        num_threads_nifti_save=NUM_PROCESSES,
        lowres_segmentations=None,
        part_id=0,
        num_parts=1,
        tta=True,
        overwrite_existing=True,
        mode="normal",
        overwrite_all_in_gpu=None,
        mixed_precision=True,
        step_size=0.5,
        checkpoint_name=nnUNet_CHKPOINT,
    )
    return join(nnUNet_OUTPUT_DIR, model_name)

# todo change with your team-name
class POBOTRI():
    def __init__(self,
                 input_path: Path = DEFAULT_INPUT_PATH,
                 output_path: Path = DEFAULT_ALGORITHM_OUTPUT_IMAGES_PATH):

        self.debug = False  # False for running the docker!
        if self.debug:
            self._input_path = Path('/path-do-input-data/')
            self._output_path = Path('/path-to-output-dir/')
            self._algorithm_output_path = self._output_path / 'stroke-lesion-segmentation'
            self._output_file = self._output_path / 'results.json'
            self._case_results = []

        else:
            self._input_path = input_path
            self._output_path = output_path
            self._algorithm_output_path = self._output_path / 'stroke-lesion-segmentation'
            self._output_file = DEFAULT_ALGORITHM_OUTPUT_FILE_PATH
            self._case_results = []

    def predict(self, input_data):
        """
        Input   input_data, dict.
                The dictionary contains 3 images and 3 json files.
                keys:  'dwi_image' , 'adc_image', 'flair_image', 'dwi_json', 'adc_json', 'flair_json'

        Output  prediction, array.
                Binary mask encoding the lesion segmentation (0 background, 1 foreground).
        """
        # Get all image inputs.
        dwi_image, adc_image, flair_image = input_data['dwi_image'],\
                                            input_data['adc_image'],\
                                            input_data['flair_image']


        # Get all json inputs.
        dwi_json, adc_json, flair_json = input_data['dwi_json'],\
                                         input_data['adc_json'],\
                                         input_data['flair_json']

        ################################################################################################################
        #################################### Beginning of your prediction method. ######################################
        # todo replace with your best model here!
        
        # Convert *.mha to *.nii.gz
        for (_, modals), (_, model_dir) in zip(MODALITIES_PER_MODEL.items(), MODEL_DIRS.items()):
            model_name = model_dir.split('/')[1]
            for modal_idx, modal in modals.items():
                os.makedirs(Path(nnUNet_INPUT_DIR, model_name), exist_ok=True)
                sitk.WriteImage(input_data[f'{modal.lower()}_image'], str(Path(nnUNet_INPUT_DIR, model_name, f'sample_{modal_idx:04d}.nii.gz')))

        # Model inference
        for model, params in MODEL_COMBINATION.items():
            output_dir = model(nnunet_inference, params)
        prediction = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(f'{output_dir}/sample.nii.gz')))

        #################################### End of your prediction method. ############################################
        ################################################################################################################

        return prediction.astype(int)

    def process_isles_case(self, input_data, input_filename):
        # Get origin, spacing and direction from the DWI image.
        origin, spacing, direction = input_data['dwi_image'].GetOrigin(),\
                                     input_data['dwi_image'].GetSpacing(),\
                                     input_data['dwi_image'].GetDirection()

        # Segment images.
        prediction = self.predict(input_data) # function you need to update!

        # Build the itk object.
        output_image = SimpleITK.GetImageFromArray(prediction)
        output_image.SetOrigin(origin), output_image.SetSpacing(spacing), output_image.SetDirection(direction)

        # Write segmentation to output location.
        if not self._algorithm_output_path.exists():
            os.makedirs(str(self._algorithm_output_path))
        output_image_path = self._algorithm_output_path / input_filename
        SimpleITK.WriteImage(output_image, str(output_image_path))

        # Write segmentation file to json.
        if output_image_path.exists():
            json_result = {"outputs": [dict(type="Image", slug="stroke-lesion-segmentation",
                                                 filename=str(output_image_path.name))],
                           "inputs": [dict(type="Image", slug="dwi-brain-mri",
                                           filename=input_filename)]}

            self._case_results.append(json_result)
            self.save()


    def load_isles_case(self):
        """ Loads the 6 inputs of ISLES22 (3 MR images, 3 metadata json files accompanying each MR modality).
        Note: Cases missing the metadata will still have a json file, though their fields will be empty. """

        # Get MR data paths.
        dwi_image_path = self.get_file_path(slug='dwi-brain-mri', filetype='image')
        adc_image_path = self.get_file_path(slug='adc-brain-mri', filetype='image')
        flair_image_path = self.get_file_path(slug='flair-brain-mri', filetype='image')

        # Get MR metadata paths.
        dwi_json_path = self.get_file_path(slug='dwi-mri-acquisition-parameters', filetype='json')
        adc_json_path = self.get_file_path(slug='adc-mri-parameters', filetype='json')
        flair_json_path = self.get_file_path(slug='flair-mri-acquisition-parameters', filetype='json')

        input_data = {'dwi_image': SimpleITK.ReadImage(str(dwi_image_path)), 'dwi_json': json.load(open(dwi_json_path)),
                      'adc_image': SimpleITK.ReadImage(str(adc_image_path)), 'adc_json': json.load(open(adc_json_path)),
                      'flair_image': SimpleITK.ReadImage(str(flair_image_path)), 'flair_json': json.load(open(flair_json_path))}

        # Set input information.
        input_filename = str(dwi_image_path).split('/')[-1]
        return input_data, input_filename

    def get_file_path(self, slug, filetype='image'):
        """ Gets the path for each MR image/json file."""

        if filetype == 'image':
            file_list = list((self._input_path / "images" / slug).glob("*.mha"))
        elif filetype == 'json':
            file_list = list(self._input_path.glob("*{}.json".format(slug)))

        # Check that there is a single file to load.
        if len(file_list) != 1:
            print('Loading error')
        else:
            return file_list[0]

    def save(self):
        with open(str(self._output_file), "w") as f:
            json.dump(self._case_results, f)

    def process(self):
        input_data, input_filename = self.load_isles_case()
        self.process_isles_case(input_data, input_filename)


if __name__ == "__main__":
    # todo change with your team-name
    POBOTRI().process()
