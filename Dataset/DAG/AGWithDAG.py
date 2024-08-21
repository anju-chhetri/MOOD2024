import sys
sys.path.append("/users/achhetri/challenge/utils/DAG")
from fpi import FPI
from bias import BiasCorruption
from mask_generation import GetRandomLocation, CreateRandomShape
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor
# from utils import DataLoader3D, DataLoader2Dfrom3D, DataLoader2D
import utils 

from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter

alphas_keys = []
alphas_keys.append("alpha_texture")
alphas_keys.append("alpha_bias")

    
# def load_dataset_config(file_path, json_file_to_load_image, images_path, shape_dataset_directory,):

#     import json
#     config = json.load(open(file_path, 'r'))

#     config['dataset_type'] ="training"
#     config["dataset_name"] = 

#     # Load default values from environment_defaults
#     from_environment = [(json_file_to_load_image,True),(images_path,False)]

#     if config['dataset_type'] == "training":
#         from_environment.extend([('shape_dataset',True), ('shape_base_dir',False)])

#     for k,required in from_environment:
#         if k not in config:
#             if k in os.environ:
#                 config[k] = os.environ[k]

#         assert (not required) or (k in config) or (k in os.environ),f"Missing setting {k}, not found in config or environment variables"

#     # Transform config dict to Namespace
#     config = Namespace(**config)

#     return config



def get_ag_transforms(type='dag', anom_patch_size=[64,64,64], no_mask_in_background=False, shape_dataset=None,
                      p_anomaly=1.0,
                      quantized_bias_codebook=[0.00565079, 0.38283867, 0.61224216, 0.76920754, 0.9385473],
                      randomshape_kwargs={}, fpi_kwargs={}, bias_kwargs={}):

    if shape_dataset is not None and "randommask_kwargs" not in randomshape_kwargs:
        randomshape_kwargs["randommask_kwargs"] = {"dataset": shape_dataset, "spatial_prob": 1.0, "scale_masks": (0.5, 0.75)}

    ag_transforms = [
        GetRandomLocation(anom_patch_size=anom_patch_size),
        CreateRandomShape(anom_patch_size=anom_patch_size, smooth_prob=1.0,
                           no_mask_in_background=no_mask_in_background, **randomshape_kwargs),
        ]

    # Override with None if not needed
    quantized_bias_codebook = quantized_bias_codebook if type in ['dag','bias_only'] else None

    # if type == 'fpi':
    #     ag_transforms.append(
    #         FPI(image_key='data', anomaly_interpolation='linear', output_key="data_c", p_anomaly=p_anomaly,
    #             anom_patch_size=anom_patch_size, normalize_fp=False, **fpi_kwargs))


    if type in ['dag','dag_no_quant']:
        ag_transforms.extend([
            FPI(image_key='data', anomaly_interpolation='linear', output_key="data_c", p_anomaly=p_anomaly,
                anom_patch_size=anom_patch_size, normalize_fp="minmax", **fpi_kwargs),
            BiasCorruption(image_key="data_c", shape_key="shape", output_key="data_c", p_anomaly=p_anomaly,
                           quantized_bias_codebook=quantized_bias_codebook, quantized_mask_key="shape_bias", **bias_kwargs)
        ])

    # elif type == 'bias_only':
    #     ag_transforms.extend([
    #         BiasCorruption(image_key="data", shape_key="shape", output_key="data_c", p_anomaly=p_anomaly,
    #                        quantized_bias_codebook=quantized_bias_codebook, quantized_mask_key="shape_bias", **bias_kwargs)
    #     ])

    return ag_transforms


shape_dataset = "/users/achhetri/challenge/disyre/experiments/shapes/dataset.json"
shape_base_dir = "/users/achhetri/challenge/disyre/experiments/shapes"
p_anomaly = 1.0
quantized_bias_codebook = [0.00565079, 0.38283867, 0.61224216, 0.76920754, 0.9385473]
# randomshape_kwargs = {}
# fpi_kwargs = {}
# bias_kwargs = {}

config = {}
config['dataset_type'] = 'training'
config['anom_type'] = 'dag'
config['datalist_key'] = 'training'
config['anom_patch_size'] = [64,64,64]
config['num_workers'] = 24
config['batch_size'] = 1
config['patch_size'] = [128,128]
config['no_anom_in_background'] = True
config['num_channels'] = 1
config['output_mode'] = "3D"
config["shape_dataset"] = shape_dataset
config['shape_base_dir'] =  shape_base_dir

from monai.transforms import Resized
from mask_generation import resize_operation
# ag_transform = [Resized(keys = 'data', spatial_size=config["patch_size"], mode="bilinear")]
# resize = resize_operation(config['patch_size'])
# ag_transform  = [resize]
ag_transform = get_ag_transforms(
                            config["anom_type"],
                            config["anom_patch_size"],
                            config["no_anom_in_background"],
                            config["shape_dataset"],
                            p_anomaly,
                            quantized_bias_codebook)

keys_to_torch = ['data', 'data_c', 'alpha_texture', 'alpha_bias' ]
ag_transform.append(NumpyToTensor(keys=keys_to_torch,cast_to='float'))

ag_transform = Compose(ag_transform)
datalist_keys = ["training"]
import os
num_workers = os.cpu_count()
num_workers = min(num_workers,24)
dataset_config="/users/achhetri/challenge/dataset.json" # Provide path to config file which has name of images and location: Make sure everything is under json file
dataset_path="/scratch/achhetri/data/brain/seg_brain_train/brain_train"
batch_size=1
for i, datalist_key in enumerate(datalist_keys):

    dataloader_train = utils.DataLoader3D(dataset_config, datalist_key, "image",
                                datalist_base_dir=dataset_path,
                                batch_size=batch_size,
                                num_threads_in_multithreaded=num_workers,
                                intensity_percentile_scaling = [0, .98],
                                keys_label=None,
                                infinite= True,
                                meta_keys=None,
                                )
tr_gen = MultiThreadedAugmenter(dataloader_train, ag_transform,
                                num_processes=num_workers,
                                num_cached_per_queue=1,
                                seeds=None,
                                pin_memory=True)
batch = next(tr_gen)


