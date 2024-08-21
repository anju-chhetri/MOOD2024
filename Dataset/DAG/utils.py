import json
import os
import nibabel as nib
from PIL import Image
from torchvision import transforms as T

import numpy as np
import json

from argparse import Namespace
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.augmentations.utils import pad_nd_image
from batchgenerators.transforms.abstract_transforms import AbstractTransform

def load_datalist(file_path, key='training', base_dir=None, meta_keys=[]):
    with open(file_path, 'r') as f:
        datalist = json.load(f)[key]
    if base_dir is None:
        base_dir = os.path.dirname(file_path)
    datalist = [{k:os.path.join(base_dir, v) if k not in meta_keys else v for k,v in i.items()} for i in datalist]

    return datalist


class ScaleIntensityQuantile:
    def __init__(self, min_p, max_p):
        self.min_p = min_p
        self.max_p = max_p

    def __call__(self, **data_dict):
        for k,v in data_dict.items():
            v = np.clip(v, 0, None) # Remove -ve values
            minmax_q = np.quantile( v.flatten(), q= np.array((self.min_p,self.max_p)))
            v -= minmax_q[0]
            v /= (minmax_q[1] - minmax_q[0])
            v = np.clip(v, 0, 1)
            data_dict[k] = v
        return data_dict


class CropForeground(AbstractTransform):
    def __init__(self, key_input = "data", keys_to_apply = None):
        """
        Crop the foreground of an image
        :param keys_inputs:
        :param keys_to_apply:
        """
        self.key_input = key_input
        self.keys_to_apply = [key_input] if keys_to_apply is None else (keys_to_apply if isinstance(keys_to_apply, (list, tuple)) else [keys_to_apply])

    def __call__(self, **data_dict):

        outputs = {k:[] for k in self.keys_to_apply}
        for i, (data,metadata) in enumerate(zip(data_dict[self.key_input],data_dict["metadata"])):

            nonzero = np.stack(np.abs(data).sum(0).nonzero(),-1) # Reduce channel dimension and get coords not zero

            if nonzero.shape[0] != 0:
                nonzero = np.stack([np.min(nonzero, 0), np.max(nonzero, 0)],-1)
                # nonzero now has shape 3, 2. It contains the (min, max) coordinate of nonzero voxels for each axis
                for key in self.keys_to_apply:
                    if key in data_dict:
                        seg = data_dict[key][i]
                        if seg is not None:
                            # now crop to nonzero
                            seg = seg[:,
                                       nonzero[0, 0]: nonzero[0, 1] + 1,
                                       nonzero[1, 0]: nonzero[1, 1] + 1,
                                       ]
                            if nonzero.shape[0] == 3:
                                seg = seg[:,:,:, nonzero[2, 0]: nonzero[2, 1] + 1]

                            outputs[key].append(seg)

                metadata["nonzero_region"] = nonzero
            else:
                for key in self.keys_to_apply:
                    outputs[key].append(data_dict[key][i])

        # Note that the output of this is a list, instead of single numpy array because each cna have different sizes
        # Hope for anything that comes after to iterate over the list and does not expect a np.array
        for key in self.keys_to_apply:
            data_dict[key] = outputs[key]

        return data_dict

def default(config:Namespace, key, default):
    if hasattr(config,key):
        return getattr(config,key)
    return default



class DataLoader3D(DataLoader):
    def __init__(self, datalist_json, datalist_key,  keys_image, batch_size, num_threads_in_multithreaded=0,
                 keys_label = None, datalist_base_dir=None, seed_for_shuffle=1234,
                 return_incomplete=True, shuffle=True, infinite=True, intensity_percentile_scaling = None,
                 meta_keys= []):
        """
        data must be a list of patients as returned by get_list_of_patients (and split by get_split_deterministic)

        patch_size is the spatial size the retured batch will have

        """
        meta_keys = meta_keys if meta_keys is not None else []

        self.datalist = load_datalist(datalist_json,datalist_key,datalist_base_dir, meta_keys)
        super().__init__(self.datalist, batch_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete, shuffle,
                         infinite)

        self.keys_image = keys_image if isinstance(keys_image,(tuple,dict) ) else [keys_image]
        if keys_label:
            self.keys_label = keys_label if isinstance(keys_label, (tuple,dict)) else [keys_label]
        else:
            self.keys_label = None
        self.meta_keys = meta_keys
        self.indices = list(range(len(self.datalist)))

        if intensity_percentile_scaling is not None:
            self.intensity_scaling = ScaleIntensityQuantile(*intensity_percentile_scaling)
        else:
            self.intensity_scaling = None
    @staticmethod
    def load_file(file):
        data = nib.load(file)
        data_array = np.asanyarray(data.dataobj, order="C").astype(np.float32)
        from monai.transforms import Resize
        resize = Resize((128,128,128), mode = "bilinear")
        data_array = resize(np.expand_dims(data_array, 0)).squeeze(dim=0)
        # data_array = data.get_fdata()
        return data_array, {"affine": data.affine,"filename":file}

    def generate_train_batch(self):
        # DataLoader has its own methods for selecting what patients to use next, see its Documentation
        idx = self.get_indices()
        patients_for_batch = [self._data[i] for i in idx]

        # initialize empty array for image and labels
        images = []
        if self.keys_label:
            labels = []

        metadata = []
        patient_names = []

        # iterate over patients_for_batch and include them in the batch
        for i, subj in enumerate(patients_for_batch):
            patient_data = []
            for j, k in enumerate(self.keys_image):
                if k in subj:
                    img, img_mtd = self.load_file(subj[k])
                    if self.intensity_scaling is not None:
                        img = self.intensity_scaling(data=img)['data']
                    patient_data.append(img)
                    if j == 0:
                        for k in self.meta_keys:
                            img_mtd[k] = subj[k]

                        metadata.append(img_mtd)
                        patient_names.append(subj[k])
            patient_data = np.stack(patient_data,axis=0)
            # this will only pad patient_data if its shape is smaller than self.patch_size
            if self.keys_label:
                patient_label = []
                for j, k in enumerate(self.keys_label):
                    if k in subj:
                        seg, _ = self.load_file(subj[k])
                        patient_label.append(seg)
                patient_label = np.stack(patient_label, axis=0)
                labels.append(patient_label)

        if self.keys_label:
            return {'data': patient_data, 'seg': labels, 'metadata': metadata, 'names': patient_names}
        else:
            return {'data': patient_data, 'metadata': metadata, 'names': patient_names}
