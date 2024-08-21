'''
Read the config file and all the images name
images_in_train = read(config)
images in train folder = read(path)
valid_images = images in train folder - images in train
perform augmentation and keep it in a folder
add path and valid images in the config file
'''

# Anomalies

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
import os
import math
import copy
matplotlib.use('Agg')
import random
from scipy.interpolate import RegularGridInterpolator
import nibabel as nib
from tqdm import tqdm
import random


def calc_distance(xyz0 = [], xyz1 = []):
    delta_OX = (xyz0[0] - xyz1[0])**2
    delta_OY = (xyz0[1] - xyz1[1])**2
    delta_OZ = (xyz0[2] - xyz1[2])**2
    return (delta_OX+delta_OY+delta_OZ)**0.5 

#Circular deformation 
def create_mask(im,center,width):
    dims = np.shape(im)
    mask = np.zeros_like(im)
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                dist_i = calc_distance([i,j,k],center)
                if dist_i<width:
                    mask[i,j,k]=1

    return mask

def create_deformation(im,center,width,polarity=1):
    dims = np.array(np.shape(im))
    mask = np.zeros_like(im)
    
    center = np.array(center)
    xv,yv,zv = np.arange(dims[0]),np.arange(dims[1]),np.arange(dims[2])
    interp_samp = RegularGridInterpolator((xv, yv, zv), im)
    
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                dist_i = calc_distance([i,j,k],center)
                displacement_i = (dist_i/width)**2
                
                if displacement_i < 1.:

                    #within width
                    if polarity > 0:
                        #push outward
                        diff_i = np.array([i,j,k])-center
                        new_coor =  center + diff_i*displacement_i
                        new_coor = np.clip(new_coor,(0,0,0),dims-1)
                        mask[i,j,k]= interp_samp(new_coor)
                        
                    else:
                        #pull inward
                        cur_coor = np.array([i,j,k])
                        diff_i = cur_coor-center
                        new_coor = cur_coor + diff_i*(1-displacement_i)
                        new_coor = np.clip(new_coor,(0,0,0),dims-1)
                        mask[i,j,k]= interp_samp(new_coor)
                else:
                    mask[i,j,k] = im[i,j,k]
    return mask

def create_shift(ima,mask,shift):
    dims = np.array(np.shape(ima))
    im_apply_shift = copy.deepcopy(ima)
 
    pad_val = np.max(np.abs(shift))
    im_pad = np.pad(ima,pad_val,'edge')
    im_pad = np.roll(im_pad,shift,(0,1,2))
     
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                
                if mask[i,j,k] > 0:

                    im_apply_shift[i,j,k] = im_pad[i+pad_val,j+pad_val,k+pad_val]
                
    return im_apply_shift

def create_reflect(im,mask,axis):
    dims = np.array(np.shape(im))
    im_apply_reflect = copy.deepcopy(im)
 
    im_reflect = np.flip(im,axis=axis)
     
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                
                if mask[i,j,k] > 0:
                    im_apply_reflect[i,j,k] = im_reflect[i,j,k]
                
    return im_apply_reflect

#Anomalies work finish

import json
import os

def read_json_file(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)

    return data

def extract_image_paths(json_data):
    image_path_validation = [entry["image"] for entry in json_data["validation"]]
    image_path_training = [entry["image"] for entry in json_data["training"]]

    return image_path_training, image_path_validation


def append_validation_to_json(existing_json_path, images, labels, types, modes):
    # data = {
    #     modes: [],
    #     f"num{modes}": len(images)
    # }

    # for image, label, type_ in zip(images, labels, types):
    #     data[modes].append({
    #         "image": image,
    #         "label": label,
    #         "type": type_
    #     })

    # if os.path.isfile(existing_json_path):

    #     with open(existing_json_path, 'a') as f:
    #         json.dump(data, f, indent=4)
    # else:
    #     with open(existing_json_path, 'w') as f:
    #         json.dump(data, f, indent=4)
    if not os.path.isfile(existing_json_path):
        empty_dict = {}
    
    # Open the file in write mode
        with open(existing_json_path, 'w') as json_file:
        # Dump the empty dictionary to the file
            json.dump(empty_dict, json_file, indent=4)
            
    with open(existing_json_path, 'r') as f:
        data = json.load(f)
    
    # Check if the 'validation' key exists, if not, initialize it
    if f'{modes}' not in data:
        data[f'{modes}'] = []

    # Append new validation entries
    for img, lbl, typ in zip(images, labels, types):
        data[f'{modes}'].append({
            "image": img,
            "label": lbl,
            "type": typ
        })
    
    # Update the numValidation field
    data[f'num_{modes}'] = len(data[f'{modes}'])
    
    # Write the updated JSON back to the file
    with open(existing_json_path, 'w') as f:
        json.dump(data, f, indent=4)




    


def main():
    file_path = "/scratch/achhetri/data/2022Winner/abdom_train/dataset_fold_0.json"

    data = read_json_file(file_path)
    image_path = extract_image_paths(data)

    splits = ["training", "validation"]
    source_dir = "/scratch/achhetri/data/2022Winner/abdom_train"
    
    data_dir = "/scratch/achhetri/data/2022Winner/pure/synthetic_data"
    label_dir = "/scratch/achhetri/data/2022Winner/pure/synthetic_label"
    updated_path = f"{data_dir}/dataset.json"

    for split in range(len(splits)):
        images = []
        for ima in image_path[split]:
            images.append(ima.split("/")[-1])

        random.shuffle(images)

        # Generate anomalies
        label = []
        type_l = []
        data = []
        anomaly_type = ["uniform_additive_noise", "additive_noise", "deformation", "shift_volume", "reflect_volume", "healthy"]
        
        for fname in tqdm(images):
            #load original image
            im = nib.load(os.path.join(source_dir,fname))
            ima = im.get_fdata()

            #create random anomaly
            dims = np.array(np.shape(ima))
            core = dims/2#width of core region
            offset = core/2#offset to center core
            
            min_width = np.round(np.random.uniform(0.04, 0.08)*dims[0])
            max_width = np.round(np.random.uniform(0.1, 0.14)*dims[0])

            sphere_center = []
            sphere_width = []
            for i,_ in enumerate(dims):
                sphere_center.append(np.random.randint(offset[i],offset[i]+core[i]))
            sphere_width = np.random.randint(min_width,max_width)

            mask_i = create_mask(ima,sphere_center,sphere_width)

            intensity_range = np.max(ima)-np.min(ima)
            anomaly_index = np.random.randint(0, len(anomaly_type))

            if anomaly_type[anomaly_index] == "healthy":
                ima_out = ima
                mask_i = np.zeros(np.shape(ima))
                type_abnormal = "healthy"

            elif anomaly_type[anomaly_index] == "uniform_additive_noise":
                sphere_intensity = np.random.uniform(0.2*intensity_range,0.3*intensity_range,1)
                if np.random.randint(2):#random sign
                    sphere_intensity *= -1
                
                sphere_add = mask_i*sphere_intensity
                ima_out = ima+sphere_add
                type_abnormal = "uniform_additive_noise"

            elif anomaly_type[anomaly_index] == "additive_noise":
                sphere_intensity = np.random.uniform(0.05*intensity_range,0.3*intensity_range,size=np.shape(mask_i))
                if np.random.randint(2):#random sign
                    sphere_intensity *= -1

                sphere_add = mask_i*sphere_intensity
                ima_out = ima+sphere_add
                type_abnormal = "additive_noise"

            elif anomaly_type[anomaly_index] == "deformation":
                sphere_polarity = 1
                if np.random.randint(2):#random sign
                    sphere_polarity *= -1

                #apply anomaly
                ima_out = create_deformation(ima,sphere_center,sphere_width,sphere_polarity)

                type_abnormal = "deformation"


            elif anomaly_type[anomaly_index] == "shift_volume":
                sphere_shift = np.random.randint(int(0.02*dims[0]),int(0.05*dims[0]),size=3)
                for i in range(3):
                    if np.random.randint(2):#random sign
                        sphere_shift[i] *= -1

                ima_out = create_shift(ima,mask_i,sphere_shift)
                type_abnormal = "shift_volume"


            elif anomaly_type[anomaly_index] == "reflect_volume":
                axis=0
                ima_out = create_reflect(ima,mask_i,axis) 
                type_abnormal = "reflect_volume"

            #image
            ima_out_nii = nib.Nifti1Image(ima_out, affine=np.eye(4))
            nib.save(ima_out_nii, os.path.join(data_dir,fname.split('.')[0]+'_out.nii.gz'))
            #label
            sphere_add_nii = nib.Nifti1Image(mask_i, affine=np.eye(4))
            nib.save(sphere_add_nii, os.path.join(label_dir,fname.split('.')[0]+'_out.nii.gz'))

            label.append(os.path.join(label_dir,fname.split('.')[0]+'_out.nii.gz'))
            type_l.append(type_abnormal)
            data.append(os.path.join(data_dir,fname.split('.')[0]+'_out.nii.gz'))

        append_validation_to_json(updated_path , data, label, type_l, splits[split])


if __name__ == "__main__":
    main()