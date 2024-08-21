import argparse  
import shutil
import os
import numpy as np
from monai.transforms import Resize
import nibabel as nib

def main():
    parser = argparse.ArgumentParser(description="Process some folders.")
    parser.add_argument("-i", "--input", required = True, type = str)
    parser.add_argument("-m", "--mode", required = True, type = str)
    parser.add_argument("-mo", "--modality", required = True, type = str)

    args = parser.parse_args()
    input_dir = args.input
    mode = args.mode
    modality = args.modality

    k=500
    pixel_data = os.listdir(input_dir)

    if mode == "sample":
        
        for sample in pixel_data:
            if sample.endswith(".npz"):
                with np.load(os.path.join(input_dir, sample), allow_pickle=True) as pred:
                    image = pred["probabilities"][1]   #Gives 
                    image_flatten = image.flatten()
                    indices = np.argsort(image_flatten)[-k:]
                    top_k_values = image_flatten[indices]
                    score = np.mean(top_k_values)   
        
                with open(os.path.join(input_dir, sample.split('.')[0]+'.nii.gz.txt'), 'w') as file:
                    file.write(str(score))

    if mode=="pixel":
        if modality=="abdom":
            new_shape = (512,512,512)
            resize = Resize(spatial_size=new_shape, mode = "nearest")

            for sample in pixel_data:
                if sample.endswith(".nii.gz"):
                    file = nib.load(os.path.join(input_dir, sample))
                    file_data = file.get_fdata()
                    affine = file.affine
                    file_data = np.expand_dims(file_data, axis=0)
                    resized_file_data = resize(file_data)
                    nifti_file = nib.Nifti1Image(resized_file_data.squeeze(axis=0).astype(np.float32), affine = affine)
                    nib.save(nifti_file, os.path.join(input_dir, sample ))

if __name__=="__main__":
    main()