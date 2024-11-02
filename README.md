# MOOD_2024

This repository contains the official implementation from team BBMLL for the **Medical Out-of-Distribution Analysis Challenge 2024**.

## Abstract
The effectiveness of supervised learning algorithms in medical imaging is often limited by the difficulty in obtaining labeled datasets–especially for rare pathologies–and the resulting long-tailed distribution of classes. This poses a challenge when diagnosing unseen conditions, as the model needs to be retrained to detect them. Out-of-distribution (OOD) detection methods offer a promising alternative for critical tasks in computer-aided diagnosis, such as medical image segmentation and classification, by modeling healthy data—which are typically easier to collect—as in-distribution, and pathological or other conditions as out-of-distribution. Two common OOD detection approaches can be categorized into reconstruction error-based methods and self-supervised learning. While reconstruction-based OOD detection techniques employing autoencoders have been applied to segmentation tasks, they often require costly hyperparameter tuning and struggle to detect and localize subtle abnormalities.

Self-supervised learning with synthetic anomalies has gained increasing attention due to the relatively constrained data distribution in medical images compared to natural domain images, as well as the availability of domain-specific knowledge to emulate realistic anomalies. In this work, we combine two anomaly generation processes: eForeign Path Interpolation (FPI), which creates iso-intense abnormalities, and Disentangled Anomaly Generation (DAG), which is particularly effective in generating hypo- and hyper-intense abnormalities.  This combination generates a comprehensive dataset that enhances coverage of real-world anomalies. Our method employs a 3D U-Net architecture featuring a residual encoder with deep supervision and patch-based training to provide pixel-level abnormality scores. During inference, we use sliding window inference and model ensembling. We validate this approach across two volumetric data modalities: CT and MRI.
## Dataset
Our approach utilizes two datasets provided by the challenge organizers, comprising MRI and CT volumetric images from healthy patients. We employ a self-supervised method for anomaly detection by generating a synthetic dataset.

In this work, we combine two anomaly generation processes: [Foreign Path Interpolation (FPI)](https://github.com/jemtan/FPI), which creates iso-intense abnormalities, and [Disentangled Anomaly Generation (DAG)](https://github.com/snavalm/disyre), which is particularly effective in generating hypo- and hyper-intense abnormalities. This combination generates a comprehensive dataset that enhances coverage of real-world anomalies.

We follow it up with qualitative analysis to ensure the generated dataset closely replicates real-world anomalies.

## Network

We implement [nnU-Net]((https://github.com/MIC-DKFZ/nnUNet)), a fully automated framework that optimizes parameters by analyzing the data fingerprint.

### Training Details:
  1. Architecture: 3D U-Net with a residual encoder and deep supervision.
   2. Loss Function: Dice loss and Cross Entropy.
   3. Training Method: Patch-based training (due to computational constraints on 3D data).


### Inference details:
1. Sliding window inference
2. Model ensembling

## Instructions
The [Dataset](https://github.com/anju-chhetri/MOOD2024/tree/master/Dataset) folder contains the script for anomaly generation.

After generating the dataset, structure it as specified [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md).

 ### Training
 1. Place `dataset_CT.json` and `dataset_MRI.json` in their respective folders and rename them to dataset.json.
 2. Run the following command to plan and preprocess:
    ```nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity -pl nnUNetPlannerResEncM```
 4. Move the `splits_final_CT.json` and `splits_final_MRI.json` files to preprocessed/DatasetXX and rename them to splits_final.json.
 5. Train the model using:
    ```nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD -p nnUNetResEncUNetMPlans --npz```
    

### Inference
1. Run the following command to perform inference:
    ```nnUNetv2_predict -d Dataset018_CM -i INPUT_FOLDER -o OUTPUT_FOLDER -f 0 1 2 3 4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetResEncUNetMPlans```


## License

This project is licensed under the MIT License - see the [LICENSE](https://pitt.libguides.com/openlicensing/MIT) file for details.
