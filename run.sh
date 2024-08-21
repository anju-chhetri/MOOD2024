export nnUNet_raw="/scratch/achhetri/data/nnUnetD4/nnUNet_raw"
export nnUNet_preprocessed="/scratch/achhetri/data/nnUnetD4/preprocessed"
export nnUNet_results="/scratch/achhetri/experimentalResults/nnUNetD4"
nnUNetv2_train 17 3d_fullres 0 -p nnUNetResEncUNetMPlans --npz
