script_path=$(realpath "$0")
script_dir=$(dirname "$script_path")
script_dir=$(dirname "$script_dir")
export nnUNet_results="${script_dir}/network"

python3 "${script_dir}/preprocess_data.py" -i $1 -o "$2/data" -m "abdom"
cp "${script_dir}/network/Dataset017_CM/nnUNetTrainer__nnUNetPlans__3d_fullres/dataset.json" $2
nnUNetv2_predict -d Dataset017_CM -i "$2/data" -o $2  -f  0 1 2 3 4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans -npp  1 -nps 1 --save_probabilities
nnUNetv2_apply_postprocessing -i $2 -o $2 -pp_pkl_file "${script_dir}/network/Dataset017_CM/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl" -np 1 -plans_json "${script_dir}/network/Dataset017_CM/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/plans.json"
python3 "${script_dir}/postStep.py" -i $2 -m "pixel" -mo "abdom"
