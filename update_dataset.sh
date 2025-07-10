#!/bin/bash
set -e

mkdir -p Data
pushd Data
# mkdir -p reviewed

# pushd reviewed
# echo "Download the datasets from Alan's account that have been standardized to the rajagopal_no_arms.osim skeleton:"
# addb -d dev download --prefix "protected/us-west-2:e013a4d2-683d-48b9-bfe5-83a0305caf87" --pattern ".*No_Arm"
# popd

rm -rf processed
echo "Post-process the data to clean up common issues with CoP and finite-differenced accelerations, and standardize at 100 Hz:"
addb post-process /media/awear-omen/JaeBeom_T7/Data processed --geometry-folder "../Geometry/" --only-dynamics True --clean-up-noise True --sample-rate 100 --recompute-values True --root-history-len 10 --root-history-stride 1 --allowed-contact-bodies calcn_l calcn_r
# popd

rm -rf train
rm -rf dev
popd

pushd src
python3 main_origin.py create-splits  --data-folder ../Data
popd
echo "Done."