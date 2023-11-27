#!/bin/bash

# permissions: chmod +x download_all_weights.sh
# RUN: ./download_all_weights.sh

# # aws s3 cp --recursive --no-sign-request s3://bossdb-datalake/public/kasthuri-challenge-pretrained-weights/ ./pretrained_model_weights/

# Download all pre-trained weights from the specified S3 bucket into a local directory.
download_all_weights() {
    local s3_bucket="s3://bossdb-datalake/public/kasthuri-challenge-pretrained-weights/"
    local destination_path="./pretrained_model_weights/"

    # Download all model weights from S3
    aws s3 cp --recursive --no-sign-request "$s3_bucket" "$destination_path"

    # Check if the download was successful
    if [ $? -eq 0 ]; then
        echo "Successfully downloaded all pre-trained weights from S3"
    else
        echo "Error downloading model weights from S3"
        exit 1
    fi
}

download_all_weights
