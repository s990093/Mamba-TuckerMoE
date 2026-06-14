#!/bin/bash
# Upload NPZ checkpoint to Kaggle

NPZ_FILE="/home/hungwei/llm/sft_cot_bundle/output/checkpoint_sft_cot_s18500.npz"
DATASET_NAME="${1:-sft-cot-checkpoint-s18500}"
DATASET_ID="${2:-}"

if [ -z "$DATASET_ID" ]; then
    # Create new dataset
    echo "Creating new Kaggle dataset: $DATASET_NAME"

    # Create dataset metadata
    mkdir -p /tmp/kaggle_dataset
    cat > /tmp/kaggle_dataset/dataset-metadata.json <<EOF
{
  "id": "$DATASET_NAME",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ],
  "keywords": ["llm", "checkpoint", "sft-cot", "mamba3"],
  "collaborators": [],
  "data": []
}
EOF

    cp "$NPZ_FILE" /tmp/kaggle_dataset/

    echo "Uploading dataset..."
    kaggle datasets create -p /tmp/kaggle_dataset --dir-mode zip

else
    # Upload to existing dataset
    echo "Uploading to existing dataset: $DATASET_ID"
    mkdir -p /tmp/kaggle_update
    cp "$NPZ_FILE" /tmp/kaggle_update/
    kaggle datasets version -p /tmp/kaggle_update -m "Upload checkpoint_sft_cot_s18500.npz" --dir-mode zip
fi

echo "Upload complete!"
