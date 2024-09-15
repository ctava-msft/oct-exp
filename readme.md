# Introduction
This instructions give setup and steps to train models to 
generate synthetic images of Optical coherence tomography (OCT).

# Prerequisites
- Compute node with GPU has alredy been created and runding.

# Setup environment
Run the following commands to setup a python virtual env.

```
python -m venv .venv
pip install virtualenv
[windows].venv\Scripts\activate
[linux]source .venv/bin/activate
pip install -r requirements.txt
```
# Data

Public data 'OCTA-500' can be downloaded at: https://ieee-dataport.org/open-access/octa-500.

The partitions of data of our experiments are provided at `train_volume_names.json` and `test_volume_names.json`.
You'll need a password and will have to email: chen2qiang@njust.edu.cn for it.

# Download images onto compute instance for training

## Step 1: Install Azure CLI (if not already installed)
```
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

## Step 2: Login to Azure
az login

## Step 3: Set variables
CONTAINER_NAME="<redacted>"
STORAGE_ACCOUNT="<redacted>"
DESTINATION_DIR="./oct"
ACCOUNT_KEY="<redacted>"
FOLDER_PREFIX="oct-500/"

## Step 4: Download the images
az storage blob download-batch \
    --account-name $STORAGE_ACCOUNT \
    --source $CONTAINER_NAME \
    --destination $DESTINATION_DIR \
    --pattern "$FOLDER_PREFIX*" \
    --account-key $ACCOUNT_KEY

# Utils

If needed, use this script to convert images to numpy arrays and store them.
```
python ./utils/save_as_npy.py
```

# Train

There are 5 steps in the training process. 
Create the checkpoints directory with 5 subfolers:
2DAE, NHAE, LDM3D, LDM2D, AE

To execute them, configue and run the following scripts.

1. Train a 2DAE. Configure and run `train_AE2D.py`.

2. Train NHAE. Configure and run `train_NHAE.py`.

3. Train LDM3D. Configure and run `train_LDM3D.py`.

4. Train LDM2D. Configure and run `train_LDM2D.py`.

5. Train AE. Configure and run `train_AE.py`.

## Retrieve images

After each training epoch images in training_progress can be upload to storage account to be downloaded and viewed.
```
az storage blob upload \
    --account-name $STORAGE_ACCOUNT \
    --container-name $CONTAINER_NAME \
    --name ten.png \
    --file ./10.png \
    --account-key $ACCOUNT_KEY
```

# Generate

The image generation procedure is split into three stages.

1.  Generate 3D latents. Configure and run `gen_LDM3D.py`.

2.  Refine latents. Configure and run `gen_LDM2D.py`.

3.  Decode latents to images. Configure and run `gen_decodelatents.py`.


# Reference(s)

[Compute Instance](https://learn.microsoft.com/en-us/azure/machine-learning/concept-compute-instance?view=azureml-api-2)