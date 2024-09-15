# Introduction
This set of scripts that is used to work with azure machine learning studio.

# Prerequisites
- Compute node has alredy been created.


# Setup environment
```
python -m venv .venv
pip install virtualenv
[windows].venv\Scripts\activate
[linux]source .venv/bin/activate
pip install -r requirements.txt
```
# Scripts

Convert images to numpy arrays and store them.
```
python save_as_npy.py
```

```
python script.py
```

# Download files to the command line

# Step 1: Install Azure CLI (if not already installed)
# curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Step 2: Login to Azure
az login

# Step 3: Set variables
CONTAINER_NAME="<redacted>"
STORAGE_ACCOUNT="<redacted>"
DESTINATION_DIR="./oct"
ACCOUNT_KEY="<redacted>"
FOLDER_PREFIX="oct-500/"

# Step 4: Download the images
az storage blob download-batch \
    --account-name $STORAGE_ACCOUNT \
    --source $CONTAINER_NAME \
    --destination $DESTINATION_DIR \
    --pattern "$FOLDER_PREFIX*" \
    --account-key $ACCOUNT_KEY

# Step 5: Upload results
az storage blob upload \
    --account-name $STORAGE_ACCOUNT \
    --container-name $CONTAINER_NAME \
    --name ten.png \
    --file ./10.png \
    --account-key $ACCOUNT_KEY

# Reference

[AUTOML-Classification](https://learn.microsoft.com/en-us/training/modules/find-best-classification-model-automated-machine-learning/1-introduction)

# Data

Public data 'OCTA-500' can be downloaded at: https://ieee-dataport.org/open-access/octa-500.

The partitions of data of our experiments are provided at `train_volume_names.json` and `test_volume_names.json`.


# Train

Here we provide a more stable version recently discovered, which differs slightly from the version described in the paper.

1. Train a 2DAE. Run `./scripts/train_AE2D.py` and fill following args:

```python
parser.add_argument("--exp_name", type=str, default='AE2D')
parser.add_argument('--result_root', type=str, default='./results')
parser.add_argument('--data_root', type=str, default='./oct')
```

2. Train NHAE. Run `./scripts/train_NHAE.py` and fill following args:
```python
parser.add_argument("--exp_name", type=str, default='NHAE')
parser.add_argument('--result_root', type=str, default='path/to/save/dir')
parser.add_argument('--data_root', type=str, default='path/to/OCT')
parser.add_argument('--image_npy_root', type=str,default='path/to/volume/npy')
```

3. Train LDM3D. Run `./scripts/train_LDM3D.py` and fill following args:
```python
parser.add_argument("--exp_name", default='LDM3D')
parser.add_argument('--result_root', type=str, default='path/to/save/dir')
parser.add_argument('--first_stage_ckpt', type=str,default='path/to/NHVQVAE/ckpt')
parser.add_argument('--latent_root', type=str,default='path/to/NHVQVAE/latent')
```
4. Train LDM2D_refiner. Run `./scripts/train_LDM2D.py` and fill following args:
```python
parser.add_argument("--exp_name", type=str, default='LDM2D')
parser.add_argument('--result_root', type=str, default='path/to/dir')
parser.add_argument('--first_stage_ckpt', type=str, default='path/to/vqgan2d/ckpt')
parser.add_argument('--latent_1_root', type=str, default='path/to/3D/latent')
parser.add_argument('--latent_2_root', type=str, default='path/to/2D/latent')
```

5. Train multi-slice_decoder. Run `./scripts/train_AE.py` and fill following args:
```python
parser.add_argument("--exp_name", type=str, default='AE')
parser.add_argument('--result_root', type=str, default='path/to/save/dir')
parser.add_argument('--image_npy_root', type=str, default='path/to/volume/npy')
```

# Generate

We split the generation procedure into three stages.

1.  Generate 3D latents. Run `./scripts/gen_LDM3D.py` and fill following args:
```python
parser.add_argument('--result_save_dir', type=str, default='path/to/save/dir')
parser.add_argument('--first_stage_ckpt', type=str,
default='path/to/NHVQVAE/ckpt')
parser.add_argument('--ldm1_ckpt', type=str,
default='path/to/LDM3D/ckpt')
parser.add_argument('--ldm2_ckpt', type=s
```

2.  Refine latents. Run `./scripts/gen_LDM2D.py` and fill following args:
```python
parser.add_argument('--result_save_dir', type=str, default='path/to/save/dir')
parser.add_argument('--first_stage_ckpt', type=str, default='path/to/NHVQVAE/ckpt')
parser.add_argument('--ldm1_ckpt', type=str, default='path/to/LDM3D/ckpt')
parser.add_argument('--ldm2_ckpt', type=str, default='path/to/LDM2D_refiner/ckpt')
datamodule = testDatamodule(latent_root='path/to/ldm1_latent')
```

3.  Decode latents to images. Run `./scripts/gen_decodelatents.py` and fill following args:
```python
parser.add_argument('--result_save_dir', type=str,default='path/to/save/dir')
parser.add_argument('--result_save_name', type=str, default='save name')
parser.add_argument('--first_stage_ckpt', type=str, default='path/to/VQVAE_w_adaptor/ckpt')
parser.add_argument('--ldm2_latent', type=str, default='path/to/saved/ldm2_latent')
```
