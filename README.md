# MTCL-DTA
A Multimodal DTA Prediction Method Based on Triple-view Contrastive Learning
# Dependency
Python3.8  
torch2.0.1  
Cuda11.7  
torch-geometric 2.0.4  
rdkit 2022.03.2

# Data
Unpacking data.zip.
The target molecule graphs data is downloaded from https://drive.google.com/open?id=1rqAopf_IaH3jzFkwXObQ4i-6bUUwizCv. Move the downloaded folders to the directory of each dataset.
/data/davis/aln
/data/davis/pconsc4
/data/kiba/aln
/data/kiba/pconsc4

# Pretrain Model
SMILES-BERT is downloaded from:https://huggingface.co/unikei/bert-base-smiles
ESM-2 is downloaded from:https://huggingface.co/facebook/esm2_t6_8M_UR50D

# Running
python inference.py --cuda 0
