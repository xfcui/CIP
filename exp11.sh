#!/bin/bash
# This script runs the exp_encoderonly_tsne.ipynb notebook

# Activate your  environment 
source /path/to/your/venv/bin/activate

# Run Jupyter Notebook 
jupyter nbconvert --to notebook --execute --inplace ./exp_encoderonly_tsne.ipynb 
