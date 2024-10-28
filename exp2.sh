#!/bin/bash

# Activate your  environment 
source /path/to/your/venv/bin/activate

# Run Jupyter Notebook 
jupyter nbconvert --to notebook --execute --inplace ./contribution.ipynb
