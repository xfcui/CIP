#!good lucky

conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 cudatoolkit=11.8 -c pytorch -c nvidia


conda install -y -c conda-forge prody
conda install -y -c conda-forge openbabel
conda install -y -c conda-forge biopandas
conda install -y -c anaconda yaml
conda install -y -c conda-forge pyyaml
conda install -y -c conda-forge easydict
conda install -y -c conda-forge psutil
conda install -y -c conda-forge gpustat
conda install -y -c conda-forge tqdm
conda install -y -c anaconda joblib
conda install dglteam/label/cu118::dgl

pip install torchdata==0.7.1
pip install pandas==1.4.2
pip install scipy==1.14.1
pip install rdkit
pip install scikit-learn
pip install pydantic

#If an issue occurs, please run `pip uninstall` and then reinstall with `pip install`.

pip uninstall matplotlib
pip install matplotlib

pip uninstall kiwisolver
pip install kiwisolver

pip install plotly dash

pip uninstall scipy
pip install scipy