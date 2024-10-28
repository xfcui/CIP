# Enhanced Protein-Ligand Affinity Prediction with Conditional Updating and Proximity Embedding
The protein-ligand affinity prediction task aims topredict the binding strength of small molecule ligands to specificproteins, which is crucial in the fields of drug design andmolecular biology, and can accelerate the drug discovery. Thestructurecomplementarity between protein and ligand plays acritical role in determining binding strength , but most of currentdeep learning-based affinity prediction models usually extractedthe features of protein and ligand by these two detached modules.which limits the exchange of information for capturing interactions and struggles to capture proteins’ important residues.To address these limitations. we introduce CIP. which takes the combination of GNN, Conditional Updating and Proximity Embedding for the first time. Compared to existing models, CIP has several significant advantages. First, Conditional updating modifies the ligand’s local features based on the protein’s global features, and vice versa , enhancing structural complementarity
to capture intricate interactions . Second, encoding the relative distances between proximal residue-atom pairs highlights critical residues. Additionally, our model integrates covalent and noncovalent interactions to obtain more comprehensive graph representations. Experiments on the PDBbind 2016 benchmark demonstrate that CIP outperforms the original method with improvements of 2.3%, 3.2%, 2.8%, 3.8%, and 3.2% across five baselines. Furthermore, visualization results reveal that CIP effectively captures intricate interactions and crucial residues.

```
Authors: Shuai Cui, Xuan Zhang, Zizheng Nie, Haitao Jiang, Guishan Cui*, Xuefeng Cui*
    - *: To whom correspondence should be addressed.
Contact: xfcui@email.sdu.edu.cn

```


## Usage

Clone this repository by:
```bash
git clone https://github.com/xfcui/CIP.git
```

## Processed
This repository contains the processed datasets and training weights for our project. You can access the files via the following link: [Download Processed Data and Training Weights](https://drive.google.com/drive/folders/14aFDFyZ-a3tGJEObvgDewgyUjHBLGzNS?usp=sharing).


## Run
main:
```bash
chmod +x run.sh
./run.sh
```
experient1-1:
```bash
chmod +x exp11.sh
./exp11.sh
```
experient1-2:
```bash
chmod +x exp12.sh
./exp12.sh
```
experient2:
```bash
chmod +x exp2.sh
./exp2.sh
```

## Acknowledgments

- Thank you to all the contributors ！
