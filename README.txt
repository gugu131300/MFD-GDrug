# MFD–GDrug
MFD–GDrug: multimodal feature fusion-based deep learning for GPCR–drug interaction prediction

This model has been publised in Methods (March 2024), you can find the full publication here: https://www.sciencedirect.com/science/article/pii/S1046202324000355?dgcid=coauthor

## Abstract
The accurate identification of drug–protein interactions (DPIs) is crucial in drug development, especially concerning G protein-coupled receptors (GPCRs), which are vital targets in drug discovery. However, experimental validation of GPCR–drug pairings is costly, prompting the need for accurate predictive methods. To address this, we propose MFD–GDrug, a multimodal deep learning model. Leveraging the ESM pretrained model, we extract protein features and employ a CNN for protein feature representation. For drugs, we integrated multimodal features of drug molecular structures, including three-dimensional features derived from Mol2vec and the topological information of drug graph structures extracted through Graph Convolutional Neural Networks (GCN). By combining structural characterizations and pretrained embeddings, our model effectively captures GPCR–drug interactions. Our tests on leading GPCR–drug interaction datasets show that MFD–GDrug outperforms other methods, demonstrating superior predictive accuracy.

## Results
This study successfully developed a novel DPI prediction model named MFD–GDrug. The model employs advanced deep learning techniques for drug and protein feature extraction, integrating GCN and pretrained models (Mol2vec and ESM) to enhance the accuracy of predicting drug–protein interactions.

## Run the MFD–GDrug model for DPI prediction
### Download the data.
The GPCR dataset is divided into train.csv and test.csv, which can be downloaded from "https://github.com/gugu131300/MFD-GDrug".

**Protein Representation**:
1、esm pre-training model extracts protein 320-dimensional features, pro_esmtr.npy is an array of len(train)*320 dimensions
2、protein_trainembeddings400.npy is to use create_data.py file to get len(train)*300 dimensional protein embedding, which is then used as the original input for the part of 1D CNN network in the model
    $  create_data.py

**Drug Representation**:
1、mol2vec pre-training model to extract drug 300-dimensional features, dr_moltr.npy is a len(train)*300-dimensional array
2、drug_train.npy is a column name "COMPOUND_SMILES", the content of the train array of all the SMILES characters of the drug

**Label**:
label_train.npy is an array of labels for the train dataset

**Model**:
    $ python core.py

**train&test**:
    $ python main.py