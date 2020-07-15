# COVID-19 Drug Design using Generative RNN-LSTM

COVID-19 is an infectious disease caused by a newly discovered strain of coronavirus (SARS-CoV-2), a type of virus known to cause respiratory infections in humans. This new strain was unknown before December 2019, when an outbreak of a pneumonia of unidentified cause emerged in Wuhan, China.

Basic Local Alignment Search Tool (BLAST) results show close homology to the bat Coronavirus. A crystal structure of the main protease of the virus was obtained by Liu et al., found at https://www.rcsb.org/structure/6LU7

Since the outbreak, researchers have been collaborating and working closely to stop the spread of the disease and to propose possible treatment plans. New advances in machine intelligence have introduced algorithms that can learn important patterns from vast amounts of data, approaching expert-level of ability in some tasks. This means that anyone with these models can contribute to the global research effort. 

This project uses many ideas and implementations developed by others, and bring them together towards a common task. My main reference was [Topazape's](https://github.com/topazape/LSTM_Chem) repo which implements the paper [Generative Recurrent Networks for De Novo Drug Design](https://doi.org/10.1002/minf.201700111). 

The aim of this project is to find drug candidates (ligand) with a high binding affinity with the COVID-19 main protease using deep learning.

1. Outline of the problem and introduction

2. Dataset preparation

3. Train LSTM-based RNN model

4. Generate SMILES strings

5. Use transfer learning to fine-tune model, generating molecules that are structurally similar to potential protease inhibitors of COVID-19

6. Use PyRx to get binding scores of molecules with SARS-CoV-2 main protease

7. Report highest scoring candidates 


## Requirements

This model is built using Python 3.7, and utilizes the following packages;

- numpy
- pandas
- tensorflow
- tqdm
- Bunch
- matplotlib 
- RDKit 
- scikit-learn

## Dataset Preparation

Datasets from two sources: i) [Moses data set](https://github.com/molecularsets/moses) and ii) [ChEMBL data set](https://www.ebi.ac.uk/chembl/) were combined. Together these two data sets represent about 2.5 million smiles.

Preprocess dataset to remove duplicates, salts, stereochemical information, nucleic acids and long peptides.

In terminal, cd to the file and run python cleanup_smiles.py datasets/all_smiles.txt datasets/all_smiles_clean.txt

After cleaning the smiles using the cleanup_smiles.py script and only retaining smiles between 34 to 128 characters in length, './datasets/all_smiles_clean.txt' contains the final list of 180793 smiles on which the model was trained.


#### Potential COVID-19 protease inhibitors were included for model fine-tuning using transfer learning

According to this paper - [Binding site analysis of potential protease inhibitors of COVID-19 using AutoDock](https://link.springer.com/article/10.1007/s13337-020-00585-z)

SMILES obtained from [PubChem](https://pubchem.ncbi.nlm.nih.gov/)

| Protease Inhibitor | SMILES |
| ---- | ---- |
| Remdesivir | CCC(CC)COC(=O)C(C)NP(=O)(OCC1C(C(C(O1)(C#N)C2=CC=C3N2N=CN=C3N)O)O)OC4=CC=CC=C4 |
| Nelfinavir | CC1=C(C=CC=C1O)C(=O)NC(CSC2=CC=CC=C2)C(CN3CC4CCCCC4CC3C(=O)NC(C)(C)C)O |
| Lopinavir | CC1=C(C(=CC=C1)C)OCC(=O)NC(CC2=CC=CC=C2)C(CC(CC3=CC=CC=C3)NC(=O)C(C(C)C)N4CCCNC4=O)O |
| Ritonavir | CC(C)C1=NC(=CS1)CN(C)C(=O)NC(C(C)C)C(=O)NC(CC2=CC=CC=C2)CC(C(CC3=CC=CC=C3)NC(=O)OCC4=CN=CS4)O |
| Darunavir | CC(C)CN(CC(C(CC1=CC=CC=C1)NC(=O)OC2COC3C2CCO3)O)S(=O)(=O)C4=CC=C(C=C4)N |
| Atazanavir | CC(C)(C)C(C(=O)NC(CC1=CC=CC=C1)C(CN(CC2=CC=C(C=C2)C3=CC=CC=N3)NC(=O)C(C(C)(C)C)NC(=O)OC)O)NC(=O)OC |

These protease inhibitors SMILES are added into datasets/protease_inhibitors_for_fine-tune.txt


## Train LSTM-based RNN model to generate SMILES

#### Configuration
See `config.json` in base_experiment.

| parameters | meaning |
| ---- | ---- |
| exp_name | experiment name (default: `LSTM_Chem`) |
| data_filename | filepath for training the model (`SMILES file with newline as delimiter`) |
| data_length | number of SMILES for training. If you set 0, all the data is used (default: `0`) |
| units | size of hidden state vector of two LSTM layers (default: `256`, see the paper) |
| num_epochs | number of epochs (`42`) |
| optimizer | optimizer (default: `adam`) |
| seed | random seed (default: `71`) |
| batch_size | batch size (default: `512`) |
| validation_split | split ratio for validation (default: `0.10`) |
| varbose_training | verbosity mode (default: `True`) |
| checkpoint_monitor | quantity to monitor (default: `val_loss`) |
| checkpoint_mode | one of {`auto`, `min`, `max`} (default: `min`) |
| checkpoint_save_best_only | the latest best model according to the quantity monitored will not be overwritten (default: `False`)|
| checkpoint_save_weights_only | If True, then only the model's weights will be saved (default: `True`)|
| checkpoint_verbose | verbosity mode while `ModelCheckpoint` (default: `1`) |
| tensorboard_write_graph | whether to visualize the graph in TensorBoard (defalut: `True`) |
| sampling_temp | sampling temperature (default: `0.75`, see the paper) |
| smiles_max_length | maximum size of generated SMILES (symbol) length (default: `128`)|
| finetune_epochs | epochs for fine-tuning (default: `12`, see the paper) |
| finetune_batch_size | batch size of finetune (default: `1`) |
| finetune_filename | filepath for fine-tune the model (`SMILES file with newline as delimiter`) |


## Docking procedure with PyRx:

Download here: https://pyrx.sourceforge.io

PyRX ligand docking tutorial https://www.youtube.com/watch?v=2t12UlI6vuw

1. Open the structure of the protein and ligand complex (.cif crystallographic information file) https://www.rcsb.org/3d-view/6LU7
2. Select the ligand chain, delete the ligand, and save the file as a .pdb
3. Process generated SMILES and save it as .sdf file
4. Follow the video tutorial to get binding scores and save it as a csv file

### Final Results 

![final result](https://github.com/jiaminho/COVID-19-Drug-Discovery/blob/master/images/final-result.png)

| Ligand | Binding Affinity (kcal/mol) |
| ---- | ---- |
| Lopinavir | -6.9 |
| Generated_3 | -6.8 |
| Generated_5 | -6.3 |
| Darunavir | -6.1 |
| Generated_1 | -6.0 |
| Generated_6 | -5.7 |
| Generated_2 | -5.5 |
| Nelfinavir | -5.5 |
| Atazanavir | -5.4 |
| Remdesivir | -5.3 |
| Ritonavir | -5.3 |
| Generated_4 | -5.2 |

#### COVID-19 Main Protease (6LU7) in complex with Generated_3 Molecule

![generated3](https://github.com/jiaminho/COVID-19-Drug-Discovery/blob/master/pyrx/AAAC-generated3.png)

## References

Generative Recurrent Network for De Novo Drug Design https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5836943/ https://github.com/topazape/LSTM_Chem

Binding site analysis of potential protease inhibitors of COVID-19 using AutoDock https://link.springer.com/article/10.1007/s13337-020-00585-z

PubChem data related to COVID-19 https://pubchemdocs.ncbi.nlm.nih.gov/covid-19

Refer here for COVID-19 drugs in clinical trials https://pubchem.ncbi.nlm.nih.gov/#tab=compound&query=covid-19%20clinicaltrials

Crystal structure of COVID-19 main protease https://www.rcsb.org/structure/6LU7

RDKit https://www.rdkit.org/docs/GettingStartedInPython.html


## Related Work

https://github.com/topazape/LSTM_Chem

https://github.com/forkwell-io/fch-drug-discovery

https://github.com/mattroconnor/deep_learning_coronavirus_cure

https://github.com/tmacdou4/2019-nCov
