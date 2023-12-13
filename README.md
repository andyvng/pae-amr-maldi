## Predicting Pseudomonas aeruginosa drug resistance using artificial intelligence and clinical MALDI-TOF mass spectra

This is the code repository of the [paper](https://www.biorxiv.org/content/early/2023/10/26/2023.10.25.563934)

```
@article{Nguyen2023.10.25.563934,
	author = {Hoai-An Nguyen and Anton Y Peleg and Jiangning Song and Bhavna Antony and Geoffrey I Webb and Jessica A Wisniewski and Luke V Blakeway and Gnei Z Badoordeen and Ravali Theegala and Helen Zisis and David L Dowe and Nenad Macesic},
	doi = {10.1101/2023.10.25.563934},
	journal = {bioRxiv},
	title = {Predicting Pseudomonas aeruginosa drug resistance using artificial intelligence and clinical MALDI-TOF mass spectra},
	url = {https://www.biorxiv.org/content/early/2023/10/26/2023.10.25.563934},
	year = {2023}}
```

### System requirements and Installation

Create conda environment

```
conda env create -f envs/maldi_amr.yml
```

Activate conda environment

```
conda activate maldi_amr
```

### Spectra preprocessing

To run this preprocessing pipeline, please extract your raw spectra to csv file with the first column contains mass (m/z) values and the second column contains the corresponding intensity values. The script will search for all spectra files (.txt extension) in the specified input directory.  
**IMPORTANT**: Please do not include headers or you need to use the skiprows option.

```
cd src/preprocess
python preprocess.py ${INPUT_DIR} ${OUTPUT_DIR}
```

**Arguments**  
--binned_output_dir: create 1-Da binning files  
--delimiter: specify delimiter other than comma  
--skiprows: skip metadata rows (if needed)

<!-- ### Dynamic binning profile generation

```
cd src

``` -->

### AMR prediction

```
cd src
python src/predict.py ${CONFIG_PATH} ${ANTIMICROBIAL} ${MODEL}
```

Example of config file and the corresponding input files can be found [here](./examples/prediction). Antimicrobial should be aligned with columns in [susceptibility testing file](./examples/prediction/ast_data.csv). Model list includes logistic regression ('LR'), random forest ('RF'), support vector machine ('SVM'), XGBoost ('XGB'), and multi-layer perceptron ('MLP').

<!-- ### Stacking -->

<!-- ### Latent representation learning

Please visit [this README](./vit/PRETRAIN.md) for further information -->
