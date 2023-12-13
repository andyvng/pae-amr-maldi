import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class MALDIDataset(Dataset):
    def __init__(self, 
                 input_path,
                 label_path,
                 ids_path,
                 normalising_factor,
                 model="VanillaDNN",
                 label_col="class",
                 get_ids=False):
        super(MALDIDataset, self).__init__()
        self.input_path = input_path
        self.label_col = label_col
        self.labels = pd.read_csv(label_path,
                                  header=0)
        # remove sample without label
        self.labels = self.labels[self.labels[label_col].notnull()]\
                            .reset_index(drop=True)
        self.ids = pd.read_csv(ids_path, header=None).values
        self.normalising_factor = normalising_factor
        self.model = model
        self.get_ids = get_ids

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        tmp_id = self.ids.item(idx)
        y = self.labels[self.labels['id'] == tmp_id][self.label_col]\
            .to_numpy().astype(int).item()
        feature_fp = os.path.join(self.input_path,
                                  f"{tmp_id}.txt")
        X = pd.read_csv(feature_fp, 
                        header=None, 
                        names=['mass', 'intensity'])['intensity']
        X = X.to_numpy().squeeze() / self.normalising_factor
        
        # Add dimension to fit [Channels, Heigth, Width]
        if self.model != "VanillaDNN": 
            X = X[np.newaxis, ...]
            X = X[np.newaxis, ...]
        
        if self.get_ids:
            return torch.tensor(X).float(), y, tmp_id
        else:
            return torch.tensor(X).float(), y
    

def plot_confusion_matrix(y_test, y_scores, fp):
    cm = confusion_matrix(y_test, y_scores)
    plt.figure(figsize=(7, 7), dpi=100)
    heatmap_plot = sns.heatmap(cm,
                               cmap='Reds')
    
    heatmap_plot.figure.savefig(fp, dpi=300)
    return
        



