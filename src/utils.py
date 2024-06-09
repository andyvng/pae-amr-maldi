import os
import numpy as np
import pandas as pd
import torch
import scipy.signal
import scipy.stats

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, log_loss
from sklearn.metrics import confusion_matrix, precision_score, recall_score


class JointDataset(Dataset):
    '''
    Creating data object
    '''

    def __init__(self, X, y, balanced=False):
        self.X = X
        self.y = y
        self.balanced = balanced

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DatasetFromDir(Dataset):
    '''
    Loading dataset from working directory and lable files
    '''

    def __init__(self,
                 working_dir,
                 label_path,
                 ids_path,
                 label_list,
                 max_intensity=0.009055):
        self.working_dir = working_dir
        self.label_path = label_path
        self.ids = pd.read_csv(ids_path, header=None).values
        self.max_intensity = max_intensity
        self.label_list = label_list

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx, bins=None):
        self.code = self.ids.item(idx)
        tmp_df = pd.read_csv(os.path.join(
            self.working_dir, f"{self.code}.txt"), header=None, names=["mass", "intensity"])
        masses = tmp_df["mass"].values
        intensities = tmp_df["intensity"].values

        if bins is None:
            bins = [i + 2000 for i in range(18001)]

        binned_intensities = scipy.stats.binned_statistic(masses,
                                                          intensities,
                                                          statistic="max",
                                                          bins=bins).statistic

        np.nan_to_num(binned_intensities, copy=False, nan=0)
        # binned_intensities = np.clip(binned_intensities, a_min=0, a_max=1)
        self.spectra = binned_intensities.squeeze() / self.max_intensity
        label_df = pd.read_csv(self.label_path, header=0)
        self.labels = label_df.copy(
        ).loc[label_df['id'] == self.code, self.label_list].to_numpy().squeeze()
        return torch.tensor(self.spectra), torch.tensor(self.labels)


class EarlyStopper:
    """
    Setting criteria for early stopping
    https://stackoverflow.com/a/73704579
    """

    def __init__(self, patience=10, min_delta=20):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def load_and_preprocess_spectra(input_dir,
                                ast_df,
                                antimicrobial,
                                ids,
                                bins,
                                statistic='max'):
    '''
    Loading spectra from directory and preprocess spectra
    '''

    X = None
    y = None

    for id in ids:
        # masses = [ (2000 + i) for i in range(18000) ]
        # intensities = np.loadtxt(os.path.join(input_dir, f"{id}.txt")).reshape(1, -1)
        tmp_df = pd.read_csv(os.path.join(input_dir, f"{id}.txt"),
                             header=None,
                             names=["mass", "intensity"])
        masses = tmp_df['mass'].values
        intensities = tmp_df['intensity'].values

        # Bin intensities and return bin means
        binned_intensities = scipy.stats.binned_statistic(masses,
                                                          intensities,
                                                          statistic=statistic,
                                                          bins=bins).statistic
        np.nan_to_num(binned_intensities, copy=False, nan=0)

        if X is not None:
            X = np.concatenate([X, binned_intensities.reshape(1, -1)], axis=0)
        else:
            X = binned_intensities.reshape(1, -1)

    # getting Y label
    tmp_df = pd.DataFrame({'id': ids})
    y = tmp_df.merge(ast_df, how='left', on='id')[antimicrobial].values

    nan_mask = np.isnan(y)

    return X[~nan_mask], y[~nan_mask], tmp_df[~nan_mask]['id'].values


def load_data_for_voting(input_dir,
                         feature_path,
                         ast_df,
                         antimicrobial,
                         ids,
                         bins,
                         statistic='max'):
    '''
    Loading spectra from directory and preprocess spectra
    '''

    X_binned = None
    y = None

    for id in ids:
        # masses = [ (2000 + i) for i in range(18000) ]
        # intensities = np.loadtxt(os.path.join(input_dir, f"{id}.txt")).reshape(1, -1)
        tmp_df = pd.read_csv(os.path.join(input_dir, f"{id}.txt"),
                             header=None,
                             names=["mass", "intensity"])
        masses = tmp_df['mass'].values
        intensities = tmp_df['intensity'].values

        # Bin intensities and return bin means
        binned_intensities = scipy.stats.binned_statistic(masses,
                                                          intensities,
                                                          statistic=statistic,
                                                          bins=bins).statistic
        np.nan_to_num(binned_intensities, copy=False, nan=0)

        if X_binned is not None:
            X_binned = np.concatenate(
                [X_binned, binned_intensities.reshape(1, -1)], axis=0)
        else:
            X_binned = binned_intensities.reshape(1, -1)

    # getting Y label
    tmp_df = pd.DataFrame({'id': ids})

    X_variant = pd.read_csv(feature_path, header=0)
    X_variant = tmp_df.merge(X_variant, how='left', on='id')
    selected_col = list(X_variant.columns)
    selected_col.remove("id")
    X_variant = X_variant[selected_col].values

    y = tmp_df.merge(ast_df, how='left', on='id')[antimicrobial].values

    nan_mask = np.isnan(y)

    return X_binned[~nan_mask], X_variant[~nan_mask], y[~nan_mask]


def calculate_metrics(y_true, y_prob, y_pred):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    ll = log_loss(y_true, y_prob)
    tn, fn, tp, fp = confusion_matrix(y_true, y_pred)

    return acc, auc, f1, precision, recall, ll, tn, fn, tp, fp