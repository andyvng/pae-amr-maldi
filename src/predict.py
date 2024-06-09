import os
import pandas as pd
import numpy as np
import argparse
import logging
import json
import pickle

from utils import load_and_preprocess_spectra
from train import tune_model

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.utils import class_weight

from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, confusion_matrix, log_loss, precision_score, recall_score


def main():
    np.random.seed(222)

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help='Config file path')
    parser.add_argument("antimicrobial", type=str, help='Antimicrobial')
    parser.add_argument(
        "model", type=str, help='Model type. Currently support LR, RF, SVM, XGB, MLP')
    parser.add_argument("--get_ids",
                        type=int,
                        default=0)

    args = parser.parse_args()
    antimicrobial = args.antimicrobial
    model = args.model

    # Load config file
    with open(args.config, 'r') as config_input:
        config = json.load(config_input)

    os.makedirs(config["output_dir"], exist_ok=True)

    # Loading ids
    train_ids = pd.read_csv(config["train_id_path"], header=None)\
                  .values.flatten()
    test_ids = pd.read_csv(config["test_id_path"], header=None)\
                 .values.flatten()

    # Loading AST label
    ast_df = pd.read_csv(config["ast_label_path"])

    if config["bins_path"]:
        # Load bins
        with open(config["bins_path"], 'rb') as bins_input:
            features = pickle.load(bins_input)

        if type(features) == np.ndarray:
            pass
        else:
            features = features.values

        bins = features.astype('int')
        bins = np.append(bins, [20000])  # Adding the righmost edge

        # Load training and testing dataset
        X_train, y_train, _ = load_and_preprocess_spectra(config["input_dir"],
                                                          ast_df,
                                                          antimicrobial,
                                                          train_ids,
                                                          bins)
        X_test, y_test, ids = load_and_preprocess_spectra(config["input_dir"],
                                                          ast_df,
                                                          antimicrobial,
                                                          test_ids,
                                                          bins)
    else:
        if config['input_dir'].endswith("csv"):
            X = pd.read_csv(config['input_dir'])
        elif config['input_dir'].endswith("feather"):
            X = pd.read_feather(config['input_dir'])
        else:
            raise TypeError(
                f"Unsupported extension for input file: {config['input_dir'].split('.')[-1]}")

        features = list(X.columns)
        features.remove('id')

        X_train = pd.DataFrame({'id': train_ids}).merge(X,
                                                        how='inner',
                                                        on='id')\
            .drop(labels=['id'], axis=1)\
            .to_numpy()
        y_train = pd.DataFrame({'id': train_ids}).merge(ast_df,
                                                        how='inner',
                                                        on='id')[antimicrobial].values
        X_test = pd.DataFrame({'id': test_ids}).merge(X,
                                                      how='inner',
                                                      on='id')\
            .drop(labels=['id'], axis=1)\
            .to_numpy()
        y_test = pd.DataFrame({'id': test_ids}).merge(ast_df,
                                                      how='inner',
                                                      on='id')[antimicrobial].values

    # Normalization
    norm_coeff = max(np.max(X_train), np.max(X_test))
    X_train = X_train / norm_coeff
    X_test = X_test / norm_coeff

    # Adding extra features
    if config['extra_features_path']:
        # Don't forget one-hot encoding and scaling!
        extra_feature_df = pd.read_csv(config['extra_features_path'])

        extra_features = list(extra_feature_df.columns)
        extra_features.remove('id')
        features = np.concatenate((features, extra_features))

        train_extra_feature_df = pd.DataFrame({'id': train_ids}).merge(
            extra_feature_df, on='id', how='inner').drop(labels=['id'], axis=1).to_numpy()
        test_extra_feature_df = pd.DataFrame({'id': test_ids}).merge(
            extra_feature_df, on='id', how='inner').drop(labels=['id'], axis=1).to_numpy()

        X_train = np.concatenate([X_train, train_extra_feature_df], axis=1)
        X_test = np.concatenate([X_test, test_extra_feature_df], axis=1)

    # Oversampling with SMOTE
    if config["smote_id_path"]:
        smote_ids = pd.read_csv(
            config["smote_id_path"], header=None).values.flatten()
        X_smote, y_smote = load_and_preprocess_spectra(config["input_dir"],
                                                       ast_df,
                                                       antimicrobial,
                                                       smote_ids,
                                                       bins)
        sm = SMOTE(random_state=222, sampling_strategy='not majority')
        X_res, y_res = sm.fit_resample(X_smote, y_smote)

        # Update training set
        remaining_train_ids = list(np.setdiff1d(train_ids, smote_ids))
        print(f"Number of remaining train ids: {len(remaining_train_ids)}")
        if len(remaining_train_ids) > 0:
            X_train, y_train = load_and_preprocess_spectra(config["input_dir"],
                                                           ast_df,
                                                           antimicrobial,
                                                           remaining_train_ids,
                                                           bins)

            X_train = np.concatenate([X_train, X_res], axis=0)
            y_train = np.concatenate([y_train, y_res], axis=0)
        else:
            X_train, y_train = X_res, y_res

    print(f"X_train: {X_train.shape}\ty_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}\ty_test: {y_test.shape}")

    clf, training_time = tune_model(X_train,
                                    y_train,
                                    model,
                                    antimicrobial,
                                    config["output_dir"])

    y_proba = clf.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(np.min(y_proba), np.max(y_proba), 1000)
    f1_scores = []

    for threshold in thresholds:
        tmp_y_pred = (y_proba > threshold).astype('int')
        tmp_f1 = f1_score(y_test, tmp_y_pred)
        f1_scores.append(tmp_f1)

    optimal_threshold = thresholds[np.argmax(f1_scores)]
    y_pred = (y_proba > optimal_threshold).astype('int')

    bacc = np.round(balanced_accuracy_score(y_test, y_pred), 2)
    acc = np.round(accuracy_score(y_test, y_pred), 2)
    auc = np.round(roc_auc_score(y_test, y_proba), 2)
    f1 = np.round(f1_score(y_test, y_pred), 2)
    precision = np.round(precision_score(y_test, y_pred), 2)
    recall = np.round(recall_score(y_test, y_pred), 2)
    ll = np.round(log_loss(y_test, y_proba), 2)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    result_path = os.path.join(config["output_dir"],
                               f"result_{antimicrobial}_{model}.txt")

    with open(result_path, 'w') as result_output:
        result_output.write(
            f"bacc\tacc\tauroc\tf1\tprecision\trecall\tll\ttn\tfn\ttp\tfp\tthreshold\ttraining_time\n")
        result_output.write(
            f"{bacc}\t{acc}\t{auc}\t{f1}\t{precision}\t{recall}\t{ll}\t{tn}\t{fn}\t{tp}\t{fp}\t{optimal_threshold}\t{training_time}")

    result_df = pd.DataFrame()
    result_df['id'] = ids
    result_df['y_test'] = y_test
    result_df['y_proba'] = y_proba
    result_df['y_pred'] = y_pred
    result_df.to_csv(os.path.join(config["output_dir"], f"result_{antimicrobial}_{model}.csv"), 
                     index=False)

if __name__ == "__main__":
    main()