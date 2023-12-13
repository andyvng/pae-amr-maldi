import os
import pandas as pd
import numpy as np
import argparse
import json
import pickle

from utils import load_and_preprocess_spectra
from train import tune_model

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, confusion_matrix, log_loss, precision_score, recall_score


def main():
    np.random.seed(222)

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help='Config file path')
    parser.add_argument("antimicrobial",
                        type=str)
    parser.add_argument("model",
                        type=str)
    parser.add_argument("num_iterations",
                        type=int)
    parser.add_argument("folder_prefix",
                        type=str,
                        help="For iterating over multiple runs")

    args = parser.parse_args()
    model = args.model
    antimicrobial = args.antimicrobial

    # Load config file
    with open(args.config, 'r') as config_input:
        config = json.load(config_input)

    # Loading ids
    ids = pd.read_csv(config["id_path"], header=None)\
                  .values.flatten()
    id_df = pd.DataFrame({'id': ids})
    
    # Loading AST label
    ast_df = pd.read_csv(config["ast_label_path"])

    # Loading input data
    if config["bins_path"]:
        # Load bins
        with open(config["bins_path"], 'rb') as bins_input:
            features = pickle.load(bins_input)

        if type(features) == np.ndarray:
            pass
        else:
            features = features.values
            
        bins = features.astype('int')
        bins = np.append(bins, [20000]) # Adding the righmost edge

        # Load training and testing dataset
        X_val, y_val = load_and_preprocess_spectra(config["input_dir"],
                                                    ast_df,
                                                    antimicrobial,
                                                    ids, 
                                                    bins)
    else:
        if config['input_dir'].endswith("csv"):
            X = pd.read_csv(config['input_dir'])
        elif config['input_dir'].endswith("feather"):
            X = pd.read_feather(config['input_dir'])
        else:
            raise TypeError(f"Unsupported extension for input file: {config['input_dir'].split('.')[-1]}")
        features=list(X.columns)
        features.remove('id')
        
        X_val = id_df.merge(X,
                            how='inner',
                            on='id')\
                        .drop(labels=['id'], axis=1).to_numpy()
        y_val = id_df.merge(ast_df,
                            how='inner',
                            on='id')[antimicrobial].values

    # Normalization
    norm_coeff = config["normal_coeff"]
    X_val = X_val / norm_coeff

    # Adding extra information
    if config['extra_features_path']:
        extra_feature_df = pd.read_csv(config['extra_features_path']) 
        # Don't forget one-hot encoding and scaling!
        
        extra_features = list(extra_feature_df.columns)
        extra_features.remove('id')
        features = np.concatenate((features, extra_features))

        extra_feature_df = id_df.merge(extra_feature_df, on='id', how='inner').drop(labels=['id'], axis=1).to_numpy()

        X_val = np.concatenate([X_val, extra_feature_df], axis=1)
    
    print("Complete loading data")
    print(f"X: {X_val.shape}\ty: {y_val.shape}")
    # Evaluate for each rep
    for rep in range(args.num_iterations):
        folder_path = f"{args.folder_prefix}_{rep}"
        model_path = os.path.join(folder_path,
                                    f"best_model_{antimicrobial}_{model}.pkl")
        result_path = os.path.join(folder_path,
                                    f"result_{antimicrobial}_{model}.txt")
        # load optimal threshold
        result_df = pd.read_csv(result_path,
                                header=0,
                                delimiter="\t")
        optimal_threshold = result_df['threshold'][0]

        # load model
        with open(model_path, "rb") as model_fp:
            clf = pickle.load(model_fp)      

        y_proba = clf.predict_proba(X_val)[:, 1]
        y_pred = (y_proba > optimal_threshold).astype('int')

        result_dict = {}
        result_dict['bacc'] = [np.round(balanced_accuracy_score(y_val, y_pred), 2)]
        result_dict['acc'] = [np.round(accuracy_score(y_val, y_pred), 2)]
        result_dict['auc'] = [np.round(roc_auc_score(y_val, y_proba), 2)]
        result_dict['f1'] = [np.round(f1_score(y_val, y_pred), 2)]
        result_dict['f1_macro'] = [np.round(f1_score(y_val, y_pred, average='macro'), 2)]
        result_dict['precision'] = [np.round(precision_score(y_val, y_pred), 2)]
        result_dict['recall'] = [np.round(recall_score(y_val, y_pred), 2)]
        result_dict['logloss'] = [np.round(log_loss(y_val, y_proba), 2)]
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
        result_dict['tn'] = [tn]
        result_dict['fp'] = [fp]
        result_dict['fn'] = [fn]
        result_dict['tp'] = [tp]

        result_path = os.path.join(config["output_dir"], 
                                f"result_{antimicrobial}_{model}_rep_{rep}.txt")

        pd.DataFrame(result_dict).to_csv(result_path, index=False, sep='\t')

        result_df = pd.DataFrame()
        result_df['y_val'] = y_val
        result_df['y_proba'] = y_proba
        result_df['y_pred'] = y_pred
        result_df.to_csv(os.path.join(config["output_dir"], f"result_{antimicrobial}_{model}_rep_{rep}.csv"), index=False)

if __name__ == "__main__":
    main()