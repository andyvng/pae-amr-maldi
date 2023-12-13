import os
import pickle
import json
import argparse
import pandas as pd
import numpy as np

from sklearn.metrics import f1_score

from utils import load_data_for_voting, calculate_metrics

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path",
                        type=str,
                        help='Path to config file')
    parser.add_argument("antimicrobial",
                        type=str,
                        help="Testing antimicrobial")
    
    return parser.parse_args()

def main():
    args = get_args()

    with open(args.config_path, 'r') as config_fp:
        configs = json.load(config_fp)
    
    # Load data
    X_bin_train, X_variant_train, \
        y_train = load_data_for_voting(configs['input_dir'], 
                                       configs['feature_path'], 
                                       configs['ast_path'], 
                                       args.antimicrobial, 
                                       configs['train_id_path'], 
                                       configs['bins_path'])
    
    X_bin_test, X_variant_test, \
        y_test = load_data_for_voting(configs['input_dir'], 
                                       configs['feature_path'], 
                                       configs['ast_path'], 
                                       args.antimicrobial, 
                                       configs['test_id_path'], 
                                       configs['bins_path'])

    # Load model
    # binning model
    with open(configs['binning_model'], 'rb') as bin_model_fp:
        binning_model = pickle.load(bin_model_fp)

    # variant model
    with open(configs['variant_model', 'rb']) as variant_model_fp:
        variant_model = pickle.load(variant_model_fp)


    # Calculate combined predict_proba and determine optimal threshold
    y_bin_proba_train = binning_model.predict_proba(X_bin_train)
    y_variant_proba_train = variant_model.predict_proba(X_variant_train)
    y_proba_train = np.mean(y_bin_proba_train + y_variant_proba_train, dim=1)

    max_f1 = 0
    optimal_threshold = 0

    thresholds = np.linspace(min(y_proba_train), max(y_proba_train), 100)

    for threshold in thresholds:
        tmp_y_pred_train = (y_proba_train > threshold).astype(int)
        tmp_f1 = f1_score(y_train, tmp_y_pred_train)

        if tmp_f1 >= max_f1:
            optimal_threshold = threshold
            max_f1 = tmp_f1


    y_bin_proba_test = binning_model.predict_proba(X_bin_test)
    y_variant_proba_test = variant_model.predict_proba(X_variant_test)
    y_proba_test = np.mean(y_bin_proba_test + y_variant_proba_test, dim=1)
    y_pred_test  = (y_proba_test > optimal_threshold).astype(int)

    # Calculate metrics
    acc, auc, f1,\
    precision, recall, ll,\
    tn, fn, tp, fp = calculate_metrics(y_test, y_proba_test, y_pred_test)



if __name__ == "__main__":
    main()