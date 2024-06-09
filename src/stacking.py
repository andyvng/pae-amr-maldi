import os
import argparse
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, log_loss, confusion_matrix

from train import tune_model

def parse_args():
    parser = argparse.ArgumentParser(description="Stacking classification")
    parser.add_argument("input_data",
                        type=str,
                        help="Feature input")
    parser.add_argument("label_path",
                        type=str,
                        help="Label input")
    parser.add_argument("antimicrobial",
                        type=str)
    parser.add_argument("run_index",
                        type=int)
    parser.add_argument("train_ids",
                        type=str)
    parser.add_argument("test_ids",
                        type=str)
    parser.add_argument("outdir",
                        type=str)
    parser.add_argument("--all_models",
                        type=int,
                        default=0)
    parser.add_argument("--best_models",
                        type=str,
                        help="Best model for each data type and run")
    parser.add_argument("--voting",
                        type=int,
                        default=0)
    parser.add_argument("--meta_clf",
                        type=str,
                        default="SVM")
    parser.add_argument("--combined_feats",
                        type=int,
                        default=0)

    return parser.parse_args()


def generate_clf(model_name, seed=222):
    if model_name == "LR":
        return LogisticRegression(max_iter=1000, 
                                  class_weight='balanced',
                                  random_state=seed)
    elif model_name == "RF":
        return RandomForestClassifier(n_estimators=500,
                                      class_weight='balanced',
                                      n_jobs=-1,
                                      random_state=seed)
    elif model_name == "SVM":
        return SVC(probability=True,
                   class_weight='balanced', 
                   random_state=seed)
    elif model_name == "LGB":
        return LGBMClassifier(n_estimators=500,
                              n_jobs=-1,
                              class_weight='balanced', 
                              random_state=seed)
    elif model_name == "MLP":
        return MLPClassifier(hidden_layer_sizes=[128, 256],
                             max_iter=1000, random_state=seed)


def generate_clfs(X_train, 
                  y_train,
                  antimicrobial,
                  output_dir,
                  col_prefix="spectra", 
                  seed=507):
    tmp_cols = [col for col in list(X_train.columns) if col.startswith(col_prefix)]
    tmp_X_train = X_train.copy()[tmp_cols]

    clfs =  []

    for model in ["LR", "RF", "SVM", "LGB", "MLP"]:
        print(f"Training {model} model - {col_prefix}")
        clf, _ = tune_model(tmp_X_train,
                            y_train,
                            model,
                            antimicrobial,
                            output_dir,
                            save_model=False)
        clfs.append(clf)
    
    return clfs


def determine_optimal_f1_threshold(y_true, y_proba):
    max_f1 = 0
    optimal_threshold = 0.5
    thresholds = np.linspace(min(y_proba), max(y_proba), 1000)

    for threshold in thresholds:
        tmp_pred = (y_proba > threshold).astype(int)
        tmp_f1 = f1_score(y_true, tmp_pred)
        if tmp_f1 > max_f1:
            max_f1 = tmp_f1
            optimal_threshold = threshold

    return optimal_threshold


def main():
    args = parse_args()

    print(f"{args.antimicrobial}-{args.run_index}")

    # Loading data
    X = pd.read_csv(args.input_data).set_index(keys=['id'])
    y = pd.read_csv(args.label_path)\
        .set_index(keys=['id'])[[args.antimicrobial]]

    spectra_feat_cols = [col for col in list(
        X.columns) if col.startswith("spectra")]
    vit_latent_feat_cols = [col for col in list(
        X.columns) if col.startswith("vit_latent")]

    train_ids = pd.read_csv(args.train_ids, header=None, names=['id'])
    test_ids = pd.read_csv(args.test_ids, header=None, names=['id'])

    X_train = train_ids.merge(X, right_index=True, left_on='id', how='left')\
        .set_index(keys='id')
    y_train = train_ids.merge(y, right_index=True, left_on='id', how='left')\
        .set_index(keys='id')
    X_test = test_ids.merge(X, right_index=True, left_on='id', how='left')\
        .set_index(keys='id')
    y_test = test_ids.merge(y, right_index=True, left_on='id', how='left')\
        .set_index(keys='id')

    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    meta_clf = generate_clf(args.meta_clf, seed=222)

    # Creating Stacking classifier
    # Credit to https://stackoverflow.com/questions/70596576/how-to-use-different-feature-matrices-for-sklearn-ensemble-stackingclassifier-w/70599700#70599700

    if args.all_models:
        print("Using all models")
        spectra_clfs = generate_clfs(X_train, 
                                     y_train,
                                     args.antimicrobial,
                                     "", col_prefix="spectra")
        spectra_lr_clf, spectra_rf_clf, spectra_svm_clf, spectra_lgb_clf, spectra_mlp_clf  = spectra_clfs
        vit_latent_clfs = generate_clfs(X_train, 
                                        y_train,
                                        args.antimicrobial,
                                        "", col_prefix="vit_latent")
        vit_latent_lr_clf, vit_latent_rf_clf, vit_latent_svm_clf, vit_latent_lgb_clf, vit_latent_mlp_clf = vit_latent_clfs

        pipe_spectra_lr = Pipeline([
            ('select', ColumnTransformer(
                [('sel', 'passthrough', spectra_feat_cols)], remainder='drop')),
            ('clf', spectra_lr_clf)
        ])

        pipe_spectra_rf = Pipeline([
            ('select', ColumnTransformer(
                [('sel', 'passthrough', spectra_feat_cols)], remainder='drop')),
            ('clf', spectra_rf_clf)
        ])
        pipe_spectra_svm = Pipeline([
            ('select', ColumnTransformer(
                [('sel', 'passthrough', spectra_feat_cols)], remainder='drop')),
            ('clf', spectra_svm_clf)
        ])
        pipe_spectra_lgb = Pipeline([
            ('select', ColumnTransformer(
                [('sel', 'passthrough', spectra_feat_cols)], remainder='drop')),
            ('clf', spectra_lgb_clf)
        ])
        pipe_spectra_mlp = Pipeline([
            ('select', ColumnTransformer(
                [('sel', 'passthrough', spectra_feat_cols)], remainder='drop')),
            ('clf', spectra_mlp_clf)
        ])

        pipe_vit_latent_lr = Pipeline([
            ('select', ColumnTransformer(
                [('sel', 'passthrough', vit_latent_feat_cols)], remainder='drop')),
            ('clf', vit_latent_lr_clf)
        ])
        pipe_vit_latent_rf = Pipeline([
            ('select', ColumnTransformer(
                [('sel', 'passthrough', vit_latent_feat_cols)], remainder='drop')),
            ('clf', vit_latent_rf_clf)
        ])
        pipe_vit_latent_svm = Pipeline([
            ('select', ColumnTransformer(
                [('sel', 'passthrough', vit_latent_feat_cols)], remainder='drop')),
            ('clf', vit_latent_svm_clf)
        ])
        pipe_vit_latent_lgb = Pipeline([
            ('select', ColumnTransformer(
                [('sel', 'passthrough', vit_latent_feat_cols)], remainder='drop')),
            ('clf', vit_latent_lgb_clf)
        ])
        pipe_vit_latent_mlp = Pipeline([
            ('select', ColumnTransformer(
                [('sel', 'passthrough', vit_latent_feat_cols)], remainder='drop')),
            ('clf', vit_latent_mlp_clf)
        ])

        stack = StackingClassifier(estimators=[
            ('spectra_lr', pipe_spectra_lr),
            ('spectra_rf', pipe_spectra_rf),
            ('spectra_svm', pipe_spectra_svm),
            ('spectra_lgb', pipe_spectra_lgb),
            ('spectra_mlp', pipe_spectra_mlp),
            ('vit_latent_lr', pipe_vit_latent_lr),
            ('vit_latent_rf', pipe_vit_latent_rf),
            ('vit_latent_svm', pipe_vit_latent_svm),
            ('vit_latent_lgb', pipe_vit_latent_lgb),
            ('vit_latent_mlp', pipe_vit_latent_mlp)],
            final_estimator=meta_clf,
            cv=10)

    if args.combined_feats:
        combine_lr_clf, combine_rf_clf, combine_svm_clf, combine_lgb_clf, combine_mlp_clf = generate_clfs()
        combine_knn_clf = KNeighborsClassifier(metric='cosine')

        stack = StackingClassifier(
            estimators=[
                ('lr', combine_lr_clf),
                ('rf', combine_rf_clf),
                ('svm', combine_svm_clf),
                ('mlp', combine_mlp_clf),
                ('lgb', combine_lgb_clf),
                ('knn', combine_knn_clf)],
            final_estimator=meta_clf,
            cv=10)

    stack.fit(X_train, y_train)

    y_proba = stack.predict_proba(X_test)[:, 1]

    # Determine optimal threshold for F1 score
    optimal_threshold = determine_optimal_f1_threshold(y_test, y_proba)
    y_pred = (y_proba > optimal_threshold).astype(int)

    bacc = np.round(balanced_accuracy_score(y_test, y_pred), 2)
    acc = np.round(accuracy_score(y_test, y_pred), 2)
    auc = np.round(roc_auc_score(y_test, y_proba), 2)
    f1 = np.round(f1_score(y_test, y_pred), 2)
    precision = np.round(precision_score(y_test, y_pred), 2)
    recall = np.round(recall_score(y_test, y_pred), 2)
    ll = np.round(log_loss(y_test, y_proba), 2)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()


    folder_path = os.path.join(args.outdir,
                               f"meta_{args.meta_clf}_{args.antimicrobial}_rep_{args.run_index}")
    
    os.makedirs(folder_path, exist_ok=True)


    result_path = os.path.join(folder_path, "result.txt")

    with open(result_path, 'w') as result_output:
        result_output.write(
            f"bacc\tacc\tauroc\tf1\tprecision\trecall\tll\ttn\tfn\ttp\tfp\tthreshold\n")
        result_output.write(
            f"{bacc}\t{acc}\t{auc}\t{f1}\t{precision}\t{recall}\t{ll}\t{tn}\t{fn}\t{tp}\t{fp}\t{optimal_threshold}")

    result_df = pd.DataFrame()
    result_df['y_test'] = y_test
    result_df['y_proba'] = y_proba
    result_df['y_pred'] = y_pred
    result_df.to_csv(os.path.join(folder_path, "result.csv"), index=False)


if __name__ == "__main__":
    main()
