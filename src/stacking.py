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
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, log_loss, confusion_matrix


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
        return LogisticRegression(max_iter=1000, random_state=seed)
    elif model_name == "RF":
        return RandomForestClassifier(n_estimators=500, random_state=seed)
    elif model_name == "SVM":
        return SVC(probability=True, random_state=seed)
    elif model_name == "XGB":
        return XGBClassifier(n_estimators=500, random_state=seed)
    elif model_name == "MLP":
        return MLPClassifier(hidden_layer_sizes=[128, 256],
                             max_iter=1000, random_state=seed)


def generate_clfs(seed=507):
    lr_clf = LogisticRegression(max_iter=1000, random_state=seed)
    rf_clf = RandomForestClassifier(n_estimators=500, random_state=seed)
    svm_clf = SVC(probability=True, random_state=seed)
    xgb_clf = XGBClassifier(n_estimators=500, random_state=seed)
    mlp_clf = MLPClassifier(hidden_layer_sizes=[128, 256],
                            max_iter=500, random_state=seed)

    return lr_clf, rf_clf, svm_clf, xgb_clf, mlp_clf


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

    # print(f"{X_train.shape} - {y_train.shape}")

    # Loading model
    # best_models = pd.read_csv(args.best_models)

    # mask = (best_models['antimicrobial'] == args.antimicrobial) & (
    #     best_models['run_index'] == args.run_index)
    # best_run_models = best_models.copy()[mask]
    # spectra_model_name = best_run_models['spectra'].item()
    # vit_latent_model_name = best_run_models['vit_latent'].item()

    # spectra_clf = generate_clf(spectra_model_name)
    # vit_latent_clf = generate_clf(vit_latent_model_name)
    meta_clf = generate_clf(args.meta_clf, seed=222)

    # Creating Stacking classifier
    # Credit to https://stackoverflow.com/questions/70596576/how-to-use-different-feature-matrices-for-sklearn-ensemble-stackingclassifier-w/70599700#70599700

    if args.all_models:
        print("Using all models")
        spectra_lr_clf, spectra_rf_clf, spectra_svm_clf, spectra_xgb_clf, spectra_mlp_clf = generate_clfs()
        vit_latent_lr_clf, vit_latent_rf_clf, vit_latent_svm_clf, vit_latent_xgb_clf, vit_latent_mlp_clf = generate_clfs()

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
        pipe_spectra_xgb = Pipeline([
            ('select', ColumnTransformer(
                [('sel', 'passthrough', spectra_feat_cols)], remainder='drop')),
            ('clf', spectra_xgb_clf)
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
        pipe_vit_latent_xgb = Pipeline([
            ('select', ColumnTransformer(
                [('sel', 'passthrough', vit_latent_feat_cols)], remainder='drop')),
            ('clf', vit_latent_xgb_clf)
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
            ('spectra_xgb', pipe_spectra_xgb),
            ('spectra_mlp', pipe_spectra_mlp),
            ('vit_latent_lr', pipe_vit_latent_lr),
            ('vit_latent_rf', pipe_vit_latent_rf),
            ('vit_latent_svm', pipe_vit_latent_svm),
            ('vit_latent_xgb', pipe_vit_latent_xgb),
            ('vit_latent_mlp', pipe_vit_latent_mlp)],
            final_estimator=meta_clf,
            cv=10)

    # else:
    #     pipe_spectra = Pipeline([
    #         ('select', ColumnTransformer(
    #             [('sel', 'passthrough', spectra_feat_cols)], remainder='drop')),
    #         ('clf', spectra_clf)
    #     ])

    #     pipe_vit_latent = Pipeline([
    #         ('select', ColumnTransformer(
    #             [('sel', 'passthrough', vit_latent_feat_cols)], remainder='drop')),
    #         ('clf', vit_latent_clf)
    #     ])

    #     stack = StackingClassifier(
    #         estimators=[
    #             ('spectra', pipe_spectra),
    #             ('vit_latent', pipe_vit_latent)
    #         ],
    #         final_estimator=meta_clf,
    #         cv=10
    #     )

    if args.combined_feats:
        combine_lr_clf, combine_rf_clf, combine_svm_clf, combine_xgb_clf, combine_mlp_clf = generate_clfs()
        combine_knn_clf = KNeighborsClassifier(metric='cosine')

        stack = StackingClassifier(
            estimators=[
                ('lr', combine_lr_clf),
                ('rf', combine_rf_clf),
                ('svm', combine_svm_clf),
                ('mlp', combine_mlp_clf),
                ('xgb', combine_xgb_clf),
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

    outdir = f"{args.outdir}_meta_{args.meta_clf}"
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    folder_path = os.path.join(outdir,
                               f"{args.antimicrobial}_rep_{args.run_index}")

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

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
