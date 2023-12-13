import os
import pandas as pd
import numpy as np
import optuna
import pickle
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.utils import class_weight


from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, log_loss

from utils import load_and_preprocess_spectra


def objective(trial, X, y, model_name, antimicrobial, output_dir):

    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y
    )

    if model_name == 'LR':
        params = {
            'tol': trial.suggest_float('tol', 1e-6, 1e-3, log=True),
            'C': trial.suggest_float('C', 1e-2, 1, log=True),
            'max_iter': trial.suggest_int('max_iter', 500, 1000, 100),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
        }
        model = LogisticRegression(**params, random_state=222)
    elif model_name == 'RF':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, 100),
            'max_depth': trial.suggest_int('max_depth', 1, 29, 2),
            'max_features': trial.suggest_int('max_features', 20, 800, 20),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced', 'balanced_subsample']),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
        }
        model = RandomForestClassifier(**params, random_state=222, n_jobs=-1)
    elif model_name == 'SVM':
        params = {
            'C': trial.suggest_float('C', 1e-5, 1e5, log=True),
            'kernel': trial.suggest_categorical("kernel", ["linear", "poly", "rbf"]),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
        }
        model = SVC(**params, probability=True, random_state=222)
    elif model_name == 'XGB':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, 100),
            "max_depth": trial.suggest_int("max_depth", 1, 9),
            "eta": trial.suggest_float("eta", 1e-8, 1.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
        }
        model = XGBClassifier(**params,
                              random_state=222)
    elif model_name == 'MLP':
        params = {
            "solver": trial.suggest_categorical("solver", ["sgd", "adam"]),
            "momentum": trial.suggest_float("momentum", 0.0, 1.0),
            'max_iter': trial.suggest_int('max_iter', 200, 1000, 100),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-5, 1e-3, log=True),
            "activation": trial.suggest_categorical("activation", ["tanh", "relu"]),
            "power_t": trial.suggest_float("power_t", 0.2, 0.8, step=0.1)
        }
        model = MLPClassifier(
            **params,
            hidden_layer_sizes=(256, 128, 64),
            random_state=222
        )

    # Get time seed to randomly generate 5 folds
    curr_dt = datetime.now()
    time_seed = int(curr_dt.timestamp())
    strat_k_fold = StratifiedKFold(n_splits=5,
                                   shuffle=True,
                                   random_state=time_seed)

    auc_scores = cross_val_score(model,
                                 X,
                                 y,
                                 scoring='roc_auc',
                                 cv=strat_k_fold)

    aucs_fp = os.path.join(
        output_dir, f"tuning_aucs_{antimicrobial}_{model_name}.txt")

    if os.path.exists(aucs_fp):
        aucs = np.loadtxt(aucs_fp)
        aucs = np.concatenate([aucs, auc_scores])
        np.savetxt(aucs_fp, aucs)

    else:
        np.savetxt(aucs_fp, auc_scores)

    return auc_scores.mean()


def tune_model(X_train,
               y_train,
               model,
               antimicrobial,
               output_dir,
               n_trials=100,
               random_state=222):

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if model != 'Ensemble':
        study = optuna.create_study(direction='maximize')

        def func(trial): return objective(trial,
                                          X_train,
                                          y_train,
                                          model,
                                          antimicrobial,
                                          output_dir)
        study.optimize(func, n_trials=n_trials)
        print(f"The best parameters are : {study.best_params}")

        # Saving best params
        with open(os.path.join(output_dir, f"best_params_{antimicrobial}_{model}.pkl"), 'wb') as best_params_outupt:
            pickle.dump(study.best_params, best_params_outupt)

        if model == 'LR':
            best_model = LogisticRegression(**study.best_params,
                                            random_state=random_state)
        elif model == 'RF':
            best_model = RandomForestClassifier(**study.best_params,
                                                random_state=random_state,
                                                n_jobs=-1)
        elif model == 'SVM':
            best_model = SVC(**study.best_params,
                             probability=True,
                             random_state=random_state)
        elif model == 'XGB':
            best_model = XGBClassifier(**study.best_params,
                                       random_state=random_state)
        elif model == 'MLP':
            best_model = MLPClassifier(**study.best_params,
                                       random_state=random_state)
        else:
            raise ValueError(f"Invalid model! {model}")
    else:
        raise ValueError(f"Ensemble model not available yet!")

    best_model.fit(X_train, y_train)

    # Saving best model
    with open(os.path.join(output_dir, f"best_model_{antimicrobial}_{model}.pkl"), 'wb') as best_model_output:
        pickle.dump(best_model, best_model_output)

    return best_model
