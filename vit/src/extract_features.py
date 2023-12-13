from utils import MALDIDataset
from models import MultiClassClassifier, BinaryClassifier
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from dynaconf import loaders, Dynaconf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import time
import argparse
import os
import sys
sys.path.append("../.")

# from config import settings


def parse_arguments():
    parser = argparse.ArgumentParser("Fine-tuning for AMR prediction")
    parser.add_argument("checkpoint_path",
                        type=str,
                        help="Checkpoint path for pretrained model")
    parser.add_argument("pretrain_config_path",
                        type=str,
                        help="Config path to load pretrained model")
    parser.add_argument("extract_config_path",
                        type=str,
                        help="Config path for finetuning task")
    return parser.parse_args()


def main():
    args = parse_arguments()
    torch.manual_seed(222)  # For resuming training

    pretrain_settings = Dynaconf(
        envvar_prefix="DYNACONF",
        settings_files=[args.pretrain_config_path]
    )

    extract_settings = Dynaconf(
        envvar_prefix="DYNACONF",
        settings_files=[args.extract_config_path]
    )

    # Saving settings dynaconf file for later use
    loaders.write(os.path.join(extract_settings.outdir,
                               f"extract_settings.toml"),
                  extract_settings.as_dict())

    # Load dataset and dataloader
    dataset = MALDIDataset(extract_settings.input_path,
                           extract_settings.label_path,
                           extract_settings.id_path,
                           extract_settings.normalising_factor,
                           extract_settings.model,
                           extract_settings.label_col,
                           get_ids=True)

    dataloader = DataLoader(dataset,
                            num_workers=extract_settings.num_workers, batch_size=extract_settings.batch_size)

    # Initialise model
    model = BinaryClassifier(
        extract_settings,
        pretrain_settings,
        args.checkpoint_path
    ).get_transformer()

    representation_dfs = []

    for _, (X, y, ids) in enumerate(dataloader):
        X_transformed = model(X).cpu().numpy()
        tmp_df = pd.DataFrame(X_transformed)
        tmp_df['id'] = ids
        representation_dfs.append(tmp_df)

    output_fp = os.path.join(extract_settings.outdir,
                             f"extracted_data_{extract_settings.suffix}.csv")

    pd.concat(representation_dfs,
              axis=0,
              ignore_index=True).to_csv(output_fp, index=False)

    # Min-Max scaling
    output_fp = os.path.join(extract_settings.outdir,
                             f"scaled_extracted_data_{extract_settings.suffix}.csv")
    scaler = MinMaxScaler()
    representation_df = pd.concat(representation_dfs,
                                  axis=0,
                                  ignore_index=True)\
        .set_index(keys=['id'])
    scaled_representation_df = pd.DataFrame(scaler.fit_transform(
        representation_df), index=representation_df.index).reset_index(drop=False)
    scaled_representation_df.to_csv(output_fp, index=False)

    return


if __name__ == "__main__":
    main()
