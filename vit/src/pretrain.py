import os
import sys
sys.path.append("../.")
import argparse
import time
import pandas as pd
import numpy as np
from config import settings
from dynaconf import loaders, Dynaconf

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

from models import MultiClassClassifier
from utils import MALDIDataset
    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Pretrain vision transformer model to identify bacteria")
    parser.add_argument("--config_path",
                        type=str,
                        default="",
                        help="Setting file (.toml) for configuration")
    parser.add_argument("--resume",
                        type=int,
                        default=0,
                        help="If yes resume training from last checkpoint path")
    return parser.parse_args()

def main():

    torch.manual_seed(222) # For resuming training
    
    start_time = time.time()

    args = parse_arguments()
    if args.config_path:
        print("Modify settings file")
        settings = Dynaconf(
            envvar_prefix="DYNACONF",
            settings_files=[args.config_path]
            )
    
    # Saving settings dynaconf file for later use
    loaders.write(os.path.join(settings.outdir, f"settings.toml"),
                  settings.as_dict())

    checkpoint_callback = ModelCheckpoint(
        dirpath=settings.outdir,
        save_top_k=settings.num_ckpts,
        monitor="val_loss",
        save_last=True
    )

    # Load dataset and dataloader
    train_dataset = MALDIDataset(
        settings.input_path,
        settings.label_path,
        settings.train_id_path,
        settings.normalising_factor,
        model=settings.model
        )
    val_dataset = MALDIDataset(
        settings.input_path,
        settings.label_path,
        settings.val_id_path,
        settings.normalising_factor,
        model=settings.model
        )
    test_dataset = MALDIDataset(
        settings.input_path,
        settings.label_path,
        settings.test_id_path,
        settings.normalising_factor,
        model=settings.model
        )

    # input_shape = train_dataset.__getitem__(0)[0].shape[0]
    train_dataloader = DataLoader(train_dataset,
                                  num_workers=settings.num_workers,
                                  batch_size=settings.batch_size,
                                  shuffle=True)
    val_dataloader = DataLoader(val_dataset,
                                num_workers=settings.num_workers,
                                batch_size=settings.batch_size)
    test_dataloader = DataLoader(test_dataset,
                                 num_workers=settings.num_workers,
                                 batch_size=settings.batch_size)
    
    test_dataloaders = [test_dataloader]

    # Load external dataloader
    if settings.external_input_path != "":
        external_dataset = MALDIDataset(settings.external_input_path,
                                        settings.external_label_path,
                                        settings.external_id_path,
                                        settings.normalising_factor,
                                        settings.model,
                                        settings.label_col)
        external_dataloader = DataLoader(external_dataset,
                                        num_workers=settings.num_workers,
                                        batch_size=settings.batch_size)
        test_dataloaders.append(external_dataloader)

    model = MultiClassClassifier(
        settings
    )


    tb_logger = pl_loggers.TensorBoardLogger(save_dir=settings.lightning_logs)

    if torch.cuda.is_available():
        if "A100" in torch.cuda.get_device_name():
            torch.set_float32_matmul_precision("high")
        trainer = pl.Trainer(max_epochs=settings.num_epochs, 
                            accelerator="gpu",
                            logger=tb_logger,
                            callbacks=[checkpoint_callback])
    else:
        trainer = pl.Trainer(max_epochs=settings.num_epochs,
                            logger=tb_logger,
                            callbacks=[checkpoint_callback])
        
    end_initialisation_time = time.time()
    print(f"Initialisation time: {end_initialisation_time - start_time}")

    with open(os.path.join(settings.outdir, 
                           f"running_time.txt"), 'a') as running_fp:
        running_fp.write(f"Initialisation time: {end_initialisation_time - start_time}\n")

    if args.resume:
        trainer.fit(model,
                    train_dataloaders=train_dataloader,
                    val_dataloaders=val_dataloader,
                    ckpt_path="last")
    else:
        trainer.fit(model,
                    train_dataloaders=train_dataloader,
                    val_dataloaders=val_dataloader)
    
    end_training_time = time.time()
    print(f"Training time: {end_training_time - end_initialisation_time}")

    # Testing model
    trainer.test(ckpt_path="best",
                 dataloaders=test_dataloaders
                 )

    with open(os.path.join(settings.outdir, 
                           f"running_time.txt"), 'a') as running_fp:
        running_fp.write(f"Training time: {end_training_time - end_initialisation_time}")

if __name__ == "__main__":
    main()
