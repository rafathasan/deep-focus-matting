#!/home/rafatmatting/anaconda3/envs/ml/bin/python
import torch
from pytorch_lightning import LightningModule, Trainer
from networks.UNet import UNet
from networks.MODNet import MODNet
from networks.GFM import GFM
from networks.DFM import DFM
from datasets.MattingDataModule import MattingDataModule
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    ModelSummary,
    LearningRateMonitor,
)
import os
import argparse

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Names
    parser.add_argument(
        "--model-type",
        type=str,
        default="UNet",
        choices=["UNet", "MODNet", "DFM", "GFM"],
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="AMD",
        choices=["AMD", "PPM-100"],
    )

    # Files
    parser.add_argument(
        "--dataset-config-file-path",
        default="config/datasets.yaml",
        type=str,
    )
    parser.add_argument(
        "--training-config-file-path",
        default="config/train_params.yaml",
        type=str,
    )
    parser.add_argument(
        "--log-folder",
        default="./logs",
        type=str,
    )
    parser.add_argument(
        "--resume-from-checkpoint",
         type=str,
         default=None,
         )
    # Machine Params
    parser.add_argument(
        "--gpu-indices",
        type=str,
        default="0",
        help="""A comma separated list of GPU indices. Set as value of 
        CUDA_VISIBLE_DEVICES environment variable""",
    )
    parser.add_argument("--num-workers", type=int, default=4)

    # Hyperparams
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.1)
    parser.add_argument("--lr-scheduler-patience", type=int, default=50)
    
    return parser

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_indices

    model_type = args.model_type
    dataset_name = args.dataset_name
    dataset_config = args.dataset_config_file_path
    training_config = args.training_config_file_path
    epochs = args.epochs
    num_workers = args.num_workers
    resume_from_checkpoint  = args.resume_from_checkpoint


    # Network 
    if model_type == "MODNet":
        network = MODNet(args.learning_rate,args.lr_scheduler_factor, args.lr_scheduler_patience)
    elif model_type == "UNet":
        network = UNet(args.learning_rate,args.lr_scheduler_factor, args.lr_scheduler_patience)
    elif model_type == "GFM":
        network = GFM(args.learning_rate,args.lr_scheduler_factor, args.lr_scheduler_patience)
    elif model_type == "DFM":
        network = DFM(args.learning_rate,args.lr_scheduler_factor, args.lr_scheduler_patience)
    else:
        raise Exception("model_type not given")

    # Dataset
    data_module = MattingDataModule()

    data_module.prepare_data()

    from pytorch_lightning.loggers import TensorBoardLogger

    experiment_name = f"{args.model_type}_{args.dataset_name}"
    version_name = f"epochs:{args.epochs}_lr:{args.learning_rate}_factor:{args.lr_scheduler_factor}_patience:{args.lr_scheduler_patience}"
    logger = TensorBoardLogger(args.log_folder, name=experiment_name, version=version_name)

    checkpoint_path = os.path.join(args.log_folder, experiment_name, version_name, "checkpoints")

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_path,
            every_n_epochs=1,
            monitor="train_loss",
            save_last=True,
        ),
    ]


    from pytorch_lightning.plugins import DDPPlugin

    trainer = Trainer(
        logger=logger,
        gpus=torch.cuda.device_count(),
        # strategy=DDPPlugin(find_unused_parameters=False),
        strategy=DDPPlugin(),
        callbacks=callbacks,
        max_epochs=args.epochs,
        # auto_lr_find=True,
        # auto_scale_batch_size=True,
        # overfit_batches=10,
        # fast_dev_run=1,
    )

    # trainer.tune(network, datamodule=data_module)
    trainer.fit(network, datamodule=data_module, ckpt_path=None)
