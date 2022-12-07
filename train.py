#!/home/rafatmatting/anaconda3/envs/ml/bin/python
from ensurepip import version
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
        choices=["AMD", "AMD_cropped", "PPM-100"],
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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--version", type=str, default="0")
    parser.add_argument("--resume", action='store_true')

    # Hyperparams
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    # parser.add_argument("--lr-scheduler-factor", type=float, default=0.1)
    # parser.add_argument("--lr-scheduler-patience", type=int, default=50)
    
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
    batch_size = args.batch_size
    _version = args.version
    resume = args.resume
    resume_from_checkpoint  = args.resume_from_checkpoint


    """_network_

    Raises:
        Exception
    """
    settings = {
            "learning_rate": args.learning_rate,
            "monitor": "training_loss"
        }

    if model_type == "MODNet":
        network = MODNet(settings)
    elif model_type == "UNet":
        network = UNet(settings)
    elif model_type == "GFM":
        network = GFM(settings)
    elif model_type == "DFM":
        network = DFM(settings)
    else:
        raise Exception("model_type not given")

    """_dataset_
    """
    data_module = MattingDataModule(dataset_name=args.dataset_name, num_workers=args.num_workers, batch_size=args.batch_size)

    data_module.prepare_data()

    from pytorch_lightning.loggers import TensorBoardLogger
    # from pytorch_lightning.loggers import WandbLogger

    experiment_name = f"{args.model_type}_{args.dataset_name}"
    tensorboard_logger = TensorBoardLogger(args.log_folder, name=experiment_name, version=_version)
    # wandb_logger = WandbLogger(project=experiment_name)

    checkpoint_path = os.path.join(args.log_folder, experiment_name, _version, "checkpoints")

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_path,
            filename="best_{epoch}_{training_l2loss:0:.4f}",
            # every_n_epochs=1,
            # every_n_train_steps=1,
            mode="min",
            monitor="training_l2loss",
            save_top_k=-1,
        ),
    ]

    checkpoint_file=None
    if resume:
        checkpoint_file = os.path.join(checkpoint_path,"last.ckpt")

    from pytorch_lightning.plugins import DDPPlugin

    trainer = Trainer(
        logger=tensorboard_logger,
        gpus=1,#torch.cuda.device_count(),
        # gpus=1,
        # accelerator="gpu",
        # strategy=DDPPlugin(find_unused_parameters=False),
        # strategy=DDPPlugin(),
        # callbacks=callbacks,
        max_epochs=args.epochs,
        # auto_lr_find=True,
        # auto_scale_batch_size=True,
        # overfit_batches=10,
        # fast_dev_run=1,
        # log_every_n_steps=50,
        # flush_logs_every_n_steps=100,
        resume_from_checkpoint=checkpoint_file,
        limit_val_batches=0,
        num_sanity_val_steps=0,
    )
    # trainer.tune(network, datamodule=data_module)
    trainer.fit(network, datamodule=data_module, ckpt_path=None)
