import torch
from lightning.pytorch import Trainer as LTrainer
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf

from networks.unet_cqt_oct_with_projattention_adaLN_2 import Unet_CQT_oct_with_attention
from diff_params.edm import EDM
from training.trainer import DiffusionLightningModule, AudioDataModule  # the new files we wrote

def main():
    config_path = "rep_training_config.yaml"
    args = OmegaConf.load(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    diff_params = EDM(args)
    network = Unet_CQT_oct_with_attention(args, device)

    lightning_model = DiffusionLightningModule(args, network, diff_params)
    datamodule = AudioDataModule(args)

    # Logger and checkpoint callback
    logger = CSVLogger("logs", name=args.exp.exp_name)
    checkpoint_cb = ModelCheckpoint(
        dirpath=args.model_dir,
        save_top_k=-1,
        every_n_train_steps=args.logging.save_interval,
    )

    trainer = LTrainer(
        max_steps=args.exp.total_iters,
        logger=logger,
        callbacks=[checkpoint_cb],
        log_every_n_steps=args.logging.log_interval,
        precision="16-mixed",
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
    )

    trainer.fit(lightning_model, datamodule=datamodule)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
