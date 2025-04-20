# import torch
# from lightning.pytorch import Trainer as LTrainer
# from lightning.pytorch.loggers import CSVLogger
# from lightning.pytorch.callbacks import ModelCheckpoint
# from lightning.pytorch.callbacks import ModelCheckpoint
# from omegaconf import OmegaConf

# from networks.unet_cqt_oct_with_projattention_adaLN_2 import Unet_CQT_oct_with_attention
# from diff_params.edm import EDM
# from training.trainer import DiffusionLightningModule, AudioDataModule  # the new files we wrote

# def main():
#     config_path = "rep_training_config.yaml"
#     args = OmegaConf.load(config_path)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     diff_params = EDM(args)
#     network = Unet_CQT_oct_with_attention(args, device)

#     lightning_model = DiffusionLightningModule(args, network, diff_params)
#     datamodule = AudioDataModule(args)

#     # Logger and checkpoint callback
#     logger = CSVLogger("logs", name=args.exp.exp_name)
#     checkpoint_cb = ModelCheckpoint(
#         dirpath=args.model_dir,
#         save_top_k=-1,
#         every_n_train_steps=args.logging.save_interval,
#     )

#     trainer = LTrainer(
#         max_steps=args.exp.total_iters,
#         logger=logger,
#         callbacks=[checkpoint_cb],
#         log_every_n_steps=args.logging.log_interval,
#         precision="32-true",
#         accelerator="gpu" if torch.cuda.is_available() else "cpu",
#     )

#     trainer.fit(lightning_model, datamodule=datamodule)


# if __name__ == "__main__":
#     torch.multiprocessing.set_start_method("spawn", force=True)
#     main()

import torch
from lightning.pytorch import Trainer as LTrainer
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from omegaconf import OmegaConf

from networks.unet_cqt_oct_with_projattention_adaLN_2 import Unet_CQT_oct_with_attention
from diff_params.edm import EDM
from training.trainer import DiffusionLightningModule, AudioDataModule  # the new files we wrote

class LossThresholdSaver(Callback):
    def __init__(self, checkpoint_callback, threshold=0.25):
        super().__init__()
        self.checkpoint_callback = checkpoint_callback
        self.threshold = threshold

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Get loss from outputs, if training_step returned it
        loss = outputs["loss"] if isinstance(outputs, dict) and "loss" in outputs else outputs

        # Some safety checks
        if loss is not None and isinstance(loss, torch.Tensor):
            loss_val = loss.item()
            if loss_val < self.threshold:
                self.checkpoint_callback.on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)


def main():
    config_path = "rep_training_config.yaml"
    args = OmegaConf.load(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup model and diffusion params
    diff_params = EDM(args)
    network = Unet_CQT_oct_with_attention(args, device)
    lightning_model = DiffusionLightningModule(args, network, diff_params)
    datamodule = AudioDataModule(args)

    tb_logger = TensorBoardLogger("tb_logs", name=args.exp.exp_name)
    csv_logger = CSVLogger("logs", name=args.exp.exp_name)

    # Checkpoint for best model (lowest loss)
    last_checkpoint = ModelCheckpoint(
        dirpath=args.model_dir,
        filename="last",  # Just call it 'last.ckpt'
        save_top_k=1,
        save_last=True,
        save_on_train_epoch_end=False,
        every_n_train_steps=args.logging.save_interval,
    )

    threshold_checkpoint = ModelCheckpoint(
        dirpath=args.model_dir,
        filename="under_threshold-{step}-{train_loss_step:.4f}",
        monitor="train_loss_step",
        mode="min",
        save_top_k=-1,
        every_n_train_steps=1,
        save_on_train_epoch_end=False,
    )

    loss_filter_cb = LossThresholdSaver(threshold_checkpoint, threshold=0.03)

    trainer = LTrainer(
        max_steps=args.exp.total_iters,
        logger=csv_logger,
        callbacks=[last_checkpoint, loss_filter_cb],
        log_every_n_steps=args.logging.log_interval,
        precision="32-true",
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
    )

    trainer.fit(lightning_model, datamodule=datamodule)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()

