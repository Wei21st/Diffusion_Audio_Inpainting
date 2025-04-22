import torch
import torchaudio
from lightning.pytorch import Trainer as LTrainer
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from omegaconf import OmegaConf

from networks.unet_cqt_oct_with_projattention_adaLN_2 import Unet_CQT_oct_with_attention
from diff_params.edm import EDM
from training.trainer import DiffusionLightningModule, AudioDataModule
from utils.training_utils import compute_LSD

class LossThresholdSaver(Callback):
    def __init__(self, checkpoint_callback, threshold=0.25):
        super().__init__()
        self.checkpoint_callback = checkpoint_callback
        self.threshold = threshold

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs["loss"] if isinstance(outputs, dict) and "loss" in outputs else outputs

        # Some safety checks
        if loss is not None and isinstance(loss, torch.Tensor):
            loss_val = loss.item()
            if loss_val < self.threshold:
                self.checkpoint_callback.on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
    
class LSDLogger(Callback):
    def __init__(self, log_every_n_steps=10, eval_batch_size=16):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.eval_batch_size = eval_batch_size
        self.buffer = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        self.buffer.append(batch)
        total = sum([b.shape[0] for b in self.buffer])
        if total < self.eval_batch_size:
            # print("current total: ", total)
            return

        audio = torch.cat(self.buffer, dim=0)[:self.eval_batch_size]
        self.buffer = []
        with torch.no_grad():
            audio = audio.to(pl_module.device).float()
            if pl_module.args.exp.resample_factor != 1:
                audio = torchaudio.functional.resample(audio, pl_module.args.exp.resample_factor, 1)

            input, target, cnoise = pl_module.diff_params.prepare_train_preconditioning(
                audio, pl_module.diff_params.sample_ptrain_safe(audio.shape[0]).unsqueeze(-1).to(audio.device)
            )
            estimate = pl_module.network(input, cnoise)

            lsd = compute_LSD(audio, estimate)
            pl_module.log("train_lsd", lsd, on_step=True, on_epoch=False, prog_bar=False)

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

    last_checkpoint = ModelCheckpoint(
        dirpath=args.model_dir,
        filename="last",
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

    loss_filter_cb = LossThresholdSaver(threshold_checkpoint, threshold=0.01)
    lsd_logger_cb = LSDLogger(log_every_n_steps=10)


    trainer = LTrainer(
        max_steps=args.exp.total_iters,
        logger=[csv_logger, tb_logger],
        callbacks=[last_checkpoint, loss_filter_cb, lsd_logger_cb],
        log_every_n_steps=args.logging.log_interval,
        precision="32-true",
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        accumulate_grad_batches=2,
    )

    trainer.fit(lightning_model, datamodule=datamodule)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()

