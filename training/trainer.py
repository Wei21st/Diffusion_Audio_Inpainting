import torch
import lightning as L
from torch.utils.data import DataLoader
from datasets.audiofolder import AudioFolderDataset
import torchaudio

class DiffusionLightningModule(L.LightningModule):
    def __init__(self, args, network, diff_params):
        super().__init__()
        self.args = args
        self.network = network
        self.diff_params = diff_params
        self.save_hyperparameters(ignore=["network", "diff_params"])

    def forward(self, audio):
        return self.network(audio)

    def training_step(self, batch, batch_idx):
        audio = batch.to(self.device).float()

        # Resample if needed
        if self.args.exp.resample_factor != 1:
            audio = torchaudio.functional.resample(audio, self.args.exp.resample_factor, 1)

        error, sigma = self.diff_params.loss_fn(self.network, audio)
        loss = error.mean()

        self.log("train_loss_step", loss.item(), on_step=True, on_epoch=False, prog_bar=True)
        # print('train_loss_step: ', loss.item())
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", current_lr, on_step=True, on_epoch=False, prog_bar=False)
        return loss

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(
    #         self.network.parameters(),
    #         lr=self.args.exp.lr,
    #         betas=(self.args.exp.optimizer.beta1, self.args.exp.optimizer.beta2),
    #         eps=self.args.exp.optimizer.eps
    #     )

    #     if hasattr(self.args.exp, 'lr_scheduler') and self.args.exp.lr_scheduler:
    #         scheduler = torch.optim.lr_scheduler.LambdaLR(
    #             optimizer,
    #             lr_lambda=lambda step: min(step / max(self.args.exp.lr_rampup_it, 1e-8), 1.0)
    #         )
    #         return [optimizer], [scheduler]

    #     return optimizer

    def configure_optimizers(self):
      optimizer = torch.optim.Adam(
          self.network.parameters(),
          lr=self.args.exp.lr,
          betas=(self.args.exp.optimizer.beta1, self.args.exp.optimizer.beta2),
          eps=self.args.exp.optimizer.eps
      )

      def warmup_lr_lambda(current_step):
          warmup_steps = self.args.exp.lr_rampup_it
          if current_step < warmup_steps:
              return float(current_step) / float(max(1, warmup_steps))
          return 1.0  # full lr

      scheduler = {
          "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr_lambda),
          "interval": "step",  # step-wise update
          "frequency": 1,
      }

      return {"optimizer": optimizer, "lr_scheduler": scheduler}



class AudioDataModule(L.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def setup(self, stage=None):
        self.train_dataset = AudioFolderDataset(
            dset_args=self.args.dset,
            fs=self.args.exp.sample_rate * self.args.exp.resample_factor,
            seg_len=self.args.exp.audio_len * self.args.exp.resample_factor,
            overfit=False
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.exp.batch,
            num_workers=4,
            pin_memory=True
        )
