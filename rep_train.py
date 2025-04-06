import os
import torch
import numpy as np
from omegaconf import OmegaConf

from datasets.audiofolder import AudioFolderDataset
from datasets.audiofolder_test import AudioFolderDatasetTest
from diff_params.edm import EDM
from networks.unet_cqt_oct_with_projattention_adaLN_2 import Unet_CQT_oct_with_attention
from testing.tester import Tester
from training.trainer import Trainer


def worker_init_fn(worker_id):
    st = np.random.get_state()[2]
    np.random.seed(st + worker_id)


class ConfigManager:
    def __init__(self, config_path):
        self.args = OmegaConf.load(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_model_dir()

    def _setup_model_dir(self):
        dirname = os.path.dirname(os.path.abspath("__file__"))
        self.args.model_dir = os.path.join(dirname, str(self.args.model_dir))
        os.makedirs(self.args.model_dir, exist_ok=True)
        print(f"Model directory: {self.args.model_dir}")

class DataLoaderFactory:
    def __init__(self, args):
        self.args = args

    def create_train_loader(self):
        dataset = AudioFolderDataset(
            dset_args=self.args.dset,
            fs=self.args.exp.sample_rate * self.args.exp.resample_factor,
            seg_len=self.args.exp.audio_len * self.args.exp.resample_factor,
            overfit=False
        )
        return iter(torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.args.exp.batch,
            num_workers=0,
            pin_memory=True,
            worker_init_fn=worker_init_fn
        ))

    def create_test_loader(self):
        test_dataset = AudioFolderDatasetTest(
            dset_args=self.args.dset,
            fs=self.args.exp.sample_rate * self.args.exp.resample_factor,
            seg_len=self.args.exp.audio_len * self.args.exp.resample_factor,
            num_samples=self.args.dset.test.num_samples
        )
        return torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=self.args.dset.test.batch_size,
            num_workers=0,
            pin_memory=True,
            worker_init_fn=worker_init_fn
        )


class TrainingApp:
    def __init__(self, config_path):
        self.config_mgr = ConfigManager(config_path)
        self.args = self.config_mgr.args
        self.device = self.config_mgr.device

        self.train_loader = DataLoaderFactory(self.args).create_train_loader()
        self.test_loader = DataLoaderFactory(self.args).create_test_loader()

        self.diff_params = EDM(self.args)
        self.network = Unet_CQT_oct_with_attention(self.args, self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.args.exp.lr,
            betas=(self.args.exp.optimizer.beta1, self.args.exp.optimizer.beta2),
            eps=self.args.exp.optimizer.eps
        )

        self.tester = Tester(
            args=self.args,
            network=self.network,
            test_set=self.test_loader,
            diff_params=self.diff_params,
            device=self.device
        )

        self.trainer = Trainer(
            args=self.args,
            dset=self.train_loader,
            network=self.network,
            optimizer=self.optimizer,
            diff_params=self.diff_params,
            tester=self.tester,
            device=self.device
        )

    def run(self):
        print(f"Using device: {self.device}")
        print("Training Started! ")
        self.trainer.training_loop()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    app = TrainingApp("rep_training_config.yaml")
    app.run()