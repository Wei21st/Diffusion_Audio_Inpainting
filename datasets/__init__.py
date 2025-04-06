from .audiofolder import AudioFolderDataset
# from .audiofolder_test import AudioFolderTestDataset
from .librispeech import LibrispeechTrain, LibrispeechTest
from .maestro_dataset import MaestroDataset
from .maestro_dataset_test import MaestroDatasetTestChunks

__all__ = [
    'AudioFolderDataset',
    # 'AudioFolderTestDataset',
    'LibrispeechTrain',
    'LibrispeechTest',
    'MaestroDataset',
    'MaestroDatasetTestChunks'
]
