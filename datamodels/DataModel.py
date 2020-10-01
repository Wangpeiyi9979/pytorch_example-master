import os
import numpy as np
from torch.utils.data import Dataset
from texttable import Texttable
from tqdm import tqdm

class DataModel(Dataset):
    def __init__(self, npy_path=None):
        self.x = self.load_data(npy_path)

    def __getitem__(self, idx):
        assert idx < len(self.x)
        return self.x[idx]

    def __len__(self):
        return len(self.x)

    def load_data(self, npy_path):
        data = np.load(npy_path, allow_pickle=True)
        return list(zip(data[0], data[1], data[2], data[3], data[4], data[5], data[6]))

class DataProcessor(object):
    def __init__(self, data_dir, vocab_path, label_path):
        self.data_dir = data_dir
        self.word2id = self._create_word_dict(vocab_path)
        self.label2id = self._create_label_dict(label_path)

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), 'train')

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), 'dev')

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), 'test')

    def _create_label_dict(self, label_path):
        """See base class."""
        pass

    def _create_examples(self, datas, case='train'):
        """Creates examples for the training and dev sets."""
        for data in tqdm(datas):
            pass
        # np.save(os.path.join(self.data_dir, case+'.npy'))

    def _create_single_example(self, data, max_len):
        pass

if __name__ == '__main__':
    data_processor = DataProcessor(data_dir='dataset/xxx',
                                  vocab_path='/home/wpy/data/word2vec/bert-base-cased/vocab.txt',
                                  label_path='your path to label path')
    data_processor.get_train_examples('train raw path')
    data_processor.get_dev_examples('dev raw path')
    data_processor.get_test_examples('test raw path')
