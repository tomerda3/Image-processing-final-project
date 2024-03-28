import pandas as pd


class LabelSplitter:
    def __init__(self, directory_path: str, set_name: str = 'Set',
                 train_id: str = "train", test_id: str = "test", val_id: str = "val"):
        self.directory_path = directory_path
        df = pd.read_csv(self.directory_path)

        self._train_df = df[df[set_name] == train_id].drop(set_name, axis=1)
        self._test_df = df[df[set_name] == test_id].drop(set_name, axis=1)
        self._val_df = df[df[set_name] == val_id].drop(set_name, axis=1)

    @property
    def train(self):
        return self._train_df

    @property
    def test(self):
        return self._test_df

    @property
    def val(self):
        return self._val_df
