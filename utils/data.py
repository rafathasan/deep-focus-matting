import pathlib
import sys
ROOT = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT)

import yaml
import os
import pandas
from utils.download import Downloader
CONFIG_FILE = "config/data.yaml"


from abc import ABC, abstractmethod

 
class Data_(ABC):
    def __init__(self, config_path=CONFIG_FILE,  dataset_name=None):
        self.config_path = config_path
        self.dataset_name = dataset_name

    @property
    def config_dict(self):
        with open(self.config_path, "r") as file:
            config = yaml.safe_load(file)
        return config

    @property
    def root_path(self):
        return os.path.abspath(self.config_dict['root'])

    @property
    def data_path(self):
        return os.path.abspath(
            os.path.join(
                self.config_dict['root'],
                self.dataset_name
            )
        )

    @property
    def data_keys(self):
        return list(self.config_dict['data'].keys())

    @property
    def dir_list(self):
        return self.config_dict['data'][self.dataset_name]['dir']
    
    @abstractmethod
    def load(self):
        pass
    
    @abstractmethod
    def dataframe(self):
        pass



class Data(Data_):
    def __init__(self, dataset_name):
        super().__init__(dataset_name=dataset_name)

    def load(self):
            data_info = self.config_dict['data'][self.dataset_name]
            Downloader(data_info["url"], data_info['checksum'], data_info['filename'], self.data_path).fetch()
    
    def dataframe(self):

        dir_dict = {}

        for dir in self.dir_list:
            dir_dict[dir] = os.path.abspath(os.path.join(self.data_path, dir))

        for dir in list(dir_dict.keys()):
            dir_dict[dir] = [os.path.join(dir_dict[dir], name) for name in os.listdir(dir_dict[dir])]

        df = pandas.DataFrame(
            zip(*[dir_dict[dir] for dir in list(dir_dict.keys())]),
            columns=list(dir_dict.keys()),
        )
        return df

if __name__== "__main__":
    data = Data()
    data.load()
    l = data.dataframe()
    print(l)