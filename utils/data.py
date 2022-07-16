from utils.download import Downloader
import yaml
import os
import pandas

DATA_CONFIG = "config/data.yaml"


class Data:
    def __init__(self, config_file=DATA_CONFIG, key="alphamatting"):
        self.config_file = config_file
        self.key = key
    
    @property
    def data_config(self):
        with open(self.config_file, "r") as file:
            data_config = yaml.safe_load(file)
        return data_config

    @property
    def data_root(self):
        return os.path.abspath(
            os.path.join(
                self.data_config['root'],
                self.data_config['data'][self.key]['dir']
            )
        )

    @property
    def get_data_keys(self):
        return self.data_config['data'].keys()

    def load(self):
            data_info = self.data_config['data'][self.key]
            Downloader(data_info["url"], data_info['checksum'], data_info['filename'], self.data_root).fetch()
    
    def generate_dataframe(self):

        image_path = os.path.abspath(os.path.join(self.data_root, "images"))
        mask_path = os.path.abspath(os.path.join(self.data_root, "masks"))
        trimap_path = os.path.abspath(os.path.join(self.data_root, "trimaps"))

        image_list = [os.path.join(image_path, name) for name in os.listdir(image_path)]
        mask_list = [os.path.join(mask_path, name) for name in os.listdir(mask_path)]
        trimap_list = [os.path.join(trimap_path, name) for name in os.listdir(trimap_path)]

        df = pandas.DataFrame(
            zip(image_list, mask_list, trimap_list),
            columns=['image', 'mask', 'trimap'],
        )
        return df