import yaml
import os 
import pandas

# CONFIG_FILE = "./config/datasets.yaml"

# def _read_dataset_config(dataset_name="amd"):
#     with open(CONFIG_FILE, 'r') as f:
#         dataset_configs = yaml.safe_load(f)
    
#     dataset_root = os.path.join(
#         dataset_configs["root"]["path"],
#         dataset_configs[dataset_name]["path"],
#          )

#     dataset_configs[dataset_name]["path"] = os.path.abspath(dataset_configs[dataset_name]["path"])

#     for key in dataset_configs[dataset_name]["train"].keys():
#         dataset_configs[dataset_name]["train"][key] = os.path.join(
#             dataset_root,
#             dataset_configs[dataset_name]["train"][key],
#         )
#         dataset_configs[dataset_name]["train"][key] = os.path.abspath(dataset_configs[dataset_name]["train"][key])

#     for key in dataset_configs[dataset_name]["test"].keys():
#         dataset_configs[dataset_name]["test"][key] = os.path.join(
#             dataset_root,
#             dataset_configs[dataset_name]["test"][key],
#         )
#         dataset_configs[dataset_name]["test"][key] = os.path.abspath(dataset_configs[dataset_name]["test"][key])

#     return dataset_configs[dataset_name]

def generate_dataframe(data_name="amd", train=True):
    dataset_config =  _read_dataset_config(dataset_name)

    if train:
        dataset_config = dataset_config["train"]
    else:
        dataset_config = dataset_config["test"]

    image_names = [os.path.join(dataset_config["image"], name) for name in os.listdir(dataset_config["image"])]
    mask_names = [os.path.join(dataset_config["mask"], name) for name in os.listdir(dataset_config["mask"])]
    trimap_names = [os.path.join(dataset_config["trimap"], name) for name in os.listdir(dataset_config["trimap"])]
    df = pandas.DataFrame(
        zip(image_names, mask_names, trimap_names),
        columns=['image', 'mask', 'trimap'],
    )
    return df

if __name__ == "__main__":
    print(_read_dataset_config())

    

