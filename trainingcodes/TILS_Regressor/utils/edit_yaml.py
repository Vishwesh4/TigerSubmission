"""
This script changes the locations specified in the yml file. This is made specifically for compute canada, inorder to online sampling
"""
import numpy as np
import ruamel.yaml
from pathlib import Path

def data_location_modifier(loc_mod:str,config_file:str) -> None:
    """
    Given loc_mod, changes the yaml files specified in config_file
    """
    yaml = ruamel.yaml.YAML()
    with open(config_file) as fp:
        config = yaml.load(fp)
    dataset_file = config["wholeslidedata"]["default"]["yaml_source"]

    with open(str(Path(config_file).parent/dataset_file)) as fp:
        data = yaml.load(fp)

    #Editing the yaml file    
    for i in range(len(data["training"])):
        #To consider path till wsirois
        wsi_path = Path(data["training"][i]["wsi"]["path"]).parts[-4:]
        wsa_path = Path(data["training"][i]["wsa"]["path"]).parts[-4:]
        data["training"][i]["wsi"]["path"] = str(loc_mod/Path(*wsi_path))
        data["training"][i]["wsa"]["path"] = str(loc_mod/Path(*wsa_path))

    try:
        #Test if data has key of inferece
        for i in range(len(data["inference"])):
            #To consider path till wsirois
            wsi_path = Path(data["inference"][i]["wsi"]["path"]).parts[-4:]
            wsa_path = Path(data["inference"][i]["wsa"]["path"]).parts[-4:]
            data["inference"][i]["wsi"]["path"] = str(loc_mod/Path(*wsi_path))
            data["inference"][i]["wsa"]["path"] = str(loc_mod/Path(*wsa_path))
    except:
        pass
    
    with open(str(Path(config_file).parent/dataset_file), "w") as f:
        yaml.dump(data, f)

def change_seed(yaml_loc):
    seed_change = np.random.randint(low=1,high=100000)
    yaml = ruamel.yaml.YAML()
    # yaml.preserve_quotes = True
    with open(yaml_loc) as fp:
        data = yaml.load(fp)
    data["wholeslidedata"]["default"]["seed"]=seed_change
    with open(yaml_loc, "w") as f:
        yaml.dump(data, f)

if __name__=="__main__":
    # from wholeslidedata.iterators import create_batch_iterator
    
    config_file = "./Hyperparameters/finetune_tissue_weighted_edit.yml"
    loc_mod = Path("/localscratch/ramanav.31798836.0")
    data_location_modifier(loc_mod,config_file)
    # with create_batch_iterator(mode="training", user_config="/home/ramanav/projects/rrg-amartel/ramanav/Projects/TigerChallenge/tigeralgorithmexample/modules/Hooknet/Hyperparameters/finetune_tissue_weighted_edit.yml", cpus=8) as trainloader:
        # data = next(trainloader)
        # print(np.shape(data))
