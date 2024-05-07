from abc import ABC
from pathlib import Path


class LanguageModel(ABC):
    """
    Large language model.
    """
    def __init__(self,  models_folder:Path, name:str, device:str='auto', chat_template:str=None):
        
        self.name = name
        self.pretrained_model_name_or_path = str(models_folder / name)
        self.device = device