from ultralytics import YOLO
import sys
import os
from dataclasses import dataclass # type: ignore
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ..exception import CustomException
from ..logger import logging

@dataclass
class ModelLoaderConfig:
    segmentation_model_path = 'model\model.pt'

class ModelLoader:
    def __init__(self)->None:
        self.model_loader_config = ModelLoaderConfig()
    
    def load_model(self):
        try:
            logging.info("Model loaded successfully")
            return YOLO(self.model_loader_config.segmentation_model_path)
        except Exception as e:
            raise CustomException(e,sys)
    
# if __name__=='__main__':
#     obj = ModelLoader()
#     obj.load_model()