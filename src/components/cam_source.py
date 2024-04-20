import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass # type: ignore

@dataclass
class CamSourceConfig:
    WEBCAM = 'webcam'
    WEBCAM_PATH = 0
    SOURCES_LIST = [WEBCAM]

class CamSource:
    def __init__(self)->None:
        self.cam_source_config = CamSourceConfig()
    
    def get_source_list(self):
        return self.cam_source_config.SOURCES_LIST
    
    def get_webcam_path(self):
        return self.cam_source_config.WEBCAM_PATH
    

