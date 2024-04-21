# Python In-built packages
from pathlib import Path # type: ignore
import PIL
import sys
from src.logger import logging
from src.exception import CustomException
# External packages
import streamlit as st
from src.components.model_loader import ModelLoader
from src.components.cam_source import CamSource
from src.utils import play_webcam
# Local Modules

# Setting page layout
st.set_page_config(
    page_title="Leopard Segmentation using YOLOv8",
    layout="wide",
    initial_sidebar_state="expanded"
)
logging.info("Page Config set successfully")


# Main page heading
st.title("Leopard Segmentation And Tracking using YOLOv8")
logging.info("Page Title set sucessfully")

# Sidebar
st.sidebar.header("ML Model Config")
logging.info("Sidebar Header set succesfully")

model_type = 'Segmentation'

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100
logging.info("Model Confidence slider set successfully")

# Load Pre-trained ML Model
try:
    model_loader = ModelLoader()
    model = model_loader.load_model()

    cam_source = CamSource()
    st.sidebar.header("Video Config")
    source_radio = st.sidebar.radio(
        "Select Source", cam_source.get_source_list())
    
    logging.info("Radio bar added")

    if source_radio == cam_source.cam_source_config.WEBCAM:
        play_webcam(confidence, model)
    else:
        st.error("Please select a valid source type!")

except Exception as e:
    st.error(f"Unable to load model. Check the specified path: {model_loader.model_loader_config.segmentation_model_path}")
    st.error(e)
    raise CustomException(e,sys)


