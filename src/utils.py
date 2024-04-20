import streamlit as st
import cv2
import sys
from components.cam_source import CamSource
from logger import logging
from exception import CustomException


def _display_detected_frames(conf, model, st_frame, image):
    """
    Display the segmented leopard on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.

    Returns:
    None
    """
    try:
        # Resize the image to a standard size
        image = cv2.resize(image, (720, int(720*(9/16))))

        res = model.predict(image, conf=conf)

        res_plotted = res[0].plot()
        st_frame.image(res_plotted,
                    caption='Segmented Feed',
                    channels="BGR",
                    use_column_width=True
                    )
    except Exception as e:
        raise CustomException(e,sys)

def play_webcam(conf, model):
    """
    Plays a webcam stream. Segments Leopard in real-time using the fine tuned YOLOv8 leopard segmentation model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: A fine tuned instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    try:

        logging.info("Inside play_webcam function")
        cam_source = CamSource()
        source_webcam = cam_source.get_webcam_path()
        if st.sidebar.button('Detect Objects'):
            try:
                vid_cap = cv2.VideoCapture(source_webcam)
                st_frame = st.empty()
                while (vid_cap.isOpened()):
                    success, image = vid_cap.read()
                    if success:
                        _display_detected_frames(conf,
                                                model,
                                                st_frame,
                                                image,
                                                )
                    else:
                        vid_cap.release()
                        break
            except Exception as e:
                st.sidebar.error("Error loading video: " + str(e))

    except Exception as e:
        raise CustomException(e,sys)