import streamlit as st
import cv2
import sys
from components.cam_source import CamSource
from logger import logging
from exception import CustomException
import json
from playsound import playsound

i=0
def _write_output_json(res,filename):
    """
    Write the json output to the outputs folder

    Args:
    - res : result set of the predict function of YOLO
    - filename: filename as per the image

    Returns:
    None
    """
    # Create a dictionary to store the data
    output_data = {}

    # Add shape of the image to the dictionary
    output_data["image_shape"] = res.orig_shape

    # Add confidence score to the dictionary
    output_data["confidence_score"] = res.boxes.conf.item()

    # Add bounding boxes to the dictionary
    output_data["bounding_boxes"] = res.boxes.xyxy[0].tolist()

    # Add mask values to the dictionary
    output_data["mask_values"] = res.masks.xy[0].tolist()

    # Write the dictionary to a JSON file
    filename = filename + ".json"
    with open(filename, "w") as json_file:
        json.dump(output_data, json_file, indent=4)
        logging.info(f"File {filename} wrote successfully")


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
        global i
        # Resize the image to a standard size
        image = cv2.resize(image, (720, int(720*(9/16))))

        res = model.predict(image, conf=conf)

        res_plotted = res[0].plot()

        if(len(res[0].boxes.conf)>0):
            playsound('model/warning.mp3')
            filename = 'outputs/result'+str(i)
            cv2.imwrite(filename+'.jpg',res_plotted)
            _write_output_json(res[0],filename)
            print("LEOPARD SPOTTED".center(80,'-'))
            # playsound("")
            i+=1
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
                logging.info("Starting the camera feed")
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