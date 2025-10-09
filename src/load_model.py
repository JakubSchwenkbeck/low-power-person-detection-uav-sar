# load_model.py
from ultralytics import YOLO
import tensorflow as tf
import os

def load_yolo(model_name : str, model_name_ext: str):
    """
    Loads a YOLO modle
    """
    os.makedirs("models", exist_ok=True)

    model = YOLO(model_name_ext)
    exported_path = model.export(format="saved_model")


    return exported_path