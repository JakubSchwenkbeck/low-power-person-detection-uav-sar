# load_model.py
from ultralytics import YOLO
import tensorflow as tf
import os

def load_and_convert_yolo(model_name: str):
    """
    Loads a YOLO modle
    """
    os.makedirs("models", exist_ok=True)

    model = YOLO(model_name)
    model.export(format="saved_model")

