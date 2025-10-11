# load_model.py
from ultralytics import YOLO
import tensorflow as tf
import tensorflow_hub as hub
import os

MODELS_PATH = "data/models"
os.makedirs(MODELS_PATH, exist_ok=True)

def load_yolo(model_name : str, model_name_ext: str):
    """
    Loads a YOLO modle
    """
    model = YOLO(model_name_ext)
    exported_path = model.export(format="saved_model")


    return exported_path


def load_mobilenet_ssd(model_name: str, model_url: str = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"):
    """Loads MobileNet SSD from TensorFlow Hub"""
    model = hub.load(model_url)
    saved_model_path = f"{model_name}_saved_model"
    tf.saved_model.save(model, saved_model_path)
    return saved_model_path


def load_efficientdet(model_name: str, model_url: str = "https://tfhub.dev/tensorflow/efficientdet/d0/1"):
    """Loads EfficientDet from TensorFlow Hub"""
    model = hub.load(model_url)
    saved_model_path = f"{model_name}_saved_model"
    tf.saved_model.save(model, saved_model_path)
    return saved_model_path