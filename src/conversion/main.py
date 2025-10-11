import os
from load_model import load_yolo, load_mobilenet_ssd, load_efficientdet
from optimize_model import optimize_model


def main():
    """
    Main pipeline to load and optimize models.
    """

    # Model configuration
    model_name = "yolo11n"
    model_name_ext = "yolo11n.pt"
    saved_model_path = load_yolo(model_name, model_name_ext)
    if os.path.exists(saved_model_path):
        optimize_model(model_name, saved_model_path)
    
    # MobileNet SSD 
    saved_model_path = load_mobilenet_ssd("mobilenet_ssd")
    if os.path.exists(saved_model_path):
         optimize_model("mobilenet_ssd", saved_model_path)
    
    # EfficientDet
    saved_model_path = load_efficientdet("efficientdet_d0")
    if os.path.exists(saved_model_path):
         optimize_model("efficientdet_d0", saved_model_path)


if __name__ == "__main__":
    main()
