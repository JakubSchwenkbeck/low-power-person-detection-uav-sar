import os
from load_model import load_yolo, load_mobilenet_ssd, load_efficientdet
from optimize_model import optimize_model

def main():
    """
    Main pipeline to load and optimize our selected models.
    """

    # Model configuration

    model_name = "yolo11n"
    model_name_ext = "yolo11n.pt"
    yolo_saved_model_path = load_yolo(model_name, model_name_ext)

    mobilenet_ssd_saved_model_path = load_mobilenet_ssd("mobilenet_ssd")

    efficientdet_saved_model_path = load_efficientdet("efficientdet_d0")

    if os.path.exists(yolo_saved_model_path):

        optimize_model(model_name, yolo_saved_model_path)

    if os.path.exists(mobilenet_ssd_saved_model_path):

        optimize_model("mobilenet_ssd", mobilenet_ssd_saved_model_path)

    if os.path.exists(efficientdet_saved_model_path):

        optimize_model("efficientdet_d0", efficientdet_saved_model_path)
    




if __name__ == "__main__":

    
    main()
