import os
from load_model import load_and_convert_yolo
from optimize_model import optimize_model


def main():
    """
    Main pipeline to load and optimize models.
    """
    # Model configuration
    model_name = "yolo11n"
    model_name_ext = "yolo11n.pt"  # YOLO11n model
  
    load_and_convert_yolo(model_name_ext)
    
    saved_model_path = "yolo11n_saved_model"
    
   
    if os.path.exists(saved_model_path):
        optimized_path = optimize_model(model_name, saved_model_path)
   


if __name__ == "__main__":
    main()
