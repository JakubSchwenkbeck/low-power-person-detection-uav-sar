import os
from load_model import load_and_convert
from optimize_model import optimize_model


def main():
    """
    Main pipeline to load and optimize models.
    """
    # Model configuration
    model_name = "yolo11n.pt"  # YOLO11n model
    model_type = "yolo"
  
    load_and_convert(model_name, model_type)
    
    saved_model_path = "runs/detect/train/weights/best_saved_model"
    
   
    if os.path.exists(saved_model_path):
        optimized_path = optimize_model(saved_model_path)
   


if __name__ == "__main__":
    main()
