from ultralytics import YOLO
import torch
# print(torch.__version__)
# print(torch.version.cuda)

print("PyTorch CUDA: ", torch.cuda.is_available())
# print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
# print("Ultralytics CUDA: ", YOLO("yolo11n.pt").device)


def train():
    # Load a model
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

    # Train the model using VisDrone dataset
    # results = model.train(data="VisDrone.yaml", device=0, epochs=100, imgsz=640, batch=16, plots=True)
    # Train the model using C2A dataset
    results = model.train(data="c2a-yolo.yaml", device=0, epochs=100, imgsz=640, batch=16, plots=True)

def export():
    best_model = YOLO("runs/detect/train/weights/best.pt")
    best_model.export(format="onnx")
    # best_model.export(format="tflite", imgsz=640, half=False, int8=False)
    # best_model.export(format="tflite", imgsz=640, int8=True, data="VisDrone.yaml")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()  # optional, but safe on Windows
    train()
    # export()
