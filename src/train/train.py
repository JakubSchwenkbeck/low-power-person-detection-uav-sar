from ultralytics import YOLO
import torch

print(torch.__version__)
print(torch.version.cuda)
print("PyTorch CUDA: ", torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

model_name = "yolo11n"
dataset_name = "visdrone"
data_path = "train/visdrone.yaml"
image_size = 640
epochs = 100


def train(model_name: str, dataset_name:str, data_path: str, image_size: int, epochs: int):
    # Load a pretrained model
    model = YOLO(model_name + ".pt")
    # Train the model using a custom dataset
    results = model.train(
        data=data_path, 
        device=0,
        epochs=epochs,
        imgsz=image_size,
        batch=16,
        plots=True,
        project="../models",
        name= model_name + "_" + dataset_name + "_" + str(image_size) + "p_" + str(epochs) + "ep"
    )

    return results

def export(data_path: str, best_model_path: str, image_size: int):
    best_model = YOLO(best_model_path)
    best_model.export(format="onnx")
    best_model.export(
        format="tflite",
        imgsz=image_size,
        # project="../models",
        # name="yolo11n_fp32_visdrone"
    )
    best_model.export(
        format="tflite",
        imgsz=image_size,
        half=True,
        # project="../models",
        # name="yolo11n_fp16_visdrone"
    )
    best_model.export(
        format="tflite",
        imgsz=image_size,
        int8=True,
        data=data_path,
        # project="../models",
        # name="yolo11n_int8_visdrone"
    )

if __name__ == '__main__':

    # from multiprocessing import freeze_support
    # freeze_support()

    # print("Starting training ", model_name, " on dataset ", dataset_name,
    #     " for ", epochs, " epochs with imgsz ", image_size, "p.")

    # results = train(model_name, dataset_name, data_path, image_size, epochs)
    # print(results)

    best_model_path = f"../models/{model_name}_{dataset_name}_{image_size}p_{epochs}ep/weights/best.pt"

    export(data_path=data_path, best_model_path=best_model_path, image_size=image_size)



