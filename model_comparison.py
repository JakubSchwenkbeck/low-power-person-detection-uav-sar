from csv import writer
import sys, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from PIL import Image
import tflite_runtime.interpreter as tflite
# from edge_impulse_linux.image import ImageImpulseRunner



def load_labels(label_path):
    with open(label_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels  


def postprocess_yolo(output, conf_threshold=0.5):
    # Transpose the output from (84, 8400) to (8400, 84)
    output = np.squeeze(output).T 

    boxes, scores, class_ids = [], [], []

    # Iterate over all the 8400 detections
    for row in output:
        # Get the class scores and find the one with the highest value
        class_probs = row[4:]
        max_score = np.max(class_probs)
        
        # If the highest score is above the confidence threshold
        if max_score > conf_threshold:
            # Get the class ID
            class_id = np.argmax(class_probs)
            
            # Extract the bounding box coordinates
            box = row[:4] # [center_x, center_y, width, height]
            
            # Store the results
            boxes.append(box.tolist())
            scores.append(float(max_score))
            class_ids.append(int(class_id))

    # Assemble the detections in the required format
    results = []
    for i in range(len(scores)):
        results.append({
            "bbox": boxes[i],
            "score": scores[i],
            "class_id": class_ids[i]
        })
        
    return results


def nms(detections, iou_threshold=0.45):
    if len(detections) == 0:
        return []
    boxes = np.array([d['bbox'] for d in detections])
    scores = np.array([d['score'] for d in detections])
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    keep = []
    idxs = np.argsort(-scores)
    while len(idxs) > 0:
        current = idxs[0]
        keep.append(current)
        if len(idxs) == 1:
            break
        ious = compute_iou(boxes_xyxy[current], boxes_xyxy[idxs[1:]])
        idxs = idxs[1:][ious < iou_threshold]
    return [detections[i] for i in keep]

def compute_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter_w = np.maximum(0, x2 - x1)
    inter_h = np.maximum(0, y2 - y1)
    inter_area = inter_w * inter_h
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - inter_area
    return inter_area / (union_area + 1e-6)


def run_inference(image: np.ndarray,
                  model: str,
                  interpreter: tflite.Interpreter,
                  size: str = "fp32",
                  conf_threshold: float = 0.5,
                  profiling: bool = False
                  ):
    """
    Run inference ...

    To measure:
    - end-to-end latency
    - inference latency
    - throughput (FPS)
    - memory footprint
    - CPU utilization
    - temperature
    - power consumption
    - precision and recall
    - mAP (mean Average Precision)
    - qualitative analysis

    Args:
        image: The input image in cv2 format (numpy.ndarray).
    """

    if model not in ['yolo', 'mobilenet', 'efficientdet']:
        print("Model not supported")
        return
    
    input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_img = cv2.resize(input_img, (interpreter.get_input_details()[0]['shape'][1],
                                         interpreter.get_input_details()[0]['shape'][2]))

    resizing_factor_w = input_img.shape[1] / image.shape[1]
    resizing_factor_h = input_img.shape[0] / image.shape[0]

    # print(f"Resizing factors - Width: {resizing_factor_w}, Height: {resizing_factor_h}")

    if model in ['yolo']:

        if interpreter.get_input_details()[0]['dtype'] == np.float32:
            # print("Model expects float32 input")
            input_img = input_img.astype(np.float32) / 255.0
        else: # This assumes uint8. Check your model's exact type.
            # print("Model expects uint8 input")
            input_img = input_img.astype(np.uint8)

        # if size == "int8":
        #     print("Model expects int8 input")
        #     input_img = input_img.astype(np.uint8)
        # else:
        #     print("Model expects float32 input")
        #     input_img = input_img.astype(np.float32) / 255.0  # normalize to [0, 1]

    input_data = np.expand_dims(input_img, axis=0)

    # Inference
    start_time = time.time()
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
    print ("Inference time: {:.1f}ms".format(inference_time))
    
    # Post processing
    output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    detections = []
    boxes = []
    classes = []
    scores = []
    num_detections = 0

    # Extract the outputs
    if model in ['yolo']:
        detections = postprocess_yolo(output_data, conf_threshold=conf_threshold)
        detections = nms(detections, iou_threshold=0.45)
    elif model in ['mobilenet', 'efficientdet']:
        boxes = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]  
        classes = interpreter.get_tensor(interpreter.get_output_details()[1]['index'])[0]  
        scores = interpreter.get_tensor(interpreter.get_output_details()[2]['index'])[0]        
        num_detections = int(interpreter.get_tensor(interpreter.get_output_details()[3]['index'])[0])

    # Create the output image
    output_img = image.copy()

    # Draw bounding boxes on the ORIGINAL image
    if model in ['yolo']:
        for det in detections:

            class_id = int(det['class_id'])

            # if class_id not in [0,1]:
            #     continue

            # Bbox is [center_x, center_y, width, height] normalized to model input size
            center_x_norm, center_y_norm, w_norm, h_norm = det['bbox']

            # Scale coordinates to original image size
            box_w = int(w_norm * input_img.shape[1] / resizing_factor_w)
            box_h = int(h_norm * input_img.shape[0] / resizing_factor_h)
            center_x = int(center_x_norm * input_img.shape[1] / resizing_factor_w)
            center_y = int(center_y_norm * input_img.shape[0] / resizing_factor_h)

            # box_w = int(w_norm / resizing_factor_w)
            # box_h = int(h_norm / resizing_factor_h)
            # center_x = int(center_x_norm / resizing_factor_w)
            # center_y = int(center_y_norm / resizing_factor_h)

            # Calculate top-left corner (x1, y1)
            x1 = center_x - (box_w // 2)
            y1 = center_y - (box_h // 2)
            
            # Calculate bottom-right corner (x2, y2)
            x2 = x1 + box_w
            y2 = y1 + box_h
            
            # Draw the rectangle
            cv2.rectangle(output_img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
            
            # Prepare the label text
            class_name = labels[class_id]
            score = det['score']
            label = f'{class_name}: {score:.2f}'

            # Draw the label text above the box
            cv2.putText(output_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    elif model in ['mobilenet', 'efficientdet']:
        
        for i in range(num_detections):
            if scores[i] > conf_threshold:  # Adjust threshold as needed
                ymin, xmin, ymax, xmax = boxes[i]
                x1 = int(xmin * input_img.shape[1] / resizing_factor_w)
                y1 = int(ymin * input_img.shape[0] / resizing_factor_h)
                x2 = int(xmax * input_img.shape[1] / resizing_factor_w)
                y2 = int(ymax * input_img.shape[0] / resizing_factor_h)

                class_id = int(classes[i])
                class_name = labels[class_id]
                score = scores[i]
                label = f'{class_name}: {score:.2f}'

                cv2.rectangle(output_img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(output_img, (x1, y1 - text_h - baseline), (x1 + text_w, y1), (0, 0, 255), -1) # -1 thickness for filled
                cv2.putText(output_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Black text

    return output_img


# mobilenet_model_path = "./models/mobilenet_ssd_latency_dynamic.tflite"
mobilenet_model_path = "./models/mobilenet_ssd_latency_dynamic.tflite"
efficientdet_model_path = "./models/efficientdet.tflite"
# yolo_model_path = "models/yolo11n_float32.tflite"
# yolo_model_path = "./models/yolo11n_latency_dynamic.tflite"
# yolo_model_path = "./models/c2a_best_float32.tflite"
yolo_model_path = "./models/visdrone_480_best_float32.tflite"

mobilenet_interpreter = tflite.Interpreter(model_path=mobilenet_model_path)
mobilenet_interpreter.allocate_tensors()

efficientdet_interpreter = tflite.Interpreter(model_path=efficientdet_model_path)
efficientdet_interpreter.allocate_tensors()

yolo_interpreter = tflite.Interpreter(model_path=yolo_model_path)
yolo_interpreter.allocate_tensors()

labels = load_labels('./models/coco_labels.txt')
len(labels)

confidence = 0.3

img_path = "./images/beatch.jpg"
original_img = cv2.imread(img_path)


# print("Running SSD-MobileNet V1 inference...")
# start_time = time.time()
# mobilenet_output = run_inference(original_img,
#                                  model='mobilenet',
#                                  interpreter=mobilenet_interpreter,
#                                  conf_threshold=confidence,
#                                  profiling=True)
# end_time = time.time()
# elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
# print ("SSD-MobileNet-V1 end-to-end time: {:.1f}ms".format(elapsed_time))
# cv2.imshow("SSD-MobileNet V1", mobilenet_output)
# cv2.waitKey(0)  # Wait for a key press to close the window
# cv2.destroyAllWindows()


# print("Running EfficientDet inference...")
# start_time = time.time()
# efficientdet_output = run_inference(original_img,
#                                     model='efficientdet',
#                                     interpreter=efficientdet_interpreter,
#                                     conf_threshold=confidence,
#                                     profiling=True)
# end_time = time.time()
# elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
# print ("EfficientDet end-to-end time: {:.1f}ms".format(elapsed_time))
# cv2.imshow("EfficientDet", efficientdet_output)
# cv2.waitKey(0)  # Wait for a key press to close the window
# cv2.destroyAllWindows()


print("Running YOLO inference...")
start_time = time.time()
yolo_output = run_inference(original_img,
                            model='yolo',
                            interpreter=yolo_interpreter,
                            conf_threshold=confidence,
                            profiling=True)
end_time = time.time()
elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
print ("YOLO end-to-end time: {:.1f}ms".format(elapsed_time))
cv2.imshow("YOLO", yolo_output)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()


# detect_objects(img_path, 0.5)


# FOMO

# # Load the model file
# runner = ImageImpulseRunner(model_path)
# model_info = runner.init()

# # Load the image
# img_path = "./images/notebook/png/orig-fomo-img.png"
# orig_img = cv2.imread(img_path)
# img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

# # Display the image
# plt.imshow(img_rgb)
# plt.title("Original Image")
# plt.show()


video_path = "/media/gabriele/Data/remote_drone_footage.mp4"
# video_path = "/media/gabriele/Data/water_rescue_drone_footage.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Could not open source: {video_path}")

frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if frame_count % 10 != 0:
            continue

        # cv2.imshow("Video", frame)
        # cv2.waitKey(1)

        start_time = time.time()
        yolo_output = run_inference(frame,
                                    model='yolo',
                                    interpreter=yolo_interpreter,
                                    conf_threshold=confidence,
                                    profiling=True)
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
        print ("YOLO end-to-end time: {:.1f}ms".format(elapsed_time))
        cv2.imshow("YOLO", yolo_output)
        cv2.waitKey(10)


finally:
    cap.release()
    cv2.destroyAllWindows()