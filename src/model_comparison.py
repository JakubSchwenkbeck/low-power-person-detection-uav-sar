import sys, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from PIL import Image
import tflite_runtime.interpreter as tflite
# from edge_impulse_linux.image import ImageImpulseRunner


# print("NumPy:", np.__version__)
# print("Pillow:", Image.__version__)

# model_path = "./models/ssd-mobilenet-v1-tflite-default-v1.tflite"
# model_path = "models/yolo11n_float32.tflite"
# model_path = "models/yolo11n_latency_dynamic.tflite"
model_path = "./data/models/yolov5s_f16.tflite"
# fomo_model_path = "models/FOMO-int8.eim"

interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
print("TFLite Interpreter created successfully!")

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# print(input_details)
# print(output_details)

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


def detect_objects(img_path, conf=0.5):

    original_img = cv2.imread(img_path)

    # Get original image dimensions
    original_h, original_w = original_img.shape[:2]

    img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_details[0]['shape'][1], 
                      input_details[0]['shape'][2]))
    
    resizing_factor_w = img.shape[1] / original_w
    resizing_factor_h = img.shape[0] / original_h
    print(f"Resizing factors - Width: {resizing_factor_w}, Height: {resizing_factor_h}")

    img = img.astype(np.float32) / 255.0  # normalize to [0, 1]
    input_data = np.expand_dims(img, axis=0)
    
    # Inference
    start_time = time.time()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
    print ("Inference time: {:.1f}ms".format(inference_time))
    
    # Post processing
    output_data = interpreter.get_tensor(output_details[0]['index'])
    # print(output_data)
    # print(output_data.shape)
    detections = postprocess_yolo(output_data, conf_threshold=0.5)
    # print(detections)
    detections = nms(detections, iou_threshold=0.45)
    print(f"Found {len(detections)} objects after NMS")
    # print(detections)


    # Draw bounding boxes on the ORIGINAL image
    for det in detections:
        # Bbox is [center_x, center_y, width, height] normalized to model input size
        center_x_norm, center_y_norm, w_norm, h_norm = det['bbox']

        # Scale coordinates to original image size
        box_w = int(w_norm / resizing_factor_w)
        box_h = int(h_norm / resizing_factor_h)
        center_x = int(center_x_norm / resizing_factor_w)
        center_y = int(center_y_norm / resizing_factor_h)

        # Calculate top-left corner (x1, y1)
        x1 = center_x - (box_w // 2)
        y1 = center_y - (box_h // 2)
        
        # Calculate bottom-right corner (x2, y2)
        x2 = x1 + box_w
        y2 = y1 + box_h
        
        # Draw the rectangle
        # Note: OpenCV uses BGR color format, so (0, 255, 0) is green
        cv2.rectangle(original_img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
        # cv2.rectangle(img, 
        #               (int(center_x_norm - w_norm / 2), int(center_y_norm - h_norm / 2)), 
        #               (int(center_x_norm + w_norm / 2), int(center_y_norm + h_norm / 2)),
        #               color=(0, 255, 0),
        #               thickness=2)
        
        # Prepare the label text
        print(det)
        class_id = int(det['class_id'])
        print(class_id)
        class_name = labels[class_id]
        score = det['score']
        label = f'{class_name}: {score:.2f}'

        # Draw the label text above the box
        cv2.putText(original_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("YOLO", original_img)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()


    # plt.figure(figsize=(12, 8))
    # plt.imshow(img)
    # for det in detections:
    #     if det.score > conf:  # Adjust threshold as needed
    #         ymin, xmin, ymax, xmax = det.bbox
    #         (left, right, top, bottom) = (xmin * img.shape[1], xmax * img.shape[1], 
    #                                       ymin * img.shape[0], ymax * img.shape[0])
    #         rect = plt.Rectangle((left, top), right-left, bottom-top, 
    #                              fill=False, color='red', linewidth=2)
    #         plt.gca().add_patch(rect)
    #         class_id = int(det.class_id)
    #         class_name = labels[class_id]
    #         plt.text(left, top-10, f'{class_name}: {det.score:.2f}', 
    #                  color='red', fontsize=12, backgroundcolor='white')
            

    # for det in detections:
    #     # Bbox is [center_x, center_y, width, height] normalized to input size (e.g., 640x640)
    #     center_x, center_y, w, h = det['bbox']
        
    #     # Scale coordinates back to original image size
    #     x_scaled = center_x * original_w
    #     y_scaled = center_y * original_h
    #     w_scaled = w * original_w
    #     h_scaled = h * original_h
        
    #     # Calculate top-left corner (x1, y1)
    #     left = x_scaled - (w_scaled / 2)
    #     top = y_scaled - (h_scaled / 2)

    #     # Create rectangle
    #     rect = plt.Rectangle((left, top), w_scaled, h_scaled, 
    #                          fill=False, color='red', linewidth=2)
    #     plt.gca().add_patch(rect)
        
    #     # Add text label
    #     class_id = int(det['class_id'])
    #     class_name = labels[class_id]
    #     score = det['score']
    #     plt.text(left, top - 10, f'{class_name}: {score:.2f}', 
    #              color='red', fontsize=12, backgroundcolor='white')


    # Extract the outputs
    # boxes = interpreter.get_tensor(output_details[0]['index'])[0]  
    # classes = interpreter.get_tensor(output_details[1]['index'])[0]  
    # scores = interpreter.get_tensor(output_details[2]['index'])[0]        
    # num_detections = int(interpreter.get_tensor(output_details[3]['index'])[0])
    
    # Visualize the results
    # plt.figure(figsize=(12, 8))
    # plt.imshow(img)
    # for i in range(num_detections):
    #     if scores[i] > conf:  # Adjust threshold as needed
    #         ymin, xmin, ymax, xmax = boxes[i]
    #         (left, right, top, bottom) = (xmin * img.shape[1], xmax * img.shape[1], 
    #                                       ymin * img.shape[0], ymax * img.shape[0])
    #         rect = plt.Rectangle((left, top), right-left, bottom-top, 
    #                              fill=False, color='red', linewidth=2)
    #         plt.gca().add_patch(rect)
    #         class_id = int(classes[i])
    #         class_name = labels[class_id]
    #         plt.text(left, top-10, f'{class_name}: {scores[i]:.2f}', 
    #                  color='red', fontsize=12, backgroundcolor='white')
            

labels = load_labels('./data/labels/coco_labels.txt')
len(labels)
print(labels[:20])


img_path = "./data/images/val2017/000000481404.jpg"
detect_objects(img_path, 0.5)


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
