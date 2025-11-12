try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

try:
    from edge_impulse_linux.image import ImageImpulseRunner
except ImportError:
    ImageImpulseRunner = None

import numpy as np
import cv2


PERSON_CLASS_ID = 0

class Model:

    def __init__(self, model_type: str, path: str):
        assert model_type in ['yolo', 'fomo']
        self.model_type = model_type

        if model_type == 'yolo':
            self.interpreter = tflite.Interpreter(
                model_path=path,
                num_threads=4,
            )
            self.interpreter.allocate_tensors()

            self.interpreter_input_details = self.interpreter.get_input_details()[0]
            self.interpreter_output_details = self.interpreter.get_output_details()[0]

            print(self.interpreter_input_details)

            self.input_size = (self.interpreter_input_details['shape'][1], self.interpreter_input_details['shape'][2])
        else:
            if ImageImpulseRunner is None:
                raise ImportError("edge_impulse_linux is required for FOMO models")
            self.interpreter = ImageImpulseRunner(path)
            self.interpreter.init()


    def inference(self, image: np.ndarray, postprocess: bool = True, conf_threshold: float = 0.5, nms_threshold: float = 0.45):
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.model_type == 'yolo':
            input_img = cv2.resize(input_img, self.input_size)
            input_img = input_img.astype(np.float32) / 255.0

            input_data = np.expand_dims(input_img, axis=0)

            # if self.interpreter_input_details['dtype'] == np.int8:
            #     scale, zero_point = self.interpreter_input_details['quantization']
            #     input_data = input_data / 255.0
            #     input_data = input_data / scale + zero_point
            #     input_data = np.clip(input_data, -128, 127).astype(np.int8)

            self.interpreter.set_tensor(self.interpreter_input_details['index'], input_data)
            self.interpreter.invoke()
            
            output_data = self.interpreter.get_tensor(self.interpreter_output_details['index'])
            if postprocess:
                return self.postprocess(image, output_data, conf_threshold=conf_threshold, nms_threshold=nms_threshold)
        else:
            features, cropped = self.interpreter.get_features_from_image_auto_studio_settings(input_img)
            res = self.interpreter.classify(features)
            if postprocess:
                return self.postprocess(image, res['result'], conf_threshold=conf_threshold)


    def postprocess(self, image, output, conf_threshold=0.5, nms_threshold=0.45):
        if self.model_type == 'yolo':
                
            output = np.squeeze(output).T
            
            print(f"Debug: Output shape after transpose: {output.shape}")

            boxes = output[:, :4]
            class_scores = output[:, 4:]

            person_scores = class_scores[:, PERSON_CLASS_ID]

            mask = person_scores > conf_threshold
            filtered_boxes = boxes[mask]
            filtered_scores = person_scores[mask]
            
            print(f"Debug: Found {len(filtered_boxes)} boxes above threshold {conf_threshold}")
            if len(filtered_boxes) > 0:
                print(f"Debug: First box values (x,y,w,h): {filtered_boxes[0]}")

            indices = cv2.dnn.NMSBoxes(
                bboxes=filtered_boxes.tolist(),
                scores=filtered_scores.tolist(),
                score_threshold=conf_threshold,
                nms_threshold=nms_threshold
            )

            if len(indices) == 0:
                return None
                
            results = [list(box) + [score] for box, score in zip(filtered_boxes[indices], filtered_scores[indices])]
            print(f"Debug: After NMS, {len(results)} detections remaining")

        else:
            bboxes = output['bounding_boxes']
            results = [[box['x'], box['y'], box['width'], box['height'], box['value']] for box in bboxes if box['label'] == '0' and box['value'] > conf_threshold]

        if len(results) == 0:
            return None
        
        img_height, img_width, _ = image.shape
        print(f"Debug: Image dimensions: {img_width}x{img_height}")
        
        # Create a copy to draw on
        image = image.copy()

        for i, result in enumerate(results):
            x, y, w, h, score = result
            print(f"Debug: Box {i} raw values - x:{x:.3f}, y:{y:.3f}, w:{w:.3f}, h:{h:.3f}")
            
            x1 = int(x * img_width - w * img_width / 2)
            y1 = int(y * img_height - h * img_height / 2)
            x2 = int(x * img_width + w * img_width / 2)
            y2 = int(y * img_height + h * img_height / 2)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, img_width - 1))
            y1 = max(0, min(y1, img_height - 1))
            x2 = max(0, min(x2, img_width - 1))
            y2 = max(0, min(y2, img_height - 1))
            
            print(f"Debug: Drawing box at ({x1},{y1}) to ({x2},{y2}) with score {score:.3f}")

            # Draw rectangle - BGR format, so (0, 0, 255) is RED
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # Draw filled background for text
            label = f"{score:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 0, 255), -1)

            # Draw text
            cv2.putText(image, 
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, lineType=cv2.LINE_AA)

        return image


