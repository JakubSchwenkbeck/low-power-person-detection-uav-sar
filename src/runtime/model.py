import numpy as np
import cv2

PERSON_CLASS_ID = 0


class Model:

    def __init__(self, model_type: str, path: str):
        assert model_type in ['yolo', 'fomo']
        self.model_type = model_type

        # Lazy imports for optional heavy dependencies
        if model_type == 'yolo':
            try:
                import tflite_runtime.interpreter as tflite
            except Exception as e:
                raise ImportError(
                    "tflite_runtime is required to load 'yolo' models."
                    " Install tflite_runtime in your environment or use a different model type."
                ) from e

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
            try:
                from edge_impulse_linux.image import ImageImpulseRunner
            except Exception as e:
                raise ImportError(
                    "edge_impulse_linux is required to load 'fomo' models."
                    " Install edge_impulse_linux in your environment or use a different model type."
                ) from e

            self.interpreter = ImageImpulseRunner(path)
            self.interpreter.init()


    def inference(self, image: np.ndarray, postprocess: bool = True):
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
                return self.postprocess(image, output_data, conf_threshold=0.5, nms_threshold=0.45)
        else:
            features, cropped = self.interpreter.get_features_from_image_auto_studio_settings(input_img)
            res = self.interpreter.classify(features)
            if postprocess:
                return self.postprocess(image, res['result'])


    def postprocess(self, image, output, conf_threshold=0.5, nms_threshold=0.45):
        if self.model_type == 'yolo':
                
            output = np.squeeze(output).T
            boxes = output[:, :4]
            class_scores = output[:, 4:]

            person_scores = class_scores[:, PERSON_CLASS_ID]

            mask = person_scores > conf_threshold
            filtered_boxes = boxes[mask]
            filtered_scores = person_scores[mask]

            # No detections
            if filtered_boxes.size == 0:
                return None

            img_height, img_width, _ = image.shape

            # Convert normalized center-format boxes [cx,cy,w,h] to pixel [x,y,w,h]
            pixel_boxes = []
            for b in filtered_boxes:
                cx, cy, w, h = b
                x = int(cx * img_width - (w * img_width) / 2)
                y = int(cy * img_height - (h * img_height) / 2)
                w_px = int(w * img_width)
                h_px = int(h * img_height)
                # ensure box inside image
                x = max(0, x)
                y = max(0, y)
                w_px = max(0, min(w_px, img_width - x))
                h_px = max(0, min(h_px, img_height - y))
                pixel_boxes.append([x, y, w_px, h_px])

            # Run NMS on pixel boxes
            # OpenCV expects lists, and returns indices in various formats depending on version
            try:
                indices = cv2.dnn.NMSBoxes(pixel_boxes, filtered_scores.tolist(), conf_threshold, nms_threshold)
            except Exception:
                # fallback: no nms
                indices = list(range(len(pixel_boxes)))

            # normalize indices to flat list
            flat_indices = []
            if indices is None:
                flat_indices = []
            elif isinstance(indices, (list, tuple)):
                # could be list of ints or list of [i]
                for it in indices:
                    if isinstance(it, (list, tuple, np.ndarray)):
                        flat_indices.append(int(it[0]))
                    else:
                        flat_indices.append(int(it))
            else:
                try:
                    flat = np.array(indices).flatten()
                    flat_indices = [int(x) for x in flat]
                except Exception:
                    flat_indices = []

            results_pixels = []
            for i in flat_indices:
                if i < 0 or i >= len(pixel_boxes):
                    continue
                x, y, w_px, h_px = pixel_boxes[i]
                score = float(filtered_scores[i])
                results_pixels.append([x, y, x + w_px, y + h_px, score])

        else:
            bboxes = output['bounding_boxes']
            results_pixels = []
            img_height, img_width, _ = image.shape
            for box in bboxes:
                if box['label'] == '0' and box['value'] > conf_threshold:
                    cx = box['x']
                    cy = box['y']
                    w = box['width']
                    h = box['height']
                    x1 = int(cx * img_width - w * img_width / 2)
                    y1 = int(cy * img_height - h * img_height / 2)
                    x2 = int(cx * img_width + w * img_width / 2)
                    y2 = int(cy * img_height + h * img_height / 2)
                    results_pixels.append([x1, y1, x2, y2, float(box['value'])])

        if len(results_pixels) == 0:
            return None

        # Draw boxes
        for result in results_pixels:
            x1, y1, x2, y2, score = result
            # clip
            x1 = max(0, min(x1, image.shape[1] - 1))
            x2 = max(0, min(x2, image.shape[1] - 1))
            y1 = max(0, min(y1, image.shape[0] - 1))
            y2 = max(0, min(y2, image.shape[0] - 1))

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image,
                        f"{score:.2f}",
                        (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        return image


