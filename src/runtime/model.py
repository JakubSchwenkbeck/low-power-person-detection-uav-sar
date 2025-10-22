import tflite_runtime.interpreter as tflite
from edge_impulse_linux.image import ImageImpulseRunner
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

            self.input_size = (self.interpreter_input_details['shape'][1], self.interpreter_input_details['shape'][2])
        else:
            self.interpreter = ImageImpulseRunner(path)
            self.interpreter.init()


    def inference(self, image: np.ndarray, postprocess: bool = True):
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.model_type == 'yolo':
            input_img = cv2.resize(input_img, self.input_size)
            input_img = input_img.astype(np.float32) / 255.0

            input_data = np.expand_dims(input_img, axis=0)

            if self.interpreter_input_details['dtype'] == np.int8:
                scale, zero_point = self.interpreter_input_details['quantization']
                input_data = input_data / 255.0
                input_data = input_data / scale + zero_point
                input_data = np.clip(input_data, -128, 127).astype(np.int8)

            self.interpreter.set_tensor(self.interpreter_input_details['index'], input_data)
            self.interpreter.invoke()
            
            output_data = self.interpreter.get_tensor(self.interpreter_output_details['index'])
            if postprocess:
                return self.postprocess(image, output_data)
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

            indices = cv2.dnn.NMSBoxes(
                bboxes=filtered_boxes,
                scores=filtered_scores,
                score_threshold=conf_threshold,
                nms_threshold=nms_threshold
            )

            results = [list(box) + [score] for box, score in zip(filtered_boxes[indices], filtered_scores[indices])]

        else:
            bboxes = output['bounding_boxes']
            results = [[box['x'], box['y'], box['width'], box['height'], box['value']] for box in bboxes if box['label'] == '0' and box['value'] > conf_threshold]

        if len(results) == 0:
            return None

        for result in results:
            x, y, w, h, score = result
            x1, y1, x2, y2 = int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1

            cv2.putText(image, str(score), (x1 + 10, y1 + 20), font, font_scale, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)


        return image


