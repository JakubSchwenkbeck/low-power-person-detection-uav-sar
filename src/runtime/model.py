import tflite_runtime.interpreter as tflite
import numpy as np
import cv2


PERSON_CLASS_ID = 0

class YoloModel:

    def __init__(self, path: str):
        self.interpreter = tflite.Interpreter(model_path=path)
        self.interpreter.allocate_tensors()

        self.interpreter_input_details = self.interpreter.get_input_details()[0]
        self.interpreter_output_details = self.interpreter.get_output_details()[0]

        self.input_size = (self.interpreter_input_details['shape'][1], self.interpreter_input_details['shape'][2])



    def inference(self, image: np.ndarray):
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, self.input_size)
        input_img = input_img.astype(np.float32) / 255.0

        input_data = np.expand_dims(input_img, axis=0)

        self.interpreter.set_tensor(self.interpreter_input_details['index'], input_data)
        self.interpreter.invoke()
        
        output_data = self.interpreter.get_tensor(self.interpreter_output_details['index'])
        return self.postprocess(image, output_data)


    def postprocess(self, image, output, conf_threshold=0.5, nms_threshold=0.45):
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

        if len(indices) == 0:
            return None

        for index in indices:
            box = filtered_boxes[index]
            score = filtered_scores[index]

            x, y, w, h = box
            x1, y1, x2, y2 = int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1

            cv2.putText(image, str(score), (x1 + 10, y1 + 20), font, font_scale, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)


        return image


