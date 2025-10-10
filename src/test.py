# picamera_tflite.py
"""
Capture from Picamera2 and run TFLite inference in a producer/consumer pattern.
- Requires: python3-picamera2 (system), tflite-runtime (pip) or tensorflow
- Headless friendly: no cv2.imshow used by default.
- Writes annotated output video if OUTPUT_FILE is set.
"""

import time
import threading
import queue
import argparse
import numpy as np
import cv2

# Prefer tflite-runtime for small installs; fall back to TensorFlow if available.
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except Exception:
    from tensorflow.lite.python.interpreter import Interpreter  # type: ignore

# Picamera2 import (system package)
try:
    from picamera2 import Picamera2, Preview
except Exception as e:
    raise SystemExit("Picamera2 not available: install python3-picamera2 (apt) and run under system python") from e


class CameraProducer(threading.Thread):
    """Continuously capture frames from Picamera2 and put the latest into a queue."""
    def __init__(self, q, width=640, height=480, fps=30):
        super().__init__(daemon=True)
        self.q = q
        self.width = width
        self.height = height
        self.fps = fps
        self._stop = threading.Event()

        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration({"main": {"size": (self.width, self.height), "format": "XRGB8888"}})
        self.picam2.configure(config)

    def run(self):
        self.picam2.start()
        try:
            while not self._stop.is_set():
                # capture_array returns RGB (H,W,3) uint8
                arr = self.picam2.capture_array()
                # convert to RGB numpy (it already is RGB) and ensure dtype=uint8
                frame = arr.astype(np.uint8)
                # non-blocking put newest frame; drop old
                try:
                    # empty queue then put latest so consumer always sees newest frame
                    while True:
                        self.q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.q.put_nowait(frame)
                except queue.Full:
                    # improbable because we emptied it, but ignore
                    pass
                # sleep is optional - capture_array is already paced by camera
        finally:
            self.picam2.stop()

    def stop(self):
        self._stop.set()


class TFLiteRunner(threading.Thread):
    """Consume frames and run TFLite inference. Print or annotate results."""
    def __init__(self, q, model_path, output_file=None, score_threshold=0.5, save_annotated=False):
        super().__init__(daemon=True)
        self.q = q
        self.model_path = model_path
        self.score_threshold = score_threshold
        self._stop = threading.Event()
        self.out = None
        self.save_annotated = save_annotated
        if output_file and save_annotated:
            self.output_file = output_file
        else:
            self.output_file = None

        # load interpreter
        self.interpreter = Interpreter(self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # model input shape and dtype
        inp = self.input_details[0]
        self.in_dtype = inp['dtype']
        # input shape could be (1,h,w,3) or (1,3,h,w) - we assume NHWC (common)
        self.input_shape = inp['shape']  # e.g. [1, 300, 300, 3]
        if len(self.input_shape) == 4:
            _, self.in_h, self.in_w, self.in_c = self.input_shape
        elif len(self.input_shape) == 3:
            # maybe missing batch dim
            self.in_h, self.in_w, self.in_c = self.input_shape
        else:
            raise RuntimeError("Unsupported input shape: %s" % (self.input_shape,))

        # detect model type heuristically (SSD detection outputs often have 4 tensors)
        self.model_type = self._detect_model_type()

    def _detect_model_type(self):
        # Many detection models (SSD) have outputs: boxes, classes, scores, num_detections
        names = [o.get('name', '') for o in self.output_details]
        # Look for score-like and box-like outputs
        if len(self.output_details) >= 3:
            return "ssd"
        else:
            return "classification"

    def _preprocess(self, frame):
        # frame is RGB uint8 HxWx3
        img = cv2.resize(frame, (self.in_w, self.in_h))
        if np.issubdtype(self.in_dtype, np.floating):
            # models expecting float: normalize to [0,1] or [-1,1] depending on common patterns.
            # We'll normalize to [0,1]
            input_data = img.astype(np.float32) / 255.0
        else:
            # uint8 quantized model: ensure dtype=uint8
            input_data = img.astype(self.in_dtype)
        # add batch dim
        input_data = np.expand_dims(input_data, axis=0)
        return input_data

    def _run_inference(self, input_data):
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        outputs = {}
        for out in self.output_details:
            outputs[out['name']] = self.interpreter.get_tensor(out['index'])
        return outputs

    def run(self):
        writer = None
        try:
            if self.output_file:
                # video writer for annotated frames (BGR expected)
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                writer = cv2.VideoWriter(self.output_file, fourcc, 20.0, (self.in_w, self.in_h))
            while not self._stop.is_set():
                try:
                    frame = self.q.get(timeout=1.0)  # frame is RGB
                except queue.Empty:
                    continue

                input_data = self._preprocess(frame)

                outputs = self._run_inference(input_data)

                # parse outputs
                if self.model_type == "ssd":
                    # heuristics: try common names
                    # Candidate tensor names: 'StatefulPartitionedCall:0' etc. We'll simply index by order.
                    # Many TFLite SSD outputs order: boxes, classes, scores, num_detections
                    ovals = list(outputs.values())
                    if len(ovals) >= 3:
                        boxes = ovals[0][0]  # [N,4] normalized y1,x1,y2,x2
                        classes = ovals[1][0].astype(np.int32)
                        scores = ovals[2][0]
                        h, w = frame.shape[:2]
                        detections = []
                        for b, c, s in zip(boxes, classes, scores):
                            if s < self.score_threshold:
                                continue
                            y1, x1, y2, x2 = b
                            x1i, y1i, x2i, y2i = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
                            detections.append((x1i, y1i, x2i, y2i, c, float(s)))
                        # print detection summary
                        if detections:
                            print(f"[{time.strftime('%H:%M:%S')}] Detections: {detections}")
                        # annotate frame (convert RGB->BGR)
                        annotated = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        for (x1i, y1i, x2i, y2i, c, s) in detections:
                            cv2.rectangle(annotated, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
                            cv2.putText(annotated, f"{c}:{s:.2f}", (x1i, y1i-6),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                        if writer:
                            # resize annotated to model input dims to keep consistent size
                            writer.write(cv2.resize(annotated, (self.in_w, self.in_h)))
                else:
                    # classification / embedding
                    # assume single output probabilities vector
                    outputs_list = list(outputs.values())
                    preds = outputs_list[0][0]
                    topk = preds.argsort()[-5:][::-1]
                    # print top-1
                    print(f"[{time.strftime('%H:%M:%S')}] Top-1: idx={topk[0]} score={preds[topk[0]]:.3f}")
                    # optionally annotate and write
                    if writer:
                        annotated = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        cv2.putText(annotated, f"Top:{topk[0]} {preds[topk[0]]:.2f}", (10,20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                        writer.write(cv2.resize(annotated, (self.in_w, self.in_h)))
                # small sleep to yield
                time.sleep(0.001)
        finally:
            if writer:
                writer.release()

    def stop(self):
        self._stop.set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .tflite model")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output", default=None, help="Annotated output video file (optional)")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    q = queue.Queue(maxsize=1)  # hold latest frame only

    producer = CameraProducer(q, width=args.width, height=args.height, fps=args.fps)
    runner = TFLiteRunner(q, model_path=args.model, output_file=args.output,
                          score_threshold=args.threshold, save_annotated=bool(args.output))

    print("Starting camera producer and TFLite runner...")
    producer.start()
    runner.start()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        producer.stop()
        runner.stop()
        producer.join(timeout=2.0)
        runner.join(timeout=2.0)
        print("Exited cleanly.")


if __name__ == "__main__":
    main()
