from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import numpy as np
import cv2
import base64

app = Flask(__name__)

model = YOLO("yolov8n.pt")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    data = request.json.get('image')
    if not data:
        return jsonify({"objects": []})

    image_data = base64.b64decode(data.split(',')[1])
    np_arr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Resize for cloud speed
    frame = cv2.resize(frame, (640, 480))

    results = model(frame, imgsz=416)

    names = results[0].names
    boxes = results[0].boxes

    detected = []

    for box in boxes:
        confidence = float(box.conf[0])

        if confidence > 0.6:
            cls = int(box.cls[0])
            label = names[cls]

            x1, y1, x2, y2 = box.xyxy[0]
            box_width = (x2 - x1).item()

            if box_width > 0:
                distance = round(400 / box_width, 2)

                if distance < 0.8:
                    distance_text = "very close"
                elif distance < 2:
                    distance_text = "near"
                else:
                    distance_text = "far"

                detected.append(f"{label} {distance_text}")

    return jsonify({"objects": list(set(detected))})


if __name__ == "__main__":
    app.run(debug=True)