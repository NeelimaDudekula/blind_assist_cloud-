from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import numpy as np
import cv2
import base64
import os

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

    frame = cv2.resize(frame, (640,480))

    results = model(frame, imgsz=320, conf=0.5)

    names = results[0].names
    boxes = results[0].boxes

    detected = []

    for box in boxes:

        confidence = float(box.conf[0])

        if confidence > 0.6:

            cls = int(box.cls[0])
            label = names[cls]

            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)

            box_width = (x2-x1)

            if box_width > 0:

                distance = round(400/box_width,2)

                if distance < 0.8:
                    distance_text = "very close"
                elif distance < 2:
                    distance_text = "near"
                else:
                    distance_text = "far"

                text = f"{label} {distance_text}"

                detected.append(text)

                # GREEN RECTANGLE
                cv2.rectangle(
                    frame,
                    (x1,y1),
                    (x2,y2),
                    (0,255,0),
                    2
                )

                # LABEL ABOVE BOX
                cv2.putText(
                    frame,
                    text,
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,0,255),
                    2
                )

    # convert frame to base64
    _,buffer = cv2.imencode('.jpg',frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "objects": list(set(detected)),
        "image":"data:image/jpeg;base64,"+frame_base64
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)