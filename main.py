# Importation
from ultralytics import YOLO
import cv2 as cv
import cvzone
import math
from sort import *

# Initialization and variable naming
model = YOLO("path/to/yolov8l.pt")
vid = cv.VideoCapture("assets/traffic_cam.mp4")

class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

tracker = Sort(max_age = 22, min_hits = 3, iou_threshold = 0.3)
line_up = [180, 410, 640, 410]
line_down = [680, 400, 1280, 450]
count_up = []
count_down = []
total_count = []
mask = cv.imread("assets/mask.png") # For blocking out noise

# Setting up video writer properties (for saving the output result)
width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = vid.get(cv.CAP_PROP_FPS)
video_writer = cv.VideoWriter(("result.mp4"), cv.VideoWriter_fourcc("m", "p", "4", "v"),
                              fps, (width, height))

while True:
    ref, frame = vid.read()
    frame_region = cv.bitwise_and(frame, mask)
    result = model(frame_region, stream=True)

    # Total count graphics
    frame_graphics = cv.imread("assets/graphics.png", cv.IMREAD_UNCHANGED)
    frame = cvzone.overlayPNG(frame, frame_graphics, (0,0))

    # Vehicle count graphics
    frame_graphics1 = cv.imread("assets/graphics1.png", cv.IMREAD_UNCHANGED)
    frame = cvzone.overlayPNG(frame, frame_graphics1, (420,0))

    detections = np.empty((0, 5))

    for r in result:
        boxes = r.boxes
        for box in boxes:
            # Bounding boxes
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = (x2-x1), (y2-y1)

            #Detection confidence
            conf = math.floor(box.conf[0]*100)/100

            # Class names
            cls = int(box.cls[0])
            vehicle_names = class_names[cls]

            if vehicle_names == "car" or vehicle_names == "truck" or vehicle_names == "bus"\
                or vehicle_names == "motorbike":
                current_detection = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current_detection))

    # Tracking codes
    tracker_updates = tracker.update(detections)
    # Tracking lines
    cv.line(frame, (line_up[0], line_up[1]), (line_up[2], line_up[3]), (0, 0, 255), thickness = 3)
    cv.line(frame, (line_down[0] ,line_down[1]), (line_down[2], line_down[3]), (0, 0, 255), thickness = 3)

    # Geting bounding boxes points and vehicle ID
    for update in tracker_updates:
        x1, y1, x2, y2, id = update
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        w, h = (x2-x1), (y2-y1)

        # Getting tracking marker
        cx, cy = (x1+w//2), (y1+h//2)
        cv.circle(frame, (cx, cy), 5, (255, 0, 255), cv.FILLED)

        # Code for left lane (vehicles driving up)
        if line_up[0] < cx < line_up[2] and line_up[1] - 5 < cy < line_up[3] + 5:
            if total_count.count(id) == 0:
                total_count.append(id)
                cv.line(frame, (line_up[0], line_up[1]), (line_up[2], line_up[3]), (0, 255, 0), thickness = 3)

                if count_up.count(id) == 0:
                    count_up.append(id)

        # Code for right lane (vehicles driving down)
        if line_down[0] < cx < line_down[2] and line_down[1] + 15 < cy < line_down[3] + 15:
            if total_count.count(id) == 0:
                total_count.append(id)
                cv.line(frame, (line_down[0], line_down[1]), (line_down[2], line_down[3]), (0, 255, 0), thickness = 3)

                if count_down.count(id) == 0:
                    count_down.append(id)

        # Adding rectangles and texts
        cvzone.cornerRect(frame, (x1, y1, w, h), l=5, colorR=(255, 0, 255), rt=1)
        cvzone.putTextRect(frame, f'{id}', (x1, y1), scale=1, thickness=2)

    # Adding texts to graphics
    cv.putText(frame, str(len(total_count)), (255, 100), cv.FONT_HERSHEY_PLAIN, 5, (200, 50, 200), thickness=7)
    cv.putText(frame, str(len(count_up)), (600, 85), cv.FONT_HERSHEY_PLAIN, 5, (200, 50, 200), thickness=7)
    cv.putText(frame, str(len(count_down)), (850, 85), cv.FONT_HERSHEY_PLAIN, 5, (200, 50, 200), thickness=7)

    cv.imshow("vid", frame)

    # Saving the video frame output
    video_writer.write(frame)

    cv.waitKey(1)

# Closing down everything
vid.release()
cv.destroyAllWindows()
video_writer.release()