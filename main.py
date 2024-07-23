import cvzone
from ultralytics import YOLO
import cv2
import math

# array for classification
classNames = ['Gloves', 'Helmet', 'Non-Helmet', 'Person', 'Shoes', 'Vest', 'bare-arms']

# Initialize the webcam
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('8964796-uhd_3840_2160_25fps.mp4')

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("2: Webcam initialized")

# Set webcam properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("3: Webcam properties set")

try:
    model = YOLO("ppe.pt")
    print("4: YOLO model loaded")
except Exception as e:
    print(f"Error: Failed to load YOLO model: {e}")
    exit()

while True:
    success, img = cap.read()
    if not success:
        print("Error: Failed to read frame.")
        continue  # Skip this iteration

    print("5: Frame captured")

    try:
        results = model(img, stream=True)
        # print(results, 'result')
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 3)
                w, h = x2 - x1, y2 - y1

                # confidence
                conf = math.ceil(box.conf[0] * 100) / 100
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                # if currentClass == 'person' and conf > 0.5:
                cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)), scale=2, thickness=2,
                                   offset=3)
                cvzone.cornerRect(img, (x1, y1, w, h), l=10)

    except Exception as e:
        print(f"Error: YOLO processing failed: {e}")
        break

    # Resize the frame
    # new_width = 1000
    # new_height = 600
    # img_resized = cv2.resize(img, (new_width, new_height))

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("6: Cleanup complete")
