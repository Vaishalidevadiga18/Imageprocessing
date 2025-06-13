import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

def show_image(img, title="Image"):
    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 6))
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()

def detect_with_haar(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    show_image(img, "Haar Cascade Detection")

def detect_with_hog(image_path):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    img = cv2.imread(image_path)
    img = cv2.resize(img, (640, 480))  # Resize for better speed

    (boxes, weights) = hog.detectMultiScale(img, winStride=(8, 8), padding=(8, 8), scale=1.05)

    for (x, y, w, h) in boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    show_image(img, "HOG + SVM Detection")

def detect_with_yolo(image_path):
    model = YOLO('yolov8n.pt')  # Automatically downloads on first use
    results = model(image_path)
    annotated_img = results[0].plot()
    show_image(annotated_img, "YOLOv8 Detection")

# --------------- MAIN ----------------
if __name__ == "__main__":
    image_path = r"C:\Users\HP\OneDrive\Pictures\yolo.jpg"  # Change this to your actual image path

    print("Choose detection method:")
    print("1 - Haar Cascades")
    print("2 - HOG + SVM")
    print("3 - YOLOv8")

    choice = input("Enter your choice (1/2/3): ").strip()

    if choice == '1':
        detect_with_haar(image_path)
    elif choice == '2':
        detect_with_hog(image_path)
    elif choice == '3':
        detect_with_yolo(image_path)
    else:
        print("Invalid choice!")