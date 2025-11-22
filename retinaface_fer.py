import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from retinaface import RetinaFace

# ===== Config =====
train_model = "ResNet"
if train_model == "Inception":
    img_width, img_height = 139, 139
elif train_model == "ResNet":
    img_width, img_height = 197, 197

emotions = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']

# ===== Load mô hình đã huấn luyện =====
model_path = "/media/quanghuy/New Volume/NCKH/mohinh/ResNet-50.h5"
model = load_model(model_path)

# ===== Tiền xử lý ảnh cho ResNet/Inception =====
def preprocess_input(image):
    image = cv2.resize(image, (img_width, img_height))
    x = np.expand_dims(image.astype(np.float32), axis=0)

    if train_model == "Inception":
        x /= 127.5
        x -= 1.
    elif train_model == "ResNet":
        x -= 128.8006   # mean
        x /= 64.6497    # std

    return x

# ===== Dự đoán cảm xúc =====
def predict_emotion(face_img):
    x = preprocess_input(face_img)
    prediction = model.predict(x)[0]  # (7,)
    return prediction

# ===== Xử lý ảnh đầu vào =====
image_path = "/home/quanghuy/Documents/code_NCKH/h7.png"  
frame = cv2.imread(image_path)

if frame is None:
    raise ValueError("Không thể đọc ảnh, kiểm tra lại đường dẫn!")

frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# ===== Phát hiện khuôn mặt bằng RetinaFace =====
faces = RetinaFace.detect_faces(frame_rgb)

for key in faces.keys():
    face_info = faces[key]
    # Lấy bounding box [x1, y1, x2, y2]
    x1, y1, x2, y2 = face_info['facial_area']
    x1, y1 = max(0, x1), max(0, y1)

    # Cắt khuôn mặt
    face = frame_rgb[y1:y2, x1:x2]

    # Dự đoán cảm xúc
    prediction = predict_emotion(face)

    # Lấy top-3 cảm xúc
    top_indices = prediction.argsort()[-3:][::-1]
    for i, idx in enumerate(top_indices):
        label = emotions[idx]
        conf = prediction[idx] * 100
        text = f"{label}: {conf:.1f}%"
        cv2.putText(frame_rgb, text, (x1, y2 + 30 + i*30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Vẽ bounding box
    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

# ===== Hiển thị ảnh kết quả =====
plt.imshow(frame_rgb)
plt.axis("off")
plt.show()
