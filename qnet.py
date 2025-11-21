# -*- coding: utf-8 -*-

from keras.models import load_model
from retinaface import RetinaFace
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# === THAY ĐƯỜNG DẪN ẢNH Ở ĐÂY ===
image_path = "h2.png"

# === Tải mô hình đã huấn luyện ===
model = load_model('/media/quanghuy/New Volume/NCKH/mohinh/FaceQnet.h5')


# === Phát hiện và cắt nhiều khuôn mặt với RetinaFace ===
def detect_and_crop_faces(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # RetinaFace detection
    faces = RetinaFace.detect_faces(image_rgb)
    if isinstance(faces, dict) is False or len(faces) == 0:
        raise ValueError("Không phát hiện khuôn mặt trong ảnh!")

    cropped_faces = []
    face_locations = []

    # Lặp qua tất cả khuôn mặt
    for fid, face_info in faces.items():
        x1, y1, x2, y2 = face_info['facial_area']

        # Chống lỗi chỉ số âm
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = max(0, x2), max(0, y2)

        # Crop khuôn mặt
        face = image[y1:y2, x1:x2]

        if face.size == 0:
            continue

        # Resize về 224x224 cho FaceQnet
        face_resized = cv2.resize(face, (224, 224))

        cropped_faces.append(face_resized)
        face_locations.append((x1, y1, x2, y2))

    return image_rgb, cropped_faces, face_locations


# === Xử lý ảnh ===
original_image, faces, locations = detect_and_crop_faces(image_path)

scores = []

# Dự đoán chất lượng cho từng khuôn mặt
for face_img in faces:
    face_input = np.expand_dims(face_img.astype(np.float32), axis=0)
    score = model.predict(face_input, batch_size=1, verbose=0)[0][0]
    scores.append(score)

# === Vẽ kết quả ===
for i, ((x1, y1, x2, y2), score) in enumerate(zip(locations, scores)):
    cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(original_image, f"Face {i+1}: {score:.4f}",
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 0), 2)

plt.imshow(original_image)
plt.title("Chất lượng khuôn mặt")
plt.axis('off')
plt.show()

# === In báo cáo kết quả ===
print("===== KẾT QUẢ ĐÁNH GIÁ =====")
print(f"Tên ảnh: {os.path.basename(image_path)}\n")

for i, score in enumerate(scores):
    print(f"Khuan mat {i+1}: Diem chat luong = {score:.4f}")
