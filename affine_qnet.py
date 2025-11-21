# -*- coding: utf-8 -*-
from keras.models import load_model
from retinaface import RetinaFace
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# === THAY ĐƯỜNG DẪN ẢNH Ở ĐÂY ===
image_path = "h2.png"

# === Tải model FaceQnet ===
model = load_model('/media/quanghuy/New Volume/NCKH/mohinh/FaceQnet.h5')


# === ALIGN FACE BẰNG 5 LANDMARKS CỦA RETINAFACE ===
def align_face(image, landmarks, out_size=256):
    left_eye = landmarks["left_eye"]
    right_eye = landmarks["right_eye"]
    mouth_left = landmarks["mouth_left"]
    mouth_right = landmarks["mouth_right"]

    mouth_center = (
        (mouth_left[0] + mouth_right[0]) / 2,
        (mouth_left[1] + mouth_right[1]) / 2
    )

    src_pts = np.float32([
        left_eye,
        right_eye,
        mouth_center
    ])

    dst_pts = np.float32([
        [out_size * 0.3, out_size * 0.35],
        [out_size * 0.7, out_size * 0.35],
        [out_size * 0.5, out_size * 0.70]
    ])

    M = cv2.getAffineTransform(src_pts, dst_pts)
    aligned = cv2.warpAffine(image, M, (out_size, out_size))

    return aligned


# === PHÁT HIỆN + CĂN CHỈNH + TRÍCH KHUÔN MẶT ===
def detect_crop_align_faces(image_path, out_size=256):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Không thể đọc ảnh!")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = RetinaFace.detect_faces(image_rgb)

    if not isinstance(faces, dict) or len(faces) == 0:
        raise ValueError("Không phát hiện khuôn mặt nào!")

    aligned_faces = []
    locations = []

    for fid, face_info in faces.items():
        x1, y1, x2, y2 = face_info["facial_area"]
        landmarks = face_info["landmarks"]

        aligned = align_face(image_rgb, landmarks, out_size=out_size)
        aligned_224 = cv2.resize(aligned, (224, 224))

        aligned_faces.append(aligned_224)
        locations.append((x1, y1, x2, y2))

    return image_rgb, aligned_faces, locations


# === THỰC THI ===
original_image, aligned_faces, locations = detect_crop_align_faces(image_path)

face_scores = []

for face in aligned_faces:
    face_input = np.expand_dims(face.astype(np.float32), axis=0)
    score = model.predict(face_input, batch_size=1, verbose=0)[0][0]
    face_scores.append(score)


# === HIỂN THỊ ẢNH GỐC + KẾT QUẢ ===
for (x1, y1, x2, y2), score in zip(locations, face_scores):
    cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(original_image, f"{score:.4f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

plt.figure(figsize=(10, 6))
plt.imshow(original_image)
plt.title("Ảnh gốc + điểm chất lượng")
plt.axis("off")
plt.show()


# === HIỂN THỊ TỪNG ẢNH ĐÃ AFFINE ===
cols = 5
rows = int(np.ceil(len(aligned_faces) / cols))
plt.figure(figsize=(15, 3 * rows))

for i, face in enumerate(aligned_faces):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(face)
    plt.title(f"Face {i+1}\nScore={face_scores[i]:.4f}")
    plt.axis("off")

plt.tight_layout()
plt.show()


# === IN KẾT QUẢ ===
print("\n===== KẾT QUẢ =====")
print(f"Tên ảnh: {os.path.basename(image_path)}\n")

for i, score in enumerate(face_scores):
    print(f"Khuôn mặt {i+1}: Điểm chất lượng = {score:.4f}")
