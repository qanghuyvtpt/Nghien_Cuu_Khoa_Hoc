# -*- coding: utf-8 -*-
from keras.models import load_model
from retinaface import RetinaFace
import numpy as np
import cv2
import os
import time


# === Video đầu vào ===
video_path = "/home/quanghuy/Downloads/football.mp4"    # đổi thành video của bạn

# === Tải FaceQnet ===
model = load_model('/media/quanghuy/New Volume/NCKH/mohinh/FaceQnet.h5')

# === Tạo thư mục output ===
output_dir = "output_faces"
os.makedirs(output_dir, exist_ok=True)

# ========================
#     ALIGN FACE
# ========================
def align_face(image, landmarks, out_size=256):
    left_eye = landmarks["left_eye"]
    right_eye = landmarks["right_eye"]
    mouth_left = landmarks["mouth_left"]
    mouth_right = landmarks["mouth_right"]

    mouth_center = (
        (mouth_left[0] + mouth_right[0]) / 2,
        (mouth_left[1] + mouth_right[1]) / 2
    )

    src_pts = np.float32([left_eye, right_eye, mouth_center])
    dst_pts = np.float32([
        [out_size * 0.3, out_size * 0.35],
        [out_size * 0.7, out_size * 0.35],
        [out_size * 0.5, out_size * 0.70],
    ])

    M = cv2.getAffineTransform(src_pts, dst_pts)
    aligned = cv2.warpAffine(image, M, (out_size, out_size))
    return aligned


# ========================
#      XỬ LÝ VIDEO
# ========================
cap = cv2.VideoCapture(video_path)
frame_count = 0
saved_count = 0

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        faces = RetinaFace.detect_faces(rgb)
    except:
        faces = {}

    if isinstance(faces, dict):
        for fid, face_info in faces.items():
            x1, y1, x2, y2 = face_info["facial_area"]
            landmarks = face_info["landmarks"]

            # ==== ALIGN FACE ====
            aligned_face = align_face(rgb, landmarks, out_size=256)
            face_224 = cv2.resize(aligned_face, (224, 224))

            # ==== DỰ ĐOÁN CHẤT LƯỢNG ====
            inp = np.expand_dims(face_224.astype(np.float32), axis=0)
            score = model.predict(inp, batch_size=1, verbose=0)[0][0]

            # ==== VẼ LÊN VIDEO ====
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            # ==== LƯU ẢNH FACE TỐT ====
            if score >= 0.50:
                save_path = os.path.join(
                    output_dir,
                    f"face_{saved_count}_s{score:.2f}_f{frame_count}.jpg"
                )
                cv2.imwrite(save_path, cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR))
                saved_count += 1

    # ==== HIỂN THỊ VIDEO ====
    cv2.imshow("Face Quality Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

print(f"\n======================")
print(f"Video xử lý xong!")
print(f"Số ảnh khuôn mặt tốt đã lưu: {saved_count}")
print(f"Folder: {output_dir}/")
print("======================")
