"""Facial recognition: Haar cascade detection + a KNN classifier over
flattened grayscale crops, exactly as validated in the thesis (Chapter 3.2,
3.6) and measured in Chapter 4.2 (85% accuracy under normal conditions).
"""

import os

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from . import config


class FaceRecognizer:
    def __init__(self, data_dir=config.FACE_DATA_DIR, cascade_path=config.CASCADE_PATH):
        self.data_dir = str(data_dir)
        os.makedirs(self.data_dir, exist_ok=True)

        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Haar cascade file not found: {cascade_path}")
        self.detector = cv2.CascadeClassifier(str(cascade_path))

        self.faces, self.labels = self.load_data()
        self.knn = KNeighborsClassifier(n_neighbors=config.FACE_KNN_NEIGHBORS)
        if len(self.labels) > 0:
            self.knn.fit(self.faces, self.labels)

    @property
    def is_ready(self):
        return len(self.labels) > 0

    def load_data(self):
        """Load every enrolled face image from data_dir/<name>/*.jpg."""
        faces, labels = [], []
        for label in os.listdir(self.data_dir):
            person_dir = os.path.join(self.data_dir, label)
            if not os.path.isdir(person_dir):
                continue
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, config.FACE_IMAGE_SIZE).flatten()
                faces.append(img)
                labels.append(label)
        return np.array(faces), np.array(labels)

    def refresh(self):
        """Reload enrolled faces and refit the classifier. Returns is_ready."""
        self.faces, self.labels = self.load_data()
        if self.is_ready:
            self.knn.fit(self.faces, self.labels)
        return self.is_ready

    def detect(self, gray_frame):
        return self.detector.detectMultiScale(
            gray_frame,
            scaleFactor=config.FACE_DETECT_SCALE_FACTOR,
            minNeighbors=config.FACE_DETECT_MIN_NEIGHBORS,
        )

    def predict(self, gray_crop):
        """Predict the enrolled name for a grayscale face crop, or None."""
        resized = cv2.resize(gray_crop, config.FACE_IMAGE_SIZE).flatten().reshape(1, -1)
        output = self.knn.predict(resized)[0]
        return str(output) if output in self.labels else None

    def enroll(self, name, logger=lambda message: None, max_images=config.FACE_ENROLL_MAX_IMAGES):
        """Capture up to max_images webcam shots of `name`'s face, save them,
        and refit the classifier. Opens its own OpenCV preview window;
        press 'q' to stop early."""
        user_dir = os.path.join(self.data_dir, name)
        os.makedirs(user_dir, exist_ok=True)

        cap = cv2.VideoCapture(0)
        count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                for (x, y, w, h) in self.detect(gray):
                    count += 1
                    face_img = cv2.resize(gray[y:y + h, x:x + w], config.FACE_IMAGE_SIZE)
                    cv2.imwrite(os.path.join(user_dir, f"{name}_{count}.jpg"), face_img)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, str(count), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    logger(f"Image {count} capturée")

                cv2.imshow("Enrolement du visage - Appuyez sur 'q' pour quitter", frame)
                if cv2.waitKey(1) & 0xFF == ord("q") or count >= max_images:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

        return self.refresh()

    def run_live_recognition(self, should_continue, logger=lambda message: None):
        """Blocking loop: shows a webcam preview, detects and labels faces
        until should_continue() returns False or 'q' is pressed."""
        video = cv2.VideoCapture(0)
        try:
            while should_continue():
                ret, frame = video.read()
                if not ret:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                for (x, y, w, h) in self.detect(gray):
                    name = self.predict(gray[y:y + h, x:x + w]) or "Inconnu(e)"
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    logger(f"Visage reconnu : {name}")

                cv2.imshow("Reconnaissance Faciale - Appuyez sur 'q' pour quitter", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            video.release()
            cv2.destroyAllWindows()
