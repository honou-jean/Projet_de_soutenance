import os
import cv2
import numpy as np
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from PIL import Image
import easyocr
import torch
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import sounddevice as sd
import wavio
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Charger le classificateur de visages OpenCV
cascade_path = r"C:\cours_openclassrom\projetStage\haarcascade_frontalface_default.xml"
if not os.path.exists(cascade_path):
    print("Erreur : Impossible de trouver le fichier haarcascade.")
    exit()
facedetect = cv2.CascadeClassifier(cascade_path)

# Classe pour l'application GUI
class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Reconnaissance Multimodale avec OCR et Wav2Vec")
        self.geometry("800x700")
        self.configure(bg="#1e272e")
        self.create_widgets()

        # Initialisation de la reconnaissance faciale
        self.face_data_dir = 'face_data'
        os.makedirs(self.face_data_dir, exist_ok=True)
        self.faces, self.labels = self.load_face_data()
        self.knn_face = KNeighborsClassifier(n_neighbors=5)
        if len(self.labels) > 0:
            self.knn_face.fit(self.faces, self.labels)
        else:
            self.display_message("Aucune donnée de visage disponible.")

        # Initialisation de Wav2Vec2
        self.processor, self.wav2vec_model = self.initialize_wav2vec()

        # Initialisation de la reconnaissance vocale
        self.voice_data_dir = 'voice_data'
        os.makedirs(self.voice_data_dir, exist_ok=True)
        self.voice_embeddings, self.voice_labels = self.enroll_speakers(self.voice_data_dir)
        self.le = LabelEncoder()
        if len(self.voice_labels) > 0:
            self.voice_labels_encoded = self.le.fit_transform(self.voice_labels)
            self.knn_voice = KNeighborsClassifier(n_neighbors=3)
            self.knn_voice.fit(self.voice_embeddings, self.voice_labels_encoded)
        else:
            self.knn_voice = None
            self.display_message("Aucune donnée vocale disponible.")

        # Initialiser l'OCR
        self.reader = easyocr.Reader(['fr'], gpu=False)

        self.face_recognition_running = False
        self.voice_recognition_running = False

    def create_widgets(self):
        title_label = tk.Label(
            self,
            text="BioID Vision Reconnaissance multimodale",
            font=("Helvetica", 18, "bold"),
            bg="#1e272e",
            fg="white",
        )
        title_label.pack(pady=20)

        # Cadre pour les boutons
        button_frame = tk.Frame(self, bg="#1e272e")
        button_frame.pack(pady=20)

        self.start_face_button = ttk.Button(
            button_frame,
            text="Démarrer Reconnaissance Faciale",
            command=self.start_face_recognition,
        )
        self.start_face_button.grid(row=0, column=0, padx=10, pady=10)

        self.stop_face_button = ttk.Button(
            button_frame,
            text="Arrêter Reconnaissance Faciale",
            command=self.stop_face_recognition,
            state=tk.DISABLED,
        )
        self.stop_face_button.grid(row=0, column=1, padx=10, pady=10)

        self.start_voice_button = ttk.Button(
            button_frame,
            text="Démarrer Reconnaissance Vocale",
            command=self.start_voice_recognition,
        )
        self.start_voice_button.grid(row=1, column=0, padx=10, pady=10)

        self.stop_voice_button = ttk.Button(
            button_frame,
            text="Arrêter Reconnaissance Vocale",
            command=self.stop_voice_recognition,
            state=tk.DISABLED,
        )
        self.stop_voice_button.grid(row=1, column=1, padx=10, pady=10)

        self.import_ocr_button = ttk.Button(
            button_frame,
            text="Importer une Image pour OCR",
            command=self.run_ocr_import,
        )
        self.import_ocr_button.grid(row=2, column=0, padx=10, pady=10)

        self.capture_ocr_button = ttk.Button(
            button_frame,
            text="Capturer une Image pour OCR",
            command=self.run_ocr_capture,
        )
        self.capture_ocr_button.grid(row=2, column=1, padx=10, pady=10)

        self.enroll_face_button = ttk.Button(
            button_frame,
            text="Enrôler un nouveau visage",
            command=self.enroll_face,
        )
        self.enroll_face_button.grid(row=3, column=0, padx=10, pady=10)

        self.enroll_voice_button = ttk.Button(
            button_frame,
            text="Enrôler une nouvelle voix",
            command=self.enroll_voice,
        )
        self.enroll_voice_button.grid(row=3, column=1, padx=10, pady=10)

        # Zone de texte pour afficher les résultats
        self.text_area = tk.Text(
            self,
            height=15,
            width=80,
            bg="#dcdde1",
            fg="#2f3640",
            font=("Helvetica", 12),
        )
        self.text_area.pack(pady=20)

        self.save_button = ttk.Button(
            self,
            text="Enregistrer dans Excel",
            command=self.save_to_excel
        )
        self.save_button.pack(pady=10)

    def display_message(self, message):
        self.text_area.insert(tk.END, f"{message}\n")
        self.text_area.see(tk.END)

    # Fonction pour charger les données de visages
    def load_face_data(self):
        faces = []
        labels = []

        for label in os.listdir(self.face_data_dir):
            person_dir = os.path.join(self.face_data_dir, label)
            if not os.path.isdir(person_dir):
                continue
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (50, 50))  # Redimensionner à 50x50 pixels
                img = img.flatten()
                faces.append(img)
                labels.append(label)

        return np.array(faces), np.array(labels)

    # Fonction pour initialiser Wav2Vec2
    def initialize_wav2vec(self):
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        return processor, model

    # Fonction pour extraire les embeddings vocaux avec Wav2Vec2
    def extract_voice_embedding(self, audio_path):
        speech_array, sampling_rate = torchaudio.load(audio_path)

        if speech_array.shape[0] > 1:
            speech_array = torch.mean(speech_array, dim=0, keepdim=True)

        if sampling_rate != 16000:
            resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
            speech_array = resampler(speech_array)

        input_values = self.processor(speech_array.squeeze(0), sampling_rate=16000, return_tensors="pt", padding=True).input_values

        with torch.no_grad():
            embeddings = self.wav2vec_model(input_values).last_hidden_state.mean(dim=1)

        return embeddings.numpy()[0]

    # Fonction pour enregistrer les locuteurs pour la reconnaissance vocale
    def enroll_speakers(self, voice_data_dir):
        embeddings = []
        labels = []

        if not os.path.exists(voice_data_dir):
            print(f"Erreur : Le dossier {voice_data_dir} n'existe pas.")
            return np.array([]), np.array([])

        for person_name in os.listdir(voice_data_dir):
            person_dir = os.path.join(voice_data_dir, person_name)
            if not os.path.isdir(person_dir):
                continue

            audio_files = [os.path.join(person_dir, f) for f in os.listdir(person_dir) if f.endswith('.wav')]
            if not audio_files:
                print(f"Aucun fichier audio trouvé pour {person_name}")
                continue

            for audio_file in audio_files:
                embedding = self.extract_voice_embedding(audio_file)
                if embedding is not None:
                    embeddings.append(embedding)
                    labels.append(person_name)

        return np.array(embeddings), np.array(labels)

    # Fonction pour enrôler un nouveau visage
    def enroll_face(self):
        name = simpledialog.askstring("Enrôlement du visage", "Entrez le nom de l'utilisateur :")
        if not name:
            return

        user_dir = os.path.join(self.face_data_dir, name)
        os.makedirs(user_dir, exist_ok=True)

        cap = cv2.VideoCapture(0)
        count = 0
        max_images = 50

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                count += 1
                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (50, 50))
                file_path = os.path.join(user_dir, f'{name}_{count}.jpg')
                cv2.imwrite(file_path, face_img)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, str(count), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                self.display_message(f"Image {count} capturée")

            cv2.imshow("Enrôlement du visage - Appuyez sur 'q' pour quitter", frame)

            if cv2.waitKey(1) & 0xFF == ord('q') or count >= max_images:
                break

        cap.release()
        cv2.destroyAllWindows()

        self.update_face_model()

    # Fonction pour mettre à jour le modèle de reconnaissance faciale
    def update_face_model(self):
        self.faces, self.labels = self.load_face_data()
        if len(self.labels) > 0:
            self.knn_face.fit(self.faces, self.labels)
            self.display_message("Modèle de reconnaissance faciale mis à jour")
        else:
            self.display_message("Aucune donnée de visage disponible.")

    # Fonction pour enrôler une nouvelle voix
    def enroll_voice(self):
        name = simpledialog.askstring("Enrôlement de la voix", "Entrez le nom de l'utilisateur :")
        if not name:
            return

        user_dir = os.path.join(self.voice_data_dir, name)
        os.makedirs(user_dir, exist_ok=True)

        num_samples = 3
        duration = 5
        fs = 16000

        for i in range(1, num_samples + 1):
            messagebox.showinfo("Enregistrement vocal", f"Préparez-vous pour l'enregistrement {i}")
            self.display_message(f"Enregistrement {i} en cours...")
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
            sd.wait()
            filename = os.path.join(user_dir, f'audio{i}.wav')
            wavio.write(filename, recording, fs, sampwidth=2)
            self.display_message(f"Enregistrement {i} terminé")

        self.update_voice_model()

    # Fonction pour mettre à jour le modèle de reconnaissance vocale
    def update_voice_model(self):
        self.voice_embeddings, self.voice_labels = self.enroll_speakers(self.voice_data_dir)
        if len(self.voice_labels) > 0:
            self.voice_labels_encoded = self.le.fit_transform(self.voice_labels)
            self.knn_voice = KNeighborsClassifier(n_neighbors=3)
            self.knn_voice.fit(self.voice_embeddings, self.voice_labels_encoded)
            self.display_message("Modèle de reconnaissance vocale mis à jour")
        else:
            self.display_message("Erreur : Aucun locuteur enregistré")
            self.knn_voice = None

    # Fonctions pour la reconnaissance faciale
    def start_face_recognition(self):
        if not self.face_recognition_running:
            if len(self.labels) == 0:
                messagebox.showerror("Erreur", "Aucune donnée de visage disponible.")
                return
            self.face_recognition_running = True
            self.start_face_button.config(state=tk.DISABLED)
            self.stop_face_button.config(state=tk.NORMAL)
            self.face_thread = threading.Thread(target=self.face_recognition)
            self.face_thread.start()

    def stop_face_recognition(self):
        if self.face_recognition_running:
            self.face_recognition_running = False
            self.start_face_button.config(state=tk.NORMAL)
            self.stop_face_button.config(state=tk.DISABLED)

    def face_recognition(self):
        video = cv2.VideoCapture(0)
        while self.face_recognition_running:
            ret, frame = video.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_detected = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces_detected:
                crop_img = gray[y:y + h, x:x + w]
                resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
                output = self.knn_face.predict(resized_img)

                name = str(output[0]) if output[0] in self.labels else "Inconnu(e)"

                # Dessiner un rectangle autour du visage et afficher le nom
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
                cv2.putText(frame, name, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                self.display_message(f"Visage reconnu : {name}")

            cv2.imshow("Reconnaissance Faciale - Appuyez sur 'q' pour quitter", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.face_recognition_running = False
                break

        video.release()
        cv2.destroyAllWindows()
        self.face_recognition_running = False
        self.start_face_button.config(state=tk.NORMAL)
        self.stop_face_button.config(state=tk.DISABLED)

    # Fonctions pour la reconnaissance vocale
    def start_voice_recognition(self):
        if not self.voice_recognition_running:
            if self.knn_voice is None:
                messagebox.showerror("Erreur", "Le classificateur vocal n'est pas disponible.")
                return
            self.voice_recognition_running = True
            self.start_voice_button.config(state=tk.DISABLED)
            self.stop_voice_button.config(state=tk.NORMAL)
            self.voice_thread = threading.Thread(target=self.voice_recognition_loop)
            self.voice_thread.start()

    def stop_voice_recognition(self):
        if self.voice_recognition_running:
            self.voice_recognition_running = False
            self.start_voice_button.config(state=tk.NORMAL)
            self.stop_voice_button.config(state=tk.DISABLED)

    def voice_recognition_loop(self):
        fs = 16000
        duration = 5

        while self.voice_recognition_running:
            # Prévenir l'utilisateur de parler
            ready_to_speak = messagebox.askokcancel("Reconnaissance vocale", "Appuyez sur OK et commencez à parler.")
            if not ready_to_speak:
                self.display_message("Reconnaissance vocale annulée.")
                return

            # Enregistrer l'audio
            self.display_message("Enregistrement audio en cours...")
            audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
            sd.wait()
            temp_audio_path = "temp_audio.wav"
            wavio.write(temp_audio_path, audio, fs, sampwidth=2)

            # Extraire l'embedding et prédire le locuteur
            try:
                embedding = self.extract_voice_embedding(temp_audio_path)
                predicted_label = self.knn_voice.predict([embedding])[0]
                recognized_speaker = self.le.inverse_transform([predicted_label])[0]
                name = recognized_speaker if recognized_speaker in self.voice_labels else "Inconnu(e)"
                self.display_message(f"Locuteur reconnu : {name}")
            except Exception as e:
                self.display_message(f"Erreur lors de la reconnaissance vocale : {e}")

            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

            if not self.voice_recognition_running:
                break

        self.voice_recognition_running = False
        self.start_voice_button.config(state=tk.NORMAL)
        self.stop_voice_button.config(state=tk.DISABLED)

    # Fonctions pour l'OCR
    def run_ocr_import(self):
        image_path = filedialog.askopenfilename(
            title="Sélectionnez une image", filetypes=[("Images", "*.jpg;*.jpeg;*.png")]
        )
        if image_path:
            self.display_message("Extraction du texte en cours...")
            try:
                extracted_text = self.ocr_image(image_path)
                self.display_message("Texte extrait :")
                for text in extracted_text:
                    self.display_message(f"- {text}")
            except Exception as e:
                messagebox.showerror("Erreur OCR", f"Une erreur est survenue : {e}")

    def run_ocr_capture(self):
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            messagebox.showerror("Erreur", "Impossible d'accéder à la webcam.")
            return
        while True:
            ret, frame = video_capture.read()
            if not ret:
                continue
            cv2.imshow('Capture pour OCR - Appuyez sur "c" pour capturer', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                temp_image_path = "temp_capture.jpg"
                cv2.imwrite(temp_image_path, frame)
                cv2.destroyAllWindows()
                break
            elif key == ord('q'):
                cv2.destroyAllWindows()
                video_capture.release()
                return
        video_capture.release()
        try:
            extracted_text = self.ocr_image(temp_image_path)
            self.display_message("Texte extrait :")
            for text in extracted_text:
                self.display_message(f"- {text}")
            os.remove(temp_image_path)
        except Exception as e:
            messagebox.showerror("Erreur OCR", f"Une erreur est survenue : {e}")

    def ocr_image(self, image_path):
        image = Image.open(image_path)
        img_array = np.array(image.convert('RGB'))
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        max_dimension = 1000
        height, width = img.shape[:2]
        if max(height, width) > max_dimension:
            scaling_factor = max_dimension / max(height, width)
            img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = self.reader.readtext(gray)
        extracted_text = [detection[1] for detection in result]
        return extracted_text

    # Fonction pour enregistrer les données dans Excel
    def save_to_excel(self):
        data = self.text_area.get("1.0", tk.END).strip().split("\n")
        if not data:
            messagebox.showwarning("Avertissement", "Aucune donnée à enregistrer.")
            return
        df = pd.DataFrame(data, columns=["Informations"])
        filepath = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
        if filepath:
            df.to_excel(filepath, index=False)
            messagebox.showinfo("Succès", "Données enregistrées avec succès dans Excel.")

# Fonction principale
def main():
    app = Application()
    app.mainloop()

if __name__ == "__main__":
    main()
