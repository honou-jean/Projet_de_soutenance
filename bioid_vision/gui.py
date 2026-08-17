"""Tkinter application: wires the three recognizers together behind the
"BioID Vision" interface described in the thesis (Chapter 4.5).
"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2

from . import config
from .export import export_lines_to_excel
from .face_recognition import FaceRecognizer
from .ocr import TextExtractor
from .voice_recognition import VoiceRecognizer


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Reconnaissance Multimodale avec OCR et Wav2Vec")
        self.geometry("800x700")
        self.configure(bg="#1e272e")
        self.create_widgets()

        self.face_recognizer = FaceRecognizer()
        if not self.face_recognizer.is_ready:
            self.display_message("Aucune donnee de visage disponible.")

        self.voice_recognizer = VoiceRecognizer()
        if not self.voice_recognizer.is_ready:
            self.display_message("Aucune donnee vocale disponible.")

        self.text_extractor = TextExtractor()

        self.face_recognition_running = False
        self.voice_recognition_running = False

    # -- layout -----------------------------------------------------------

    def create_widgets(self):
        title_label = tk.Label(
            self, text="BioID Vision Reconnaissance multimodale",
            font=("Helvetica", 18, "bold"), bg="#1e272e", fg="white",
        )
        title_label.pack(pady=20)

        button_frame = tk.Frame(self, bg="#1e272e")
        button_frame.pack(pady=20)

        self.start_face_button = ttk.Button(
            button_frame, text="Demarrer Reconnaissance Faciale", command=self.start_face_recognition,
        )
        self.start_face_button.grid(row=0, column=0, padx=10, pady=10)

        self.stop_face_button = ttk.Button(
            button_frame, text="Arreter Reconnaissance Faciale",
            command=self.stop_face_recognition, state=tk.DISABLED,
        )
        self.stop_face_button.grid(row=0, column=1, padx=10, pady=10)

        self.start_voice_button = ttk.Button(
            button_frame, text="Demarrer Reconnaissance Vocale", command=self.start_voice_recognition,
        )
        self.start_voice_button.grid(row=1, column=0, padx=10, pady=10)

        self.stop_voice_button = ttk.Button(
            button_frame, text="Arreter Reconnaissance Vocale",
            command=self.stop_voice_recognition, state=tk.DISABLED,
        )
        self.stop_voice_button.grid(row=1, column=1, padx=10, pady=10)

        self.import_ocr_button = ttk.Button(
            button_frame, text="Importer une Image pour OCR", command=self.run_ocr_import,
        )
        self.import_ocr_button.grid(row=2, column=0, padx=10, pady=10)

        self.capture_ocr_button = ttk.Button(
            button_frame, text="Capturer une Image pour OCR", command=self.run_ocr_capture,
        )
        self.capture_ocr_button.grid(row=2, column=1, padx=10, pady=10)

        self.enroll_face_button = ttk.Button(
            button_frame, text="Enroler un nouveau visage", command=self.enroll_face,
        )
        self.enroll_face_button.grid(row=3, column=0, padx=10, pady=10)

        self.enroll_voice_button = ttk.Button(
            button_frame, text="Enroler une nouvelle voix", command=self.enroll_voice,
        )
        self.enroll_voice_button.grid(row=3, column=1, padx=10, pady=10)

        self.text_area = tk.Text(
            self, height=15, width=80, bg="#dcdde1", fg="#2f3640", font=("Helvetica", 12),
        )
        self.text_area.pack(pady=20)

        self.save_button = ttk.Button(self, text="Enregistrer dans Excel", command=self.save_to_excel)
        self.save_button.pack(pady=10)

    def display_message(self, message):
        self.text_area.insert(tk.END, f"{message}\n")
        self.text_area.see(tk.END)

    # -- enrollment ---------------------------------------------------------

    def enroll_face(self):
        name = simpledialog.askstring("Enrolement du visage", "Entrez le nom de l'utilisateur :")
        if not name:
            return
        self.face_recognizer.enroll(name, logger=self.display_message)
        self.display_message("Modele de reconnaissance faciale mis a jour")

    def enroll_voice(self):
        name = simpledialog.askstring("Enrolement de la voix", "Entrez le nom de l'utilisateur :")
        if not name:
            return
        self.voice_recognizer.enroll(
            name, logger=self.display_message,
            on_ready_to_record=lambda i: messagebox.showinfo(
                "Enregistrement vocal", f"Preparez-vous pour l'enregistrement {i}"
            ),
        )
        self.display_message("Modele de reconnaissance vocale mis a jour")

    # -- facial recognition ---------------------------------------------------

    def start_face_recognition(self):
        if self.face_recognition_running:
            return
        if not self.face_recognizer.is_ready:
            messagebox.showerror("Erreur", "Aucune donnee de visage disponible.")
            return
        self.face_recognition_running = True
        self.start_face_button.config(state=tk.DISABLED)
        self.stop_face_button.config(state=tk.NORMAL)
        threading.Thread(target=self._run_face_recognition, daemon=True).start()

    def stop_face_recognition(self):
        if self.face_recognition_running:
            self.face_recognition_running = False
            self.start_face_button.config(state=tk.NORMAL)
            self.stop_face_button.config(state=tk.DISABLED)

    def _run_face_recognition(self):
        self.face_recognizer.run_live_recognition(
            should_continue=lambda: self.face_recognition_running, logger=self.display_message,
        )
        self.face_recognition_running = False
        self.start_face_button.config(state=tk.NORMAL)
        self.stop_face_button.config(state=tk.DISABLED)

    # -- voice recognition -----------------------------------------------------

    def start_voice_recognition(self):
        if self.voice_recognition_running:
            return
        if not self.voice_recognizer.is_ready:
            messagebox.showerror("Erreur", "Le classificateur vocal n'est pas disponible.")
            return
        self.voice_recognition_running = True
        self.start_voice_button.config(state=tk.DISABLED)
        self.stop_voice_button.config(state=tk.NORMAL)
        threading.Thread(target=self._run_voice_recognition_loop, daemon=True).start()

    def stop_voice_recognition(self):
        if self.voice_recognition_running:
            self.voice_recognition_running = False
            self.start_voice_button.config(state=tk.NORMAL)
            self.stop_voice_button.config(state=tk.DISABLED)

    def _run_voice_recognition_loop(self):
        temp_audio_path = os.path.join(config.REPO_ROOT, "temp_audio.wav")
        while self.voice_recognition_running:
            ready = messagebox.askokcancel("Reconnaissance vocale", "Appuyez sur OK et commencez a parler.")
            if not ready:
                self.display_message("Reconnaissance vocale annulee.")
                break

            self.display_message("Enregistrement audio en cours...")
            self.voice_recognizer.record_sample(temp_audio_path)

            try:
                embedding = self.voice_recognizer.extract_embedding(temp_audio_path)
                name = self.voice_recognizer.predict(embedding) or "Inconnu(e)"
                self.display_message(f"Locuteur reconnu : {name}")
            except Exception as error:
                self.display_message(f"Erreur lors de la reconnaissance vocale : {error}")

            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

        self.voice_recognition_running = False
        self.start_voice_button.config(state=tk.NORMAL)
        self.stop_voice_button.config(state=tk.DISABLED)

    # -- OCR ------------------------------------------------------------------

    def run_ocr_import(self):
        image_path = filedialog.askopenfilename(
            title="Selectionnez une image", filetypes=[("Images", "*.jpg;*.jpeg;*.png")],
        )
        if not image_path:
            return
        self._extract_and_display(image_path)

    def run_ocr_capture(self):
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            messagebox.showerror("Erreur", "Impossible d'acceder a la webcam.")
            return

        temp_image_path = os.path.join(config.REPO_ROOT, "temp_capture.jpg")
        try:
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    continue
                cv2.imshow('Capture pour OCR - Appuyez sur "c" pour capturer', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("c"):
                    cv2.imwrite(temp_image_path, frame)
                    break
                if key == ord("q"):
                    return
        finally:
            video_capture.release()
            cv2.destroyAllWindows()

        self._extract_and_display(temp_image_path)
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

    def _extract_and_display(self, image_path):
        self.display_message("Extraction du texte en cours...")
        try:
            for text in self.text_extractor.extract_text(image_path):
                self.display_message(f"- {text}")
        except Exception as error:
            messagebox.showerror("Erreur OCR", f"Une erreur est survenue : {error}")

    # -- export -----------------------------------------------------------------

    def save_to_excel(self):
        lines = self.text_area.get("1.0", tk.END).strip().split("\n")
        if not lines or lines == [""]:
            messagebox.showwarning("Avertissement", "Aucune donnee a enregistrer.")
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
        if filepath:
            export_lines_to_excel(lines, filepath)
            messagebox.showinfo("Succes", "Donnees enregistrees avec succes dans Excel.")
