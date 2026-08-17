"""Voice recognition: Wav2Vec2.0 embeddings + a KNN classifier, as
described in the thesis (Chapter 3.3, 3.7) and measured in Chapter 4.3
(93.3% accuracy in a quiet room, 70% with background noise).
"""

import os

import numpy as np
import sounddevice as sd
import torch
import torchaudio
import wavio
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from . import config


class VoiceRecognizer:
    def __init__(self, data_dir=config.VOICE_DATA_DIR, model_name=config.WAV2VEC_MODEL_NAME):
        self.data_dir = str(data_dir)
        os.makedirs(self.data_dir, exist_ok=True)

        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)

        self.label_encoder = LabelEncoder()
        self.knn = None
        self.embeddings, self.labels = self.load_data()
        if len(self.labels) > 0:
            self._fit(self.embeddings, self.labels)

    @property
    def is_ready(self):
        return self.knn is not None

    def extract_embedding(self, audio_path):
        """Return the mean-pooled last hidden state of a mono 16kHz signal."""
        speech, sampling_rate = torchaudio.load(audio_path)

        if speech.shape[0] > 1:
            speech = torch.mean(speech, dim=0, keepdim=True)
        if sampling_rate != config.VOICE_SAMPLE_RATE_HZ:
            resampler = torchaudio.transforms.Resample(sampling_rate, config.VOICE_SAMPLE_RATE_HZ)
            speech = resampler(speech)

        inputs = self.processor(
            speech.squeeze(0), sampling_rate=config.VOICE_SAMPLE_RATE_HZ,
            return_tensors="pt", padding=True,
        ).input_values

        with torch.no_grad():
            hidden_states = self.model(inputs).last_hidden_state

        return hidden_states.mean(dim=1).numpy()[0]

    def load_data(self):
        """Extract an embedding for every enrolled sample under data_dir/<name>/*.wav."""
        embeddings, labels = [], []
        for person_name in os.listdir(self.data_dir):
            person_dir = os.path.join(self.data_dir, person_name)
            if not os.path.isdir(person_dir):
                continue
            audio_files = [
                os.path.join(person_dir, f) for f in os.listdir(person_dir) if f.endswith(".wav")
            ]
            for audio_file in audio_files:
                embeddings.append(self.extract_embedding(audio_file))
                labels.append(person_name)
        return np.array(embeddings), np.array(labels)

    def _fit(self, embeddings, labels):
        encoded = self.label_encoder.fit_transform(labels)
        self.knn = KNeighborsClassifier(n_neighbors=config.VOICE_KNN_NEIGHBORS)
        self.knn.fit(embeddings, encoded)

    def refresh(self):
        """Reload enrolled voices and refit the classifier. Returns is_ready."""
        self.embeddings, self.labels = self.load_data()
        if len(self.labels) > 0:
            self._fit(self.embeddings, self.labels)
        else:
            self.knn = None
        return self.is_ready

    def predict(self, embedding):
        """Predict the enrolled speaker for one embedding, or None."""
        predicted = self.knn.predict([embedding])[0]
        name = self.label_encoder.inverse_transform([predicted])[0]
        return name if name in self.labels else None

    def enroll(self, name, logger=lambda message: None, num_samples=config.VOICE_ENROLL_SAMPLES,
               duration=config.VOICE_SAMPLE_DURATION_SECONDS, sample_rate=config.VOICE_SAMPLE_RATE_HZ,
               on_ready_to_record=lambda index: None):
        """Record num_samples clips of `name`'s voice and refit the classifier."""
        user_dir = os.path.join(self.data_dir, name)
        os.makedirs(user_dir, exist_ok=True)

        for i in range(1, num_samples + 1):
            on_ready_to_record(i)
            logger(f"Enregistrement {i} en cours...")
            recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
            sd.wait()
            wavio.write(os.path.join(user_dir, f"audio{i}.wav"), recording, sample_rate, sampwidth=2)
            logger(f"Enregistrement {i} termine")

        return self.refresh()

    def record_sample(self, path, duration=config.VOICE_SAMPLE_DURATION_SECONDS,
                       sample_rate=config.VOICE_SAMPLE_RATE_HZ):
        """Record one clip to `path` for a one-off recognition attempt."""
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
        sd.wait()
        wavio.write(path, recording, sample_rate, sampwidth=2)
        return path
