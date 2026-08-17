"""Central configuration: paths and tunable constants.

The original script hardcoded an absolute Windows path to the Haar
cascade file, which only worked on the author's own machine. Every
path here is resolved relative to the repository root instead.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CASCADE_PATH = REPO_ROOT / "data" / "haarcascade_frontalface_default.xml"

FACE_DATA_DIR = REPO_ROOT / "face_data"
FACE_IMAGE_SIZE = (50, 50)
FACE_ENROLL_MAX_IMAGES = 50
FACE_DETECT_SCALE_FACTOR = 1.3
FACE_DETECT_MIN_NEIGHBORS = 5
FACE_KNN_NEIGHBORS = 5

VOICE_DATA_DIR = REPO_ROOT / "voice_data"
VOICE_ENROLL_SAMPLES = 3
VOICE_SAMPLE_DURATION_SECONDS = 5
VOICE_SAMPLE_RATE_HZ = 16000
VOICE_KNN_NEIGHBORS = 3
WAV2VEC_MODEL_NAME = "facebook/wav2vec2-base"

OCR_LANGUAGES = ["fr"]
OCR_MAX_IMAGE_DIMENSION = 1000
