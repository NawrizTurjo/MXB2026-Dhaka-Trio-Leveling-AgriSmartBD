
from huggingface_hub import snapshot_download
import os

print("Starting model download...")

# This downloads the entire repo (model weights, config, preprocessor, etc.)
try:
    snapshot_download(
        repo_id="liriope/PlantDiseaseDetection",
        repo_type="model",
        local_dir="./plant_disease_model",
        local_dir_use_symlinks=False,  # Copy files directly (safer for offline)
        resume_download=True
    )
    print("Download complete! All files are now in ./plant_disease_model")
except Exception as e:
    print(f"Error downloading model: {e}")
