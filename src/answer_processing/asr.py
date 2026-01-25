import sys

import soundfile as sf
from transformers import AutoModelForCTC, Wav2Vec2Processor
import torch

ASR_MODEL_NAME = "vumichien/wav2vec2-large-xlsr-japanese"


def convert_to_text(audio_filepath: str):
    processor = Wav2Vec2Processor.from_pretrained(ASR_MODEL_NAME)
    model = AutoModelForCTC.from_pretrained(ASR_MODEL_NAME)
    model.eval()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)  # Move model to the CUDA device
        print(f"Using device: {torch.cuda.get_device_name(device)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")

    raw_speech, sr = sf.read(audio_filepath)
    inputs = processor(raw_speech, sampling_rate=sr, return_tensors="pt").to(device)  # Move inputs to the CUDA device
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    return ''.join(transcription.split())  # characters may be separated with whitespaces


if __name__ == "__main__":
    image_filepath_arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if not image_filepath_arg:
        raise ValueError("Please provide an image filepath as an argument.")
    print(convert_to_text(image_filepath_arg))
