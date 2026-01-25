import sys

from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

OCR_MODEL_NAME = "jzhang533/manga-ocr-base-2025"


def convert_to_text(image_filepath: str):
    processor = TrOCRProcessor.from_pretrained(OCR_MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(OCR_MODEL_NAME)
    model.eval()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)  # Move model to the CUDA device
        print(f"Using device: {torch.cuda.get_device_name(device)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")

    image = Image.open(image_filepath).convert("RGB")

    pixel_values = processor(image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)  # Move input tensor to the same device as model
    
    generated_ids = model.generate(pixel_values)

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return ''.join(generated_text.split())  # characters may be separated with whitespaces


if __name__ == "__main__":
    image_filepath_arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if not image_filepath_arg:
        raise ValueError("Please provide an image filepath as an argument.")
    print(convert_to_text(image_filepath_arg))
