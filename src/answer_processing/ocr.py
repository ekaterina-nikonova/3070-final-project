import sys

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

OCR_MODEL_NAME = "jzhang533/manga-ocr-base-2025"


def convert_to_text(image_filepath: str):
    processor = TrOCRProcessor.from_pretrained(OCR_MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(OCR_MODEL_NAME)

    image = Image.open(image_filepath).convert("RGB")

    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


if __name__ == "__main__":
    image_filepath_arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if not image_filepath_arg:
        raise ValueError("Please provide an image filepath as an argument.")
    print(convert_to_text(image_filepath_arg))
