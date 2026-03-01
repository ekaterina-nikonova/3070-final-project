import sys

from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

# Define the pretrained model name for image-to-text conversion.
# The Japanese/manga OCR model from Hugging Face Hub is aimed at extracting text from manga-style images, 
# which means it supports multiline input in various orientations, including vertical text.
OCR_MODEL_NAME = "jzhang533/manga-ocr-base-2025"


def convert_to_text(image_filepath: str):
    """Converts an image containing Japanese text into a string.
    
    Args:
        image_filepath: Path to the image file to process.
        
    Returns:
        Extracted text from the image with whitespace removed.
    """
    # Load the pretrained processor which handles image preprocessing and text decoding and the model
    processor = TrOCRProcessor.from_pretrained(OCR_MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(OCR_MODEL_NAME)
    # Set the model to evaluation mode (disables dropout and other training-specific layers for inference)
    model.eval()

    # Check if a CUDA-compatible GPU is available for acceleration
    # Check if a CUDA-capable GPU is available for acceleration
    if torch.cuda.is_available():
        # Create a CUDA device object representing the default GPU.
        device = torch.device("cuda")
        # Transfer the model to GPU memory for faster computation.
        model.to(device)
        # Print diagnostic information about the GPU being used, 
        # CUDA version and number of devices.
        print(f"Using device: {torch.cuda.get_device_name(device)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    else:
        # If no GPU is available, use the CPU for inference.
        device = torch.device("cpu")
        print("No CUDA-compatible GPU found. Using CPU for inference.")

    # Open the image file and convert it to RGB format (required by the model).
    # This ensures support of both grayscale and RGBA images produced by the frontend.
    image = Image.open(image_filepath).convert("RGB")

    # Preprocess the image using the processor: normalises, resize, convert to PyTorch tensor.
    pixel_values = processor(image, return_tensors="pt").pixel_values
    # IMPORTANT! Move the input tensor to the same device (GPU) as the model.
    pixel_values = pixel_values.to(device)
    
    # Generate text predictions by running the image through the encoder-decoder model.
    # Token IDs represent the predicted text.
    generated_ids = model.generate(pixel_values)

    # Decode the generated token IDs back into human-readable text.
    # The batch_decode method handles batched outputs; skip_special_tokens removes [PAD], [EOS], etc.
    # Only the first (and only) result is extracted from the batch.
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # The model may output characters separated by spaces, so we concatenate them
    # according to Japanese orthographic conventions where characters are typically 
    # written without spaces.
    return ''.join(generated_text.split())


# Entry point when running this module directly as a script (for testing purposes).
if __name__ == "__main__":
    # Get the first command-line argument as the image filepath.
    # If not provided, use an empty string.
    image_filepath_arg = sys.argv[1] if len(sys.argv) > 1 else ""
    
    # Validate that an image filepath was provided and raise an error otherwise.
    if not image_filepath_arg:
        raise ValueError("Please provide an image filepath as an argument.")
    
    # Extract the text from the image and print it to stdout.
    print(convert_to_text(image_filepath_arg))
