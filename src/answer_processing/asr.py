"""ASR (Automatic Speech Recognition) Module

This module provides functionality to convert Japanese audio files to text
using a pre-trained Wav2Vec2 model fine-tuned for Japanese language.
"""

import sys
import soundfile as sf
from transformers import AutoModelForCTC, Wav2Vec2Processor
import torch

# Define the Hugging Face model identifier for a Japanese Wav2Vec2 model.
# This model is fine-tuned on Japanese speech data 
# using XLSR (Cross-Lingual Speech Representations)
ASR_MODEL_NAME = "vumichien/wav2vec2-large-xlsr-japanese"


def convert_to_text(audio_filepath: str):
    """Convert an audio file to Japanese text using Wav2Vec2 ASR model.
    
    Args:
        audio_filepath: Path to the audio file to transcribe.
        
    Returns:
        Transcribed text with whitespace removed between characters
        (standard for Japanese orthography).
    """
    # Load the processor that handles tokenisation and feature extraction for the model.
    # The processor converts raw audio waveforms into the format expected by the model.
    processor = Wav2Vec2Processor.from_pretrained(ASR_MODEL_NAME)
    
    # Load the pre-trained Wav2Vec2 model with a CTC head for speech recognition.
    # CTC allows the model to predict character sequences from audio 
    # without requiring alignment.
    model = AutoModelForCTC.from_pretrained(ASR_MODEL_NAME)
    
    # Set the model to evaluation mode, disabling dropout and batch normalization updates.
    # This ensures consistent and deterministic inference results.
    model.eval()

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

    # Read the audio file and extract the raw waveform data and sample rate in Hz.
    raw_speech, sr = sf.read(audio_filepath)
    
    # Preprocess the audio data using the processor:
    # - Normalises the audio to the expected format
    # - Converts to PyTorch tensors ("pt")
    # - Moves the tensor to the same device as the model for processing
    inputs = processor(raw_speech, sampling_rate=sr, return_tensors="pt").to(device)
    
    # Disable gradient computation for inference to save memory and speed up processing
    # (only needed for training the model).
    with torch.no_grad():
        # Pass the preprocessed audio through the model
        # which outputs logits (unnormalised probabilities) for each time step.
        # The output shape: (batch_size, sequence_length, vocabulary_size)
        logits = model(**inputs).logits

    # Find the most probable token ID at each time step using argmax
    # over the vocabulary dimension (last dimension of the logits, -1).
    # This results in the sequence of predicted characters IDs.
    predicted_ids = torch.argmax(logits, dim=-1)
    
    # Decode the predicted token IDs back into human-readable text.
    # The batch_decode method returns a list, so we take the first (and only) element.
    transcription = processor.batch_decode(predicted_ids)[0]

    # The model may output characters separated by spaces, so we concatenate them
    # according to Japanese orthographic conventions where characters are typically 
    # written without spaces.
    return ''.join(transcription.split())


# Entry point when running this module directly as a script (for testing purposes).
if __name__ == "__main__":
    # Get the audio filepath from command-line arguments.
    # If no argument is provided, use an empty string.
    audio_filepath_arg = sys.argv[1] if len(sys.argv) > 1 else ""
    
    # Validate that an audio filepath was provided.
    if not audio_filepath_arg:
        raise ValueError("Please provide an audio filepath as an argument.")
    
    # Print the result of the audio transcription to stdout
    print(convert_to_text(audio_filepath_arg))
