"""
This module provides functions to be used when answer processing functions
(OCR and ASR) require a separate Python environment from the main application, 
to avoid dependency conflicts. This is the case for the consumer hardware platform
(Intel MacBook), where LLMs require a newer version of numpy than the one supported 
by the OCR and ASR models.

The functions in this module run the answer processing modules as subprocesses 
and require a virtual environment in the answer_processing package with the necessary 
dependencies installed (see the pyproject.toml file in the answer_processing directory).
"""

import subprocess
from pathlib import Path



# Define the path to the answer_processing package in the sibling directory.
CURRENT_MODULE_DIRPATH = Path(__file__).parent.resolve()
ANSWER_PROCESSING_DIRPATH = CURRENT_MODULE_DIRPATH.parent / "answer_processing"


def image_to_text(
    image_path: str,
    ocr_python: str = ANSWER_PROCESSING_DIRPATH / ".venv/bin/python",
    ocr_runnable: str = ANSWER_PROCESSING_DIRPATH / "ocr.py",
    timeout: int = 60,
) -> str:
    """Run the ocr.py module from the answer_processing package with its own Python and return the text it prints to stdout.
    """
    # Convert the runnable path string to a Path object for validation.
    ocr_runnable_filepath = Path(ocr_runnable)
    # Verify the OCR script exists before attempting to run it.
    if not ocr_runnable_filepath.exists():
        raise FileNotFoundError(f"`{ocr_runnable}` not found")

    # Execute the OCR script as a subprocess using the specified Python interpreter:
    # capture_output=True captures both stdout and stderr;
    # text=True returns string output instead of bytes;
    # check=False prevents automatic exception on non-zero return code (handled manually below).
    proc = subprocess.run(
        [ocr_python, str(ocr_runnable_filepath), image_path],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )

    # Check if the subprocess failed and raise an error with the stderr message.
    if proc.returncode != 0:
        raise RuntimeError(f"The OCR process failed ({proc.returncode}):\n{proc.stderr.strip()}")

    # Return the OCR output text, stripped of leading/trailing whitespace.
    return proc.stdout.strip()


def audio_to_text(
    image_path: str,
    asr_python: str = ANSWER_PROCESSING_DIRPATH / ".venv/bin/python",
    asr_runnable: str = ANSWER_PROCESSING_DIRPATH / "asr.py",
    timeout: int = 60,
) -> str:
    """Run the asr.py module from the answer_processing package with its own Python and return the text it prints to stdout.
    """
    # Convert the runnable path to a Path object for validation.
    asr_runnable_filepath = Path(asr_runnable)
    # Verify the ASR script exists before attempting to run it.
    if not asr_runnable_filepath.exists():
        raise FileNotFoundError(f"`{asr_runnable}` not found")

    # Execute the ASR script as a subprocess using the specified Python interpreter:
    # capture_output=True captures both stdout and stderr;
    # text=True returns string output instead of bytes;
    # check=False prevents automatic exception on non-zero return code (handled manually below).
    proc = subprocess.run(
        [asr_python, str(asr_runnable_filepath), image_path],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )

    # Check if the subprocess failed and raise an error with the stderr message.
    if proc.returncode != 0:
        raise RuntimeError(f"The ASR process failed ({proc.returncode}):\n{proc.stderr.strip()}")

    # Return the ASR output text, stripped of leading/trailing whitespace.
    return proc.stdout.strip()
