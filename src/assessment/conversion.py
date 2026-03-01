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
    ocr_runnable_filepath = Path(ocr_runnable)
    if not ocr_runnable_filepath.exists():
        raise FileNotFoundError(f"`{ocr_runnable}` not found")

    proc = subprocess.run(
        [ocr_python, str(ocr_runnable_filepath), image_path],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )

    if proc.returncode != 0:
        raise RuntimeError(f"The OCR process failed ({proc.returncode}):\n{proc.stderr.strip()}")

    return proc.stdout.strip()


def audio_to_text(
    image_path: str,
    asr_python: str = ANSWER_PROCESSING_DIRPATH / ".venv/bin/python",
    asr_runnable: str = ANSWER_PROCESSING_DIRPATH / "asr.py",
    timeout: int = 60,
) -> str:
    """Run the asr.py module from the answer_processing package with its own Python and return the text it prints to stdout.
    """
    asr_runnable_filepath = Path(asr_runnable)
    if not asr_runnable_filepath.exists():
        raise FileNotFoundError(f"`{asr_runnable}` not found")

    proc = subprocess.run(
        [asr_python, str(asr_runnable_filepath), image_path],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )

    if proc.returncode != 0:
        raise RuntimeError(f"The ASR process failed ({proc.returncode}):\n{proc.stderr.strip()}")

    return proc.stdout.strip()
