import subprocess
from pathlib import Path

def image_to_text(
    image_path: str,
    ocr_python: str = "../answer_processing/.venv/bin/python",
    ocr_runnable: str = "../answer_processing/ocr.py",
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
    asr_python: str = "../answer_processing/.venv/bin/python",
    asr_runnable: str = "../answer_processing/asr.py",
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
