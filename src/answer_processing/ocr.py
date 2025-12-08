import subprocess
import sys
from pathlib import Path

def invoke_convert_to_text(
    image_path: str,
    ocr_python: str = "../3070-final-project-ocr/.venv/bin/python",
    ocr_main: str = "../3070-final-project-ocr/main.py",
    timeout: int = 60,
) -> str:
    """Run the main.py module from OCR with its own Python and return the text it prints to stdout.
    """
    other_main_path = Path(ocr_main)
    if not other_main_path.exists():
        raise FileNotFoundError(f"`{ocr_main}` not found")

    proc = subprocess.run(
        [ocr_python, str(other_main_path), image_path],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )

    if proc.returncode != 0:
        raise RuntimeError(f"The OCR process failed ({proc.returncode}):\n{proc.stderr.strip()}")

    return proc.stdout.strip()


if __name__ == "__main__":
    image_filepath_arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if not image_filepath_arg:
        raise ValueError("Please provide an image filepath as an argument.")
    print(invoke_convert_to_text(image_filepath_arg))
