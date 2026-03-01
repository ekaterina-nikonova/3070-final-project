# Answer processing as a separate uv project

If the libraries for answer processing (OCR and ASR) on your platform require
a different version of Python and other dependencies (such as numpy) than the
rest of the project, create a separate uv project for answer processing by using
the `uv sync` command in this directory. Then use the functions in the `assessment.conversion` module to call OCR and ASR utilities as subprocesses.

Otherwise, `convert_to_text` functions can be invoked directly from the `answer_processing.asr` and `answer_processing.ocr` modules without the need for subprocesses, as long as the dependencies are compatible with the main project.
