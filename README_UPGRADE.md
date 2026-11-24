# InsureVerify — integrated repair
Extract this ZIP into a new folder. Do not mix with previous versions.

Windows PowerShell (Python 3.11 or 3.12 recommended):
```
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```
For scanned PDFs install Tesseract OCR separately (English language data). The pipeline searches PATH and standard Windows installation folders.
The optional LayoutLMv3 checkbox runs the original pretrained feature encoder; it requires torch, transformers and model weights. Install `requirements-model.txt` if needed. It does not predict insurance fields. Default extraction uses native PDF text or OCR, spatial rules and source verification, without downloading model weights.

Regression: `python regression_test.py PATH_TO_DATASET_ZIP`. Tests first three pages (or fewer) of five supplied documents, 28 selected field values and nonempty source boxes. This is not a complete accuracy benchmark. UI and model inference were not executed in the build environment. Scanned documents, other layouts and later pages need further validation. The app processes all pages by default. Review uncertain results against the PDF. A source box may cover a whole token when the value is embedded in it.
