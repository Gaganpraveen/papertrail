"""PaperTrail: inspectable research briefings grounded in arXiv PDFs."""

import os

# ONNX Runtime is imported transitively by the vector client. Disable optional
# telemetry before native initialization, including its background exit hooks.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

__version__ = "0.1.0"
