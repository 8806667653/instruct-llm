"""Instruct-Lite: A lightweight framework for instruction tuning and RAG."""

__version__ = "0.1.0"

# Make commonly used modules easily accessible
from . import model
from . import loader
from . import finetune
from . import formatter
from . import utils

# Optional RAG module (requires faiss)
try:
    from . import rag
    __all__ = ["model", "loader", "finetune", "formatter", "rag", "utils", "experiment"]
except ImportError:
    # RAG module not available (faiss not installed)
    __all__ = ["model", "loader", "finetune", "formatter", "utils", "experiment"]

# Optional experiment module
try:
    from . import experiment
except ImportError:
    pass

