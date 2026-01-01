"""Package initializer for separated_training_files.

Exposes train() and evaluate() convenience wrappers if needed.
"""

from .train import main as train
from .evaluate import evaluate
