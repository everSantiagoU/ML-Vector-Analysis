"""Manual reverse-mode dense networks in NumPy float64."""
from ._validation import NumericalError
from .parameters import Parameter
from .layers import Layer, LinearLayer
from .activations import Tanh, ReLU
from .mlp import MLP
from .losses import HalfMSELoss, CrossEntropyLoss
from .optimizers import Optimizer, SGDOptimizer, AdamOptimizer
from .checkpoint import TrainingState, CheckpointInfo, save_checkpoint, load_checkpoint
from .visualization import training_visualization, manifest_visualization
from .data import MiniBatchIterator
from .training import Trainer, TrainingConfig
from .training_result import TrainingResult, EpochMetrics, EvaluationMetrics

__all__ = ['Parameter', 'Layer', 'LinearLayer', 'Tanh', 'ReLU', 'MLP',
           'HalfMSELoss', 'CrossEntropyLoss', 'Optimizer', 'SGDOptimizer', 'AdamOptimizer', 'MiniBatchIterator',
           'Trainer', 'TrainingConfig', 'TrainingResult', 'EpochMetrics',
           'EvaluationMetrics', 'TrainingState', 'CheckpointInfo', 'save_checkpoint', 'load_checkpoint',
           'training_visualization', 'manifest_visualization', 'NumericalError']
