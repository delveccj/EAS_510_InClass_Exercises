"""
SVM Package - Support Vector Machine implementations and examples
Converted from Jupyter notebook to standalone Python modules.
"""

from .config import setup_environment
from .linear_svm import LinearSVMExamples
from .nonlinear_svm import NonlinearSVMExamples
from .soft_margin import SoftMarginSVMExamples
from .svm_regression import SVMRegressionExamples
from .visualization import SVMVisualizer

__all__ = [
    'setup_environment',
    'LinearSVMExamples',
    'NonlinearSVMExamples', 
    'SoftMarginSVMExamples',
    'SVMRegressionExamples',
    'SVMVisualizer'
]