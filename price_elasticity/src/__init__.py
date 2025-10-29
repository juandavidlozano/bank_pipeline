# Price Elasticity Analytics Package
"""
Advanced price elasticity modeling and analysis framework
for marketing analytics and strategic pricing decisions.
"""

__version__ = "2.1.0"
__author__ = "Juan David Lozano"
__email__ = "juandavidlozano@hotmail.com"

from .elasticity_models import ElasticityAnalyzer
from .data_processor import DataProcessor
from .visualization import ElasticityVisualizer
from .statistical_tests import StatisticalValidator

__all__ = [
    'ElasticityAnalyzer',
    'DataProcessor', 
    'ElasticityVisualizer',
    'StatisticalValidator'
]