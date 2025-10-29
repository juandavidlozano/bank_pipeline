"""
Unit Tests for Price Elasticity Models

This module contains comprehensive unit tests for the elasticity modeling
components including data processing, model estimation, and validation.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processor import DataProcessor
from elasticity_models import ElasticityAnalyzer
from statistical_tests import StatisticalValidator

class TestDataProcessor(unittest.TestCase):
    """Test cases for DataProcessor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = DataProcessor('../data/')
        
    def test_data_loading(self):
        """Test data loading functionality"""
        # This would normally load real data
        # For testing, we'll create mock data
        
        # Create sample market data
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        categories = ['Premium', 'Mid-Tier', 'Value']
        
        sample_data = []
        for date in dates:
            for category in categories:
                sample_data.append({
                    'date': date,
                    'product_category': category,
                    'price_own': np.random.uniform(10, 35),
                    'quantity_sold': np.random.randint(1000, 4000),
                    'revenue': np.random.uniform(10000, 50000),
                    'promotion_flag': np.random.choice([0, 1], p=[0.8, 0.2]),
                    'seasonality_index': np.random.uniform(0.8, 1.2),
                    'market_share': np.random.uniform(0.1, 0.6)
                })
        
        self.processor.market_data = pd.DataFrame(sample_data)
        
        # Test that data is loaded correctly
        self.assertIsNotNone(self.processor.market_data)
        self.assertGreater(len(self.processor.market_data), 0)
        self.assertIn('product_category', self.processor.market_data.columns)
        
    def test_feature_engineering(self):
        """Test elasticity feature creation"""
        # Create minimal test data
        test_data = {
            'date': pd.date_range('2023-01-01', periods=10),
            'product_category': ['Premium'] * 10,
            'price_own': np.random.uniform(25, 35, 10),
            'quantity_sold': np.random.randint(1000, 2000, 10),
            'revenue': np.random.uniform(25000, 70000, 10),
            'promotion_flag': [0] * 10,
            'seasonality_index': np.random.uniform(0.9, 1.1, 10),
            'market_share': np.random.uniform(0.15, 0.25, 10),
            'market_category': ['Premium'] * 10
        }
        
        # Create competitor data
        competitor_data = {
            'date': pd.date_range('2023-01-01', periods=10),
            'market_category': ['Premium'] * 10,
            'competitor_a_price': np.random.uniform(24, 34, 10),
            'competitor_b_price': np.random.uniform(26, 36, 10),
            'competitor_c_price': np.random.uniform(23, 33, 10),
            'competitor_a_promo': [0] * 10,
            'competitor_b_promo': [0] * 10,
            'competitor_c_promo': [0] * 10
        }
        
        self.processor.market_data = pd.DataFrame(test_data)
        self.processor.competitor_data = pd.DataFrame(competitor_data)
        
        # Test feature engineering
        processed_data = self.processor.create_elasticity_features()
        
        self.assertIsNotNone(processed_data)
        self.assertIn('price_diff_comp_a', processed_data.columns)
        self.assertIn('relative_price_comp_a', processed_data.columns)
        self.assertIn('log_price', processed_data.columns)
        
    def test_data_quality_check(self):
        """Test data quality assessment"""
        # Create data with some quality issues
        test_data = pd.DataFrame({
            'price_own': [10, 20, np.nan, 30, -5],  # Missing value and negative price
            'quantity_sold': [100, 200, 300, 400, 500],
            'revenue': [1000, 4000, 6000, 12000, -2500]  # Negative revenue
        })
        
        self.processor.market_data = test_data
        quality_report = self.processor.data_quality_check()
        
        self.assertIn('missing_values', quality_report)
        self.assertIn('consistency_issues', quality_report)
        
class TestElasticityAnalyzer(unittest.TestCase):
    """Test cases for ElasticityAnalyzer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = ElasticityAnalyzer()
        
    def test_elasticity_calculation(self):
        """Test basic elasticity calculation logic"""
        # Create synthetic data with known elasticity
        n_obs = 100
        base_price = 20
        base_quantity = 1000
        true_elasticity = -2.0
        
        # Generate price data with some variation
        prices = np.random.normal(base_price, 2, n_obs)
        
        # Generate quantity data based on elasticity
        price_changes = (prices - base_price) / base_price
        quantity_changes = true_elasticity * price_changes
        quantities = base_quantity * (1 + quantity_changes) + np.random.normal(0, 50, n_obs)
        
        # Create test dataset
        test_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=n_obs),
            'product_category': ['Test'] * n_obs,
            'price_own': prices,
            'quantity_sold': quantities,
            'competitor_a_price': prices * 0.95,
            'competitor_b_price': prices * 1.05,
            'seasonality_index': [1.0] * n_obs,
            'promotion_flag': [0] * n_obs
        })
        
        # Mock the data processor
        from unittest.mock import Mock
        mock_processor = Mock()
        mock_processor.processed_data = test_data
        
        self.analyzer.data_processor = mock_processor
        
        # Test linear elasticity estimation
        result = self.analyzer.estimate_linear_elasticity('Test')
        
        self.assertIn('own_price_elasticity', result)
        self.assertIsInstance(result['own_price_elasticity'], (int, float))
        
        # The estimated elasticity should be reasonably close to true elasticity
        # (allowing for noise and model differences)
        estimated_elasticity = result['own_price_elasticity']
        self.assertLess(abs(estimated_elasticity - true_elasticity), 1.0)
        
    def test_simulation_scenarios(self):
        """Test price change simulation functionality"""
        # Create mock results
        mock_results = {
            'Test': {
                'own_price_elasticity': -2.0
            }
        }
        
        # Create mock data
        test_data = pd.DataFrame({
            'product_category': ['Test'] * 10,
            'price_own': [20] * 10,
            'quantity_sold': [1000] * 10
        })
        
        from unittest.mock import Mock
        mock_processor = Mock()
        mock_processor.processed_data = test_data
        
        self.analyzer.data_processor = mock_processor
        self.analyzer.results = mock_results
        
        # Test simulation
        price_scenarios = {'Increase_5pct': 5, 'Decrease_10pct': -10}
        simulation = self.analyzer.run_elasticity_simulation(price_scenarios, 'Test')
        
        self.assertIn('scenarios', simulation)
        self.assertIn('Increase_5pct', simulation['scenarios'])
        self.assertIn('revenue_change_pct', simulation['scenarios']['Increase_5pct'])

class TestStatisticalValidator(unittest.TestCase):
    """Test cases for StatisticalValidator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.validator = StatisticalValidator()
        
    def test_outlier_detection(self):
        """Test outlier detection methods"""
        # Create data with known outliers
        normal_data = np.random.normal(0, 1, 100)
        outliers = np.array([10, -10, 15])  # Clear outliers
        test_data = np.concatenate([normal_data, outliers])
        
        outlier_results = self.validator._detect_outliers(test_data)
        
        self.assertIn('outliers_present', outlier_results)
        self.assertIn('n_outliers', outlier_results)
        self.assertTrue(outlier_results['outliers_present'])
        self.assertGreater(outlier_results['n_outliers'], 0)
        
    def test_normality_testing(self):
        """Test normality tests"""
        # Test with normal data
        normal_data = np.random.normal(0, 1, 1000)
        normality_results = self.validator._test_normality(normal_data)
        
        self.assertIn('is_normal', normality_results)
        self.assertIn('tests', normality_results)
        
        # Test with non-normal data (exponential)
        exp_data = np.random.exponential(1, 1000)
        non_normal_results = self.validator._test_normality(exp_data)
        
        self.assertIn('is_normal', non_normal_results)
        
    def test_elasticity_validation(self):
        """Test elasticity estimate validation"""
        # Test with reasonable elasticity estimates
        good_results = {
            'Premium': {
                'log_linear_model': {
                    'own_price_elasticity': -1.5,
                    'r_squared': 0.85,
                    'p_values': {'price_own': 0.01}
                }
            }
        }
        
        validation = self.validator.validate_elasticity_estimates(good_results)
        
        self.assertIn('overall_assessment', validation)
        self.assertIn('categories_validated', validation)
        
        # Test with problematic estimates
        bad_results = {
            'Premium': {
                'log_linear_model': {
                    'own_price_elasticity': 1.5,  # Positive elasticity (wrong sign)
                    'r_squared': 0.3,  # Poor fit
                    'p_values': {'price_own': 0.5}  # Not significant
                }
            }
        }
        
        bad_validation = self.validator.validate_elasticity_estimates(bad_results)
        
        # Should detect issues
        self.assertGreater(len(bad_validation['validation_issues']), 0)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline"""
    
    def test_pipeline_integration(self):
        """Test that all components work together"""
        # This would test the complete pipeline
        # For now, just test that classes can be instantiated together
        
        processor = DataProcessor('../data/')
        analyzer = ElasticityAnalyzer(processor)
        validator = StatisticalValidator()
        
        self.assertIsNotNone(processor)
        self.assertIsNotNone(analyzer)
        self.assertIsNotNone(validator)
        
        # Test that analyzer has reference to processor
        self.assertEqual(analyzer.data_processor, processor)

def run_tests():
    """Run all tests and print results"""
    print("🧪 Running Price Elasticity Model Tests")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestDataProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestElasticityAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestStatisticalValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed. See details above.")
        
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)