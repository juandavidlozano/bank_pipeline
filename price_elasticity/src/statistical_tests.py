"""
Statistical Testing and Validation Module for Price Elasticity Analysis

This module provides advanced statistical tests, model validation techniques,
and econometric diagnostics for ensuring robust elasticity estimates.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy import stats
    from scipy.stats import jarque_bera, shapiro, normaltest
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ SciPy not available. Some statistical tests will be limited.")

class StatisticalValidator:
    """
    Comprehensive statistical validation and testing suite
    """
    
    def __init__(self):
        """
        Initialize statistical validator
        """
        self.test_results = {}
        self.validation_summary = {}
        
    def test_model_assumptions(self, model, model_name: str = "Model") -> Dict:
        """
        Test key econometric assumptions for elasticity models
        
        Args:
            model: Fitted statsmodels regression model
            model_name (str): Name identifier for the model
            
        Returns:
            dict: Comprehensive assumption test results
        """
        results = {
            'model_name': model_name,
            'tests_performed': [],
            'assumption_violations': [],
            'overall_validity': 'Unknown'
        }
        
        if not hasattr(model, 'resid'):
            results['error'] = "Model does not have residuals available"
            return results
            
        residuals = model.resid
        fitted_values = model.fittedvalues if hasattr(model, 'fittedvalues') else None
        
        # 1. Normality of residuals
        normality_results = self._test_normality(residuals)
        results['normality_test'] = normality_results
        results['tests_performed'].append('Normality')
        
        if not normality_results['is_normal']:
            results['assumption_violations'].append('Non-normal residuals')
        
        # 2. Homoscedasticity (constant variance)
        if fitted_values is not None:
            heteroskedasticity_results = self._test_heteroskedasticity(residuals, fitted_values)
            results['heteroskedasticity_test'] = heteroskedasticity_results
            results['tests_performed'].append('Homoscedasticity')
            
            if heteroskedasticity_results['heteroskedasticity_present']:
                results['assumption_violations'].append('Heteroskedasticity detected')
        
        # 3. Autocorrelation
        autocorr_results = self._test_autocorrelation(residuals)
        results['autocorrelation_test'] = autocorr_results
        results['tests_performed'].append('Autocorrelation')
        
        if autocorr_results['autocorrelation_present']:
            results['assumption_violations'].append('Autocorrelation detected')
        
        # 4. Outlier detection
        outlier_results = self._detect_outliers(residuals)
        results['outlier_test'] = outlier_results
        results['tests_performed'].append('Outlier Detection')
        
        if outlier_results['outliers_present']:
            results['assumption_violations'].append(f'{outlier_results["n_outliers"]} outliers detected')
        
        # Overall assessment
        if len(results['assumption_violations']) == 0:
            results['overall_validity'] = 'Strong'
        elif len(results['assumption_violations']) <= 2:
            results['overall_validity'] = 'Moderate'
        else:
            results['overall_validity'] = 'Weak'
        
        self.test_results[model_name] = results
        return results
    
    def _test_normality(self, residuals: np.ndarray) -> Dict:
        """
        Test normality of residuals using multiple tests
        
        Args:
            residuals (np.ndarray): Model residuals
            
        Returns:
            dict: Normality test results
        """
        results = {
            'test_type': 'Normality',
            'is_normal': True,
            'tests': {}
        }
        
        if SCIPY_AVAILABLE:
            # Shapiro-Wilk test (good for small samples)
            if len(residuals) <= 5000:
                try:
                    shapiro_stat, shapiro_pval = shapiro(residuals)
                    results['tests']['shapiro_wilk'] = {
                        'statistic': round(shapiro_stat, 4),
                        'p_value': round(shapiro_pval, 6),
                        'is_normal': shapiro_pval > 0.05
                    }
                    if shapiro_pval <= 0.05:
                        results['is_normal'] = False
                except:
                    results['tests']['shapiro_wilk'] = {'error': 'Test failed'}
            
            # Jarque-Bera test
            try:
                jb_stat, jb_pval = jarque_bera(residuals)
                results['tests']['jarque_bera'] = {
                    'statistic': round(jb_stat, 4),
                    'p_value': round(jb_pval, 6),
                    'is_normal': jb_pval > 0.05
                }
                if jb_pval <= 0.05:
                    results['is_normal'] = False
            except:
                results['tests']['jarque_bera'] = {'error': 'Test failed'}
            
            # D'Agostino and Pearson test
            try:
                da_stat, da_pval = normaltest(residuals)
                results['tests']['dagostino_pearson'] = {
                    'statistic': round(da_stat, 4),
                    'p_value': round(da_pval, 6),
                    'is_normal': da_pval > 0.05
                }
                if da_pval <= 0.05:
                    results['is_normal'] = False
            except:
                results['tests']['dagostino_pearson'] = {'error': 'Test failed'}
        else:
            # Basic normality check using skewness and kurtosis
            skewness = stats.skew(residuals) if SCIPY_AVAILABLE else self._calculate_skewness(residuals)
            kurtosis = stats.kurtosis(residuals) if SCIPY_AVAILABLE else self._calculate_kurtosis(residuals)
            
            results['tests']['basic_normality'] = {
                'skewness': round(skewness, 4),
                'kurtosis': round(kurtosis, 4),
                'is_normal': abs(skewness) < 2 and abs(kurtosis) < 7
            }
            
            if abs(skewness) >= 2 or abs(kurtosis) >= 7:
                results['is_normal'] = False
        
        return results
    
    def _test_heteroskedasticity(self, residuals: np.ndarray, fitted_values: np.ndarray) -> Dict:
        """
        Test for heteroskedasticity using Breusch-Pagan test
        
        Args:
            residuals (np.ndarray): Model residuals
            fitted_values (np.ndarray): Fitted values from model
            
        Returns:
            dict: Heteroskedasticity test results
        """
        results = {
            'test_type': 'Heteroskedasticity',
            'heteroskedasticity_present': False,
            'tests': {}
        }
        
        try:
            # Simple Breusch-Pagan-like test
            # Regress squared residuals on fitted values
            squared_residuals = residuals ** 2
            
            # Calculate correlation between squared residuals and fitted values
            correlation = np.corrcoef(squared_residuals, fitted_values)[0, 1]
            
            # Simple test statistic
            n = len(residuals)
            test_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))
            
            # Critical value for 95% confidence (approximately 1.96)
            critical_value = 1.96
            p_value = 2 * (1 - stats.norm.cdf(abs(test_stat))) if SCIPY_AVAILABLE else 0.05
            
            results['tests']['breusch_pagan_simple'] = {
                'correlation': round(correlation, 4),
                'test_statistic': round(test_stat, 4),
                'p_value': round(p_value, 6),
                'heteroskedasticity_present': abs(test_stat) > critical_value
            }
            
            if abs(test_stat) > critical_value:
                results['heteroskedasticity_present'] = True
                
        except Exception as e:
            results['tests']['breusch_pagan_simple'] = {'error': f'Test failed: {str(e)}'}
        
        return results
    
    def _test_autocorrelation(self, residuals: np.ndarray) -> Dict:
        """
        Test for autocorrelation using Durbin-Watson statistic
        
        Args:
            residuals (np.ndarray): Model residuals
            
        Returns:
            dict: Autocorrelation test results
        """
        results = {
            'test_type': 'Autocorrelation',
            'autocorrelation_present': False,
            'tests': {}
        }
        
        try:
            # Calculate Durbin-Watson statistic
            diff_residuals = np.diff(residuals)
            dw_stat = np.sum(diff_residuals**2) / np.sum(residuals**2)
            
            # Interpret DW statistic
            # DW ≈ 2: No autocorrelation
            # DW < 1.5: Positive autocorrelation
            # DW > 2.5: Negative autocorrelation
            
            interpretation = 'No autocorrelation'
            autocorr_present = False
            
            if dw_stat < 1.5:
                interpretation = 'Positive autocorrelation'
                autocorr_present = True
            elif dw_stat > 2.5:
                interpretation = 'Negative autocorrelation'
                autocorr_present = True
            
            results['tests']['durbin_watson'] = {
                'statistic': round(dw_stat, 4),
                'interpretation': interpretation,
                'autocorrelation_present': autocorr_present
            }
            
            results['autocorrelation_present'] = autocorr_present
            
        except Exception as e:
            results['tests']['durbin_watson'] = {'error': f'Test failed: {str(e)}'}
        
        return results
    
    def _detect_outliers(self, residuals: np.ndarray) -> Dict:
        """
        Detect outliers using multiple methods
        
        Args:
            residuals (np.ndarray): Model residuals
            
        Returns:
            dict: Outlier detection results
        """
        results = {
            'test_type': 'Outlier Detection',
            'outliers_present': False,
            'n_outliers': 0,
            'outlier_indices': [],
            'methods': {}
        }
        
        try:
            # Method 1: IQR method
            Q1 = np.percentile(residuals, 25)
            Q3 = np.percentile(residuals, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = np.where((residuals < lower_bound) | (residuals > upper_bound))[0]
            
            results['methods']['iqr'] = {
                'n_outliers': len(iqr_outliers),
                'outlier_indices': iqr_outliers.tolist(),
                'lower_bound': round(lower_bound, 4),
                'upper_bound': round(upper_bound, 4)
            }
            
            # Method 2: Z-score method
            z_scores = np.abs((residuals - np.mean(residuals)) / np.std(residuals))
            zscore_outliers = np.where(z_scores > 3)[0]
            
            results['methods']['zscore'] = {
                'n_outliers': len(zscore_outliers),
                'outlier_indices': zscore_outliers.tolist(),
                'threshold': 3.0
            }
            
            # Combine outliers from both methods
            all_outliers = np.unique(np.concatenate([iqr_outliers, zscore_outliers]))
            
            results['n_outliers'] = len(all_outliers)
            results['outlier_indices'] = all_outliers.tolist()
            results['outliers_present'] = len(all_outliers) > 0
            
            # Calculate outlier percentage
            outlier_percentage = (len(all_outliers) / len(residuals)) * 100
            results['outlier_percentage'] = round(outlier_percentage, 2)
            
        except Exception as e:
            results['methods']['error'] = f'Outlier detection failed: {str(e)}'
        
        return results
    
    def validate_elasticity_estimates(self, elasticity_results: Dict) -> Dict:
        """
        Validate elasticity estimates for economic reasonableness
        
        Args:
            elasticity_results (dict): Elasticity estimation results
            
        Returns:
            dict: Validation results with recommendations
        """
        validation = {
            'validation_type': 'Elasticity Estimates',
            'categories_validated': [],
            'validation_issues': [],
            'recommendations': [],
            'overall_assessment': 'Valid'
        }
        
        for category, results in elasticity_results.items():
            if 'log_linear_model' in results:
                model_results = results['log_linear_model']
                category_validation = self._validate_single_elasticity(category, model_results)
                validation['categories_validated'].append(category_validation)
                
                # Collect issues
                if category_validation['issues']:
                    validation['validation_issues'].extend(category_validation['issues'])
                    
                # Collect recommendations
                if category_validation['recommendations']:
                    validation['recommendations'].extend(category_validation['recommendations'])
        
        # Overall assessment
        if len(validation['validation_issues']) == 0:
            validation['overall_assessment'] = 'Strong'
        elif len(validation['validation_issues']) <= 3:
            validation['overall_assessment'] = 'Moderate'
        else:
            validation['overall_assessment'] = 'Weak'
        
        return validation
    
    def _validate_single_elasticity(self, category: str, model_results: Dict) -> Dict:
        """
        Validate elasticity estimates for a single category
        
        Args:
            category (str): Product category
            model_results (dict): Model estimation results
            
        Returns:
            dict: Category-specific validation results
        """
        validation = {
            'category': category,
            'elasticity': model_results.get('own_price_elasticity', 0),
            'issues': [],
            'recommendations': [],
            'validity_score': 100
        }
        
        elasticity = validation['elasticity']
        r_squared = model_results.get('r_squared', 0)
        
        # 1. Check if elasticity is negative (as expected for demand)
        if elasticity >= 0:
            validation['issues'].append(f'{category}: Positive elasticity ({elasticity:.2f}) - unexpected for demand')
            validation['recommendations'].append(f'{category}: Review model specification and data quality')
            validation['validity_score'] -= 30
        
        # 2. Check for extreme elasticity values
        if elasticity < -10:
            validation['issues'].append(f'{category}: Extremely high elasticity ({elasticity:.2f}) - may indicate specification error')
            validation['recommendations'].append(f'{category}: Check for outliers and model stability')
            validation['validity_score'] -= 20
        elif elasticity > -0.1:
            validation['issues'].append(f'{category}: Elasticity very close to zero ({elasticity:.2f}) - may indicate inelastic demand or poor model fit')
            validation['recommendations'].append(f'{category}: Investigate alternative model specifications')
            validation['validity_score'] -= 15
        
        # 3. Check model fit quality
        if r_squared < 0.5:
            validation['issues'].append(f'{category}: Low R-squared ({r_squared:.3f}) - poor model fit')
            validation['recommendations'].append(f'{category}: Add more explanatory variables or try alternative model forms')
            validation['validity_score'] -= 25
        elif r_squared < 0.7:
            validation['issues'].append(f'{category}: Moderate R-squared ({r_squared:.3f}) - model fit could be improved')
            validation['recommendations'].append(f'{category}: Consider additional variables or interaction terms')
            validation['validity_score'] -= 10
        
        # 4. Economic reasonableness by category
        if 'Premium' in category and elasticity < -1:
            # Premium products should be less elastic
            validation['issues'].append(f'{category}: Higher than expected elasticity for premium segment ({elasticity:.2f})')
            validation['recommendations'].append(f'{category}: Verify premium positioning and customer loyalty factors')
            validation['validity_score'] -= 10
        elif 'Value' in category and elasticity > -2:
            # Value products should be more elastic
            validation['issues'].append(f'{category}: Lower than expected elasticity for value segment ({elasticity:.2f})')
            validation['recommendations'].append(f'{category}: Consider price sensitivity in value segment positioning')
            validation['validity_score'] -= 10
        
        # 5. Statistical significance
        if 'p_values' in model_results:
            price_pval = model_results['p_values'].get('price_own', 1)
            if price_pval > 0.05:
                validation['issues'].append(f'{category}: Price coefficient not statistically significant (p={price_pval:.4f})')
                validation['recommendations'].append(f'{category}: Increase sample size or improve model specification')
                validation['validity_score'] -= 20
        
        return validation
    
    def generate_validation_report(self) -> str:
        """
        Generate comprehensive validation report
        
        Returns:
            str: Formatted validation report
        """
        report = []
        report.append("=" * 60)
        report.append("STATISTICAL VALIDATION REPORT")
        report.append("=" * 60)
        report.append()
        
        if not self.test_results:
            report.append("No model validation tests have been performed.")
            return "\n".join(report)
        
        # Model assumption tests summary
        report.append("MODEL ASSUMPTION TESTS")
        report.append("-" * 25)
        
        for model_name, results in self.test_results.items():
            report.append(f"\n{model_name.upper()}")
            report.append(f"Overall Validity: {results['overall_validity']}")
            report.append(f"Tests Performed: {', '.join(results['tests_performed'])}")
            
            if results['assumption_violations']:
                report.append("Assumption Violations:")
                for violation in results['assumption_violations']:
                    report.append(f"  • {violation}")
            else:
                report.append("✅ All assumptions satisfied")
        
        report.append()
        
        # Detailed test results
        report.append("DETAILED TEST RESULTS")
        report.append("-" * 22)
        
        for model_name, results in self.test_results.items():
            report.append(f"\n{model_name}:")
            
            # Normality tests
            if 'normality_test' in results:
                norm_results = results['normality_test']
                report.append(f"  Normality: {'✅ Pass' if norm_results['is_normal'] else '❌ Fail'}")
                for test_name, test_result in norm_results['tests'].items():
                    if 'p_value' in test_result:
                        report.append(f"    {test_name}: p-value = {test_result['p_value']}")
            
            # Heteroskedasticity tests
            if 'heteroskedasticity_test' in results:
                hetero_results = results['heteroskedasticity_test']
                report.append(f"  Homoscedasticity: {'❌ Fail' if hetero_results['heteroskedasticity_present'] else '✅ Pass'}")
            
            # Autocorrelation tests
            if 'autocorrelation_test' in results:
                autocorr_results = results['autocorrelation_test']
                report.append(f"  No Autocorrelation: {'❌ Fail' if autocorr_results['autocorrelation_present'] else '✅ Pass'}")
            
            # Outlier detection
            if 'outlier_test' in results:
                outlier_results = results['outlier_test']
                report.append(f"  Outliers: {outlier_results['n_outliers']} detected ({outlier_results.get('outlier_percentage', 0):.1f}%)")
        
        report.append()
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        print(report_text)
        
        return report_text
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness manually if scipy not available"""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        skew = np.sum(((data - mean) / std) ** 3) / n
        return skew
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis manually if scipy not available"""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        kurt = np.sum(((data - mean) / std) ** 4) / n - 3  # Excess kurtosis
        return kurt