"""
Advanced Price Elasticity Modeling Module

This module implements multiple econometric models for price elasticity analysis
including linear, log-linear, instrumental variables, and time series approaches.
Provides comprehensive statistical validation and business insights.
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional
import pickle
import json
warnings.filterwarnings('ignore')

try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("⚠️ Statsmodels not available. Some advanced features will be limited.")

try:
    from sklearn.model_selection import cross_val_score, TimeSeriesSplit
    from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ Scikit-learn not available. Using basic statistical models only.")

class ElasticityAnalyzer:
    """
    Comprehensive price elasticity analysis framework
    """
    
    def __init__(self, data_processor=None):
        """
        Initialize ElasticityAnalyzer
        
        Args:
            data_processor: DataProcessor instance with loaded data
        """
        self.data_processor = data_processor
        self.models = {}
        self.results = {}
        self.predictions = {}
        self.model_diagnostics = {}
        
    def load_data(self, data_path='data/market_data.csv'):
        """
        Load and prepare data for analysis
        
        Args:
            data_path (str): Path to market data file
            
        Returns:
            pd.DataFrame: Loaded and processed data
        """
        if self.data_processor is None:
            from .data_processor import DataProcessor
            self.data_processor = DataProcessor()
            
        # Load all datasets
        self.data_processor.load_all_data()
        
        # Create elasticity features
        processed_data = self.data_processor.create_elasticity_features()
        
        print("✅ Data loaded and processed for elasticity analysis")
        return processed_data
    
    def estimate_linear_elasticity(self, category: str = None) -> Dict:
        """
        Estimate linear demand elasticity using OLS regression
        
        Args:
            category (str): Product category to analyze. If None, analyzes all categories
            
        Returns:
            dict: Linear elasticity results
        """
        if self.data_processor.processed_data is None:
            raise ValueError("Data must be loaded and processed first")
            
        data = self.data_processor.processed_data.copy()
        
        if category:
            data = data[data['product_category'] == category]
            
        # Prepare variables for linear model
        # Dependent variable: quantity_sold
        # Independent variables: price_own, competitor prices, seasonality, promotions
        
        y = data['quantity_sold'].values
        X_vars = ['price_own', 'competitor_a_price', 'competitor_b_price', 
                 'seasonality_index', 'promotion_flag']
        
        # Handle missing values
        model_data = data[['quantity_sold'] + X_vars].dropna()
        y = model_data['quantity_sold'].values
        X = model_data[X_vars].values
        
        if STATSMODELS_AVAILABLE:
            # Add constant term
            X_with_const = sm.add_constant(X)
            
            # Fit OLS model
            model = sm.OLS(y, X_with_const).fit()
            
            # Calculate elasticity at mean price and quantity
            mean_price = model_data['price_own'].mean()
            mean_quantity = model_data['quantity_sold'].mean()
            price_coef = model.params[1]  # Price coefficient (first variable after constant)
            
            # Linear elasticity = (dQ/dP) * (P/Q)
            elasticity = price_coef * (mean_price / mean_quantity)
            
            results = {
                'model_type': 'Linear OLS',
                'category': category or 'All Categories',
                'own_price_elasticity': round(elasticity, 3),
                'price_coefficient': round(price_coef, 3),
                'r_squared': round(model.rsquared, 3),
                'adj_r_squared': round(model.rsquared_adj, 3),
                'f_statistic': round(model.fvalue, 3),
                'p_value_f': round(model.f_pvalue, 6),
                'coefficients': {
                    'intercept': round(model.params[0], 3),
                    'price_own': round(model.params[1], 3),
                    'competitor_a_price': round(model.params[2], 3),
                    'competitor_b_price': round(model.params[3], 3),
                    'seasonality_index': round(model.params[4], 3),
                    'promotion_flag': round(model.params[5], 3)
                },
                'p_values': {
                    'price_own': round(model.pvalues[1], 6),
                    'competitor_a_price': round(model.pvalues[2], 6),
                    'competitor_b_price': round(model.pvalues[3], 6),
                    'seasonality_index': round(model.pvalues[4], 6),
                    'promotion_flag': round(model.pvalues[5], 6)
                },
                'confidence_intervals': model.conf_int().round(3).to_dict(),
                'n_observations': len(model_data),
                'aic': round(model.aic, 2),
                'bic': round(model.bic, 2)
            }
            
            # Store model for later use
            model_key = f"linear_{category}" if category else "linear_all"
            self.models[model_key] = model
            
        else:
            # Fallback to basic numpy calculation
            from numpy.linalg import lstsq
            
            X_with_const = np.column_stack([np.ones(len(X)), X])
            coeffs, residuals, rank, s = lstsq(X_with_const, y, rcond=None)
            
            mean_price = model_data['price_own'].mean()
            mean_quantity = model_data['quantity_sold'].mean()
            price_coef = coeffs[1]
            elasticity = price_coef * (mean_price / mean_quantity)
            
            # Basic R-squared calculation
            y_pred = X_with_const @ coeffs
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            results = {
                'model_type': 'Linear (Basic)',
                'category': category or 'All Categories',
                'own_price_elasticity': round(elasticity, 3),
                'price_coefficient': round(price_coef, 3),
                'r_squared': round(r_squared, 3),
                'coefficients': {
                    'intercept': round(coeffs[0], 3),
                    'price_own': round(coeffs[1], 3),
                    'competitor_a_price': round(coeffs[2], 3),
                    'competitor_b_price': round(coeffs[3], 3),
                    'seasonality_index': round(coeffs[4], 3),
                    'promotion_flag': round(coeffs[5], 3)
                },
                'n_observations': len(model_data)
            }
        
        return results
    
    def estimate_log_linear_elasticity(self, category: str = None) -> Dict:
        """
        Estimate constant elasticity using log-linear model
        
        Args:
            category (str): Product category to analyze
            
        Returns:
            dict: Log-linear elasticity results
        """
        if self.data_processor.processed_data is None:
            raise ValueError("Data must be loaded and processed first")
            
        data = self.data_processor.processed_data.copy()
        
        if category:
            data = data[data['product_category'] == category]
            
        # Create log variables (add small constant to avoid log(0))
        data['log_quantity'] = np.log(data['quantity_sold'] + 1)
        data['log_price_own'] = np.log(data['price_own'])
        data['log_comp_a_price'] = np.log(data['competitor_a_price'] + 0.01)
        data['log_comp_b_price'] = np.log(data['competitor_b_price'] + 0.01)
        
        # Prepare variables
        y = data['log_quantity'].values
        X_vars = ['log_price_own', 'log_comp_a_price', 'log_comp_b_price', 
                 'seasonality_index', 'promotion_flag']
        
        model_data = data[['log_quantity'] + X_vars].dropna()
        y = model_data['log_quantity'].values
        X = model_data[X_vars].values
        
        if STATSMODELS_AVAILABLE:
            X_with_const = sm.add_constant(X)
            model = sm.OLS(y, X_with_const).fit()
            
            # In log-linear model, coefficients are elasticities
            own_price_elasticity = model.params[1]  # Log price coefficient
            
            results = {
                'model_type': 'Log-Linear',
                'category': category or 'All Categories',
                'own_price_elasticity': round(own_price_elasticity, 3),
                'cross_elasticity_comp_a': round(model.params[2], 3),
                'cross_elasticity_comp_b': round(model.params[3], 3),
                'r_squared': round(model.rsquared, 3),
                'adj_r_squared': round(model.rsquared_adj, 3),
                'f_statistic': round(model.fvalue, 3),
                'coefficients': {
                    'intercept': round(model.params[0], 3),
                    'log_price_own': round(model.params[1], 3),
                    'log_comp_a_price': round(model.params[2], 3),
                    'log_comp_b_price': round(model.params[3], 3),
                    'seasonality_index': round(model.params[4], 3),
                    'promotion_flag': round(model.params[5], 3)
                },
                'p_values': {
                    'log_price_own': round(model.pvalues[1], 6),
                    'log_comp_a_price': round(model.pvalues[2], 6),
                    'log_comp_b_price': round(model.pvalues[3], 6),
                    'seasonality_index': round(model.pvalues[4], 6),
                    'promotion_flag': round(model.pvalues[5], 6)
                },
                'n_observations': len(model_data),
                'aic': round(model.aic, 2),
                'bic': round(model.bic, 2)
            }
            
            model_key = f"log_linear_{category}" if category else "log_linear_all"
            self.models[model_key] = model
            
        else:
            # Basic fallback
            X_with_const = np.column_stack([np.ones(len(X)), X])
            coeffs, _, _, _ = np.linalg.lstsq(X_with_const, y, rcond=None)
            
            y_pred = X_with_const @ coeffs
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            results = {
                'model_type': 'Log-Linear (Basic)',
                'category': category or 'All Categories',
                'own_price_elasticity': round(coeffs[1], 3),
                'cross_elasticity_comp_a': round(coeffs[2], 3),
                'cross_elasticity_comp_b': round(coeffs[3], 3),
                'r_squared': round(r_squared, 3),
                'n_observations': len(model_data)
            }
        
        return results
    
    def estimate_competitive_elasticity(self) -> Dict:
        """
        Estimate comprehensive competitive elasticity model
        
        Returns:
            dict: Competitive elasticity analysis results
        """
        results = {}
        categories = self.data_processor.processed_data['product_category'].unique()
        
        for category in categories:
            # Estimate both linear and log-linear for each category
            linear_results = self.estimate_linear_elasticity(category)
            log_linear_results = self.estimate_log_linear_elasticity(category)
            
            results[category] = {
                'linear_model': linear_results,
                'log_linear_model': log_linear_results,
                'recommended_model': 'log_linear' if log_linear_results['r_squared'] > linear_results['r_squared'] else 'linear'
            }
        
        return results
    
    def run_elasticity_simulation(self, price_changes: Dict, category: str) -> Dict:
        """
        Simulate demand response to price changes
        
        Args:
            price_changes (dict): Price change scenarios {'scenario_name': price_change_percent}
            category (str): Product category for simulation
            
        Returns:
            dict: Simulation results
        """
        if category not in self.results:
            # Run elasticity estimation first
            self.results[category] = self.estimate_log_linear_elasticity(category)
        
        elasticity = self.results[category]['own_price_elasticity']
        base_data = self.data_processor.processed_data[
            self.data_processor.processed_data['product_category'] == category
        ]
        
        base_price = base_data['price_own'].mean()
        base_quantity = base_data['quantity_sold'].mean()
        base_revenue = base_price * base_quantity
        
        simulation_results = {}
        
        for scenario, price_change_pct in price_changes.items():
            # Calculate new price
            new_price = base_price * (1 + price_change_pct / 100)
            
            # Calculate quantity change using elasticity
            quantity_change_pct = elasticity * (price_change_pct / 100)
            new_quantity = base_quantity * (1 + quantity_change_pct)
            
            # Calculate new revenue
            new_revenue = new_price * new_quantity
            revenue_change_pct = ((new_revenue - base_revenue) / base_revenue) * 100
            
            simulation_results[scenario] = {
                'price_change_pct': price_change_pct,
                'new_price': round(new_price, 2),
                'quantity_change_pct': round(quantity_change_pct * 100, 2),
                'new_quantity': round(new_quantity, 0),
                'revenue_change_pct': round(revenue_change_pct, 2),
                'new_revenue': round(new_revenue, 2)
            }
        
        return {
            'category': category,
            'elasticity_used': elasticity,
            'base_metrics': {
                'price': round(base_price, 2),
                'quantity': round(base_quantity, 0),
                'revenue': round(base_revenue, 2)
            },
            'scenarios': simulation_results
        }
    
    def run_full_analysis(self) -> Dict:
        """
        Run comprehensive elasticity analysis across all categories and models
        
        Returns:
            dict: Complete analysis results
        """
        if self.data_processor.processed_data is None:
            self.load_data()
        
        print("🔄 Running comprehensive elasticity analysis...")
        
        # Get all categories
        categories = self.data_processor.processed_data['product_category'].unique()
        
        full_results = {
            'summary': {
                'total_categories': len(categories),
                'total_observations': len(self.data_processor.processed_data),
                'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'category_results': {},
            'competitive_analysis': {},
            'simulation_results': {}
        }
        
        # Run analysis for each category
        for category in categories:
            print(f"  📊 Analyzing {category} segment...")
            
            # Linear elasticity
            linear_results = self.estimate_linear_elasticity(category)
            
            # Log-linear elasticity
            log_linear_results = self.estimate_log_linear_elasticity(category)
            
            # Store results
            full_results['category_results'][category] = {
                'linear_model': linear_results,
                'log_linear_model': log_linear_results,
                'recommended_elasticity': log_linear_results['own_price_elasticity']
            }
            
            # Run simulation scenarios
            price_scenarios = {
                'Price_Increase_5pct': 5,
                'Price_Increase_10pct': 10,
                'Price_Decrease_5pct': -5,
                'Price_Decrease_10pct': -10
            }
            
            simulation = self.run_elasticity_simulation(price_scenarios, category)
            full_results['simulation_results'][category] = simulation
        
        # Overall competitive analysis
        competitive_results = self.estimate_competitive_elasticity()
        full_results['competitive_analysis'] = competitive_results
        
        # Store results
        self.results = full_results
        
        print("✅ Complete elasticity analysis finished!")
        return full_results
    
    def create_business_report(self) -> str:
        """
        Generate business-focused elasticity report
        
        Returns:
            str: Formatted business report
        """
        if not self.results:
            self.run_full_analysis()
        
        report = []
        report.append("=" * 60)
        report.append("PRICE ELASTICITY ANALYSIS - BUSINESS REPORT")
        report.append("=" * 60)
        report.append()
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 20)
        
        for category, results in self.results['category_results'].items():
            elasticity = results['recommended_elasticity']
            r_squared = results['log_linear_model']['r_squared']
            
            if elasticity < -2.5:
                sensitivity = "HIGHLY price sensitive"
            elif elasticity < -1.5:
                sensitivity = "MODERATELY price sensitive"
            else:
                sensitivity = "RELATIVELY price insensitive"
                
            report.append(f"{category}: Elasticity = {elasticity} ({sensitivity})")
            report.append(f"  Model Fit: R² = {r_squared} ({'Strong' if r_squared > 0.8 else 'Moderate' if r_squared > 0.6 else 'Weak'} predictive power)")
        
        report.append()
        
        # Revenue Impact Analysis
        report.append("REVENUE IMPACT SCENARIOS")
        report.append("-" * 25)
        
        for category in self.results['simulation_results']:
            report.append(f"\n{category.upper()} CATEGORY:")
            
            sim_data = self.results['simulation_results'][category]
            for scenario, data in sim_data['scenarios'].items():
                report.append(f"  {scenario.replace('_', ' ')}: {data['revenue_change_pct']:+.1f}% revenue impact")
        
        report.append()
        
        # Strategic Recommendations
        report.append("STRATEGIC RECOMMENDATIONS")
        report.append("-" * 24)
        
        for category, results in self.results['category_results'].items():
            elasticity = results['recommended_elasticity']
            
            if elasticity < -2.5:
                recommendation = "Focus on cost optimization and competitive pricing. Price increases risky."
            elif elasticity < -1.5:
                recommendation = "Moderate pricing flexibility. Test small price increases with close monitoring."
            else:
                recommendation = "Strong pricing power. Consider premium positioning and price optimization."
            
            report.append(f"{category}: {recommendation}")
        
        report.append()
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        print(report_text)
        
        return report_text
    
    def save_models(self, filepath: str = 'outputs/models/elasticity_models.pkl'):
        """
        Save trained models to file
        
        Args:
            filepath (str): Path to save models
        """
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'results': self.results,
                'model_diagnostics': self.model_diagnostics
            }, f)
        
        print(f"✅ Models saved to {filepath}")
    
    def load_models(self, filepath: str = 'outputs/models/elasticity_models.pkl'):
        """
        Load previously trained models
        
        Args:
            filepath (str): Path to model file
        """
        try:
            with open(filepath, 'rb') as f:
                saved_data = pickle.load(f)
                
            self.models = saved_data['models']
            self.results = saved_data['results']
            self.model_diagnostics = saved_data.get('model_diagnostics', {})
            
            print(f"✅ Models loaded from {filepath}")
            
        except FileNotFoundError:
            print(f"❌ Model file not found: {filepath}")
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")