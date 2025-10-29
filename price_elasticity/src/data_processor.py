"""
Data Processing Module for Price Elasticity Analysis

This module handles data loading, cleaning, preprocessing, and feature engineering
for price elasticity modeling. Includes advanced data quality checks and 
statistical transformations.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """
    Comprehensive data processing pipeline for elasticity analysis
    """
    
    def __init__(self, data_path='data/'):
        """
        Initialize DataProcessor with data directory path
        
        Args:
            data_path (str): Path to data directory
        """
        self.data_path = data_path
        self.market_data = None
        self.competitor_data = None
        self.external_data = None
        self.customer_data = None
        self.processed_data = None
        self.scaler = StandardScaler()
        
    def load_all_data(self):
        """
        Load all dataset files and perform initial validation
        
        Returns:
            dict: Dictionary containing all loaded datasets
        """
        try:
            # Load primary datasets
            self.market_data = pd.read_csv(f'{self.data_path}market_data.csv')
            self.competitor_data = pd.read_csv(f'{self.data_path}competitor_data.csv')
            self.external_data = pd.read_csv(f'{self.data_path}external_factors.csv')
            self.customer_data = pd.read_csv(f'{self.data_path}customer_segments.csv')
            
            # Convert date columns
            self.market_data['date'] = pd.to_datetime(self.market_data['date'])
            self.competitor_data['date'] = pd.to_datetime(self.competitor_data['date'])
            self.external_data['date'] = pd.to_datetime(self.external_data['date'])
            
            print("✅ All datasets loaded successfully")
            return {
                'market': self.market_data,
                'competitor': self.competitor_data,
                'external': self.external_data,
                'customer': self.customer_data
            }
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return None
    
    def data_quality_check(self):
        """
        Perform comprehensive data quality assessment
        
        Returns:
            dict: Data quality metrics and issues
        """
        quality_report = {}
        
        if self.market_data is not None:
            # Missing values analysis
            missing_values = self.market_data.isnull().sum()
            quality_report['missing_values'] = missing_values[missing_values > 0].to_dict()
            
            # Outlier detection using IQR method
            numeric_cols = self.market_data.select_dtypes(include=[np.number]).columns
            outliers = {}
            
            for col in numeric_cols:
                Q1 = self.market_data[col].quantile(0.25)
                Q3 = self.market_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_count = ((self.market_data[col] < lower_bound) | 
                               (self.market_data[col] > upper_bound)).sum()
                outliers[col] = outlier_count
                
            quality_report['outliers'] = outliers
            
            # Data consistency checks
            consistency_issues = []
            
            # Check for negative prices or quantities
            if (self.market_data['price_own'] < 0).any():
                consistency_issues.append("Negative prices detected")
            if (self.market_data['quantity_sold'] < 0).any():
                consistency_issues.append("Negative quantities detected")
                
            quality_report['consistency_issues'] = consistency_issues
            
        return quality_report
    
    def create_elasticity_features(self):
        """
        Engineer features specifically for elasticity modeling
        
        Returns:
            pd.DataFrame: Enhanced dataset with elasticity features
        """
        if self.market_data is None or self.competitor_data is None:
            raise ValueError("Market and competitor data must be loaded first")
            
        # Merge datasets
        merged_data = pd.merge(self.market_data, self.competitor_data, 
                              on=['date', 'market_category'], how='left',
                              suffixes=('', '_comp'))
        
        if self.external_data is not None:
            merged_data = pd.merge(merged_data, self.external_data, 
                                 on='date', how='left')
        
        # Create price differential features
        merged_data['price_diff_comp_a'] = merged_data['price_own'] - merged_data['competitor_a_price']
        merged_data['price_diff_comp_b'] = merged_data['price_own'] - merged_data['competitor_b_price']
        merged_data['price_diff_comp_c'] = merged_data['price_own'] - merged_data['competitor_c_price']
        
        # Relative price positioning
        merged_data['relative_price_comp_a'] = merged_data['price_own'] / merged_data['competitor_a_price']
        merged_data['relative_price_comp_b'] = merged_data['price_own'] / merged_data['competitor_b_price']
        merged_data['relative_price_comp_c'] = merged_data['price_own'] / merged_data['competitor_c_price']
        
        # Lagged price variables (for dynamic effects)
        for category in merged_data['product_category'].unique():
            category_mask = merged_data['product_category'] == category
            merged_data.loc[category_mask, 'price_lag_1'] = merged_data.loc[category_mask, 'price_own'].shift(1)
            merged_data.loc[category_mask, 'price_lag_2'] = merged_data.loc[category_mask, 'price_own'].shift(2)
            merged_data.loc[category_mask, 'quantity_lag_1'] = merged_data.loc[category_mask, 'quantity_sold'].shift(1)
        
        # Price change indicators
        merged_data['price_change'] = merged_data.groupby('product_category')['price_own'].pct_change()
        merged_data['price_change_abs'] = merged_data['price_change'].abs()
        
        # Competitive pressure index
        merged_data['competitive_pressure'] = (
            (merged_data['competitor_a_promo'] * 0.4) +
            (merged_data['competitor_b_promo'] * 0.3) +
            (merged_data['competitor_c_promo'] * 0.3)
        )
        
        # Market share dynamics
        merged_data['market_share_lag'] = merged_data.groupby('product_category')['market_share'].shift(1)
        merged_data['market_share_change'] = merged_data['market_share'] - merged_data['market_share_lag']
        
        # Price elasticity of demand calculation (preliminary)
        merged_data['log_price'] = np.log(merged_data['price_own'])
        merged_data['log_quantity'] = np.log(merged_data['quantity_sold'] + 1)  # +1 to avoid log(0)
        
        # Interaction terms
        merged_data['price_seasonality'] = merged_data['price_own'] * merged_data['seasonality_index']
        merged_data['promotion_seasonality'] = merged_data['promotion_flag'] * merged_data['seasonality_index']
        
        # Store processed data
        self.processed_data = merged_data
        
        print("✅ Elasticity features created successfully")
        print(f"📊 Dataset shape: {merged_data.shape}")
        print(f"📈 Features created: {merged_data.columns.tolist()}")
        
        return merged_data
    
    def scale_features(self, features_to_scale=None):
        """
        Scale numerical features for modeling
        
        Args:
            features_to_scale (list): List of features to scale. If None, scales all numeric features
            
        Returns:
            pd.DataFrame: Dataset with scaled features
        """
        if self.processed_data is None:
            raise ValueError("Must create elasticity features first")
            
        if features_to_scale is None:
            # Select numerical features to scale
            features_to_scale = ['price_own', 'competitor_a_price', 'competitor_b_price', 
                               'competitor_c_price', 'gdp_growth', 'inflation_rate', 
                               'consumer_confidence', 'advertising_spend']
            
        scaled_data = self.processed_data.copy()
        
        # Apply standardization
        scaled_features = self.scaler.fit_transform(scaled_data[features_to_scale])
        scaled_df = pd.DataFrame(scaled_features, columns=features_to_scale, 
                               index=scaled_data.index)
        
        # Replace original features with scaled versions
        for feature in features_to_scale:
            scaled_data[f'{feature}_scaled'] = scaled_df[feature]
            
        print(f"✅ Scaled {len(features_to_scale)} features")
        return scaled_data
    
    def create_demand_curves(self, category=None):
        """
        Generate demand curve data points for visualization and analysis
        
        Args:
            category (str): Product category to analyze. If None, analyzes all categories
            
        Returns:
            dict: Demand curve data by category
        """
        if self.processed_data is None:
            raise ValueError("Must process data first")
            
        demand_curves = {}
        categories = [category] if category else self.processed_data['product_category'].unique()
        
        for cat in categories:
            cat_data = self.processed_data[self.processed_data['product_category'] == cat].copy()
            
            # Create price bins for demand curve
            price_range = np.linspace(cat_data['price_own'].min() * 0.8,
                                    cat_data['price_own'].max() * 1.2, 50)
            
            # Estimate demand for each price point using simple regression
            from sklearn.linear_model import LinearRegression
            
            # Fit simple demand model
            X = cat_data[['price_own', 'seasonality_index', 'promotion_flag']].fillna(0)
            y = cat_data['quantity_sold']
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Predict demand across price range
            demand_estimates = []
            for price in price_range:
                # Use average seasonality and no promotion for baseline curve
                features = [[price, cat_data['seasonality_index'].mean(), 0]]
                demand = model.predict(features)[0]
                demand_estimates.append(max(0, demand))  # Ensure non-negative demand
                
            demand_curves[cat] = {
                'prices': price_range,
                'quantities': demand_estimates,
                'elasticity_coef': model.coef_[0],  # Price coefficient
                'r_squared': model.score(X, y)
            }
            
        return demand_curves
    
    def get_data_summary(self):
        """
        Generate comprehensive data summary statistics
        
        Returns:
            dict: Summary statistics and insights
        """
        if self.processed_data is None:
            return {"error": "No processed data available"}
            
        summary = {
            'dataset_info': {
                'total_observations': len(self.processed_data),
                'date_range': f"{self.processed_data['date'].min()} to {self.processed_data['date'].max()}",
                'categories': self.processed_data['product_category'].unique().tolist(),
                'total_features': len(self.processed_data.columns)
            },
            'price_statistics': {},
            'elasticity_insights': {},
            'competitive_landscape': {}
        }
        
        # Price statistics by category
        for category in self.processed_data['product_category'].unique():
            cat_data = self.processed_data[self.processed_data['product_category'] == category]
            
            summary['price_statistics'][category] = {
                'avg_price': round(cat_data['price_own'].mean(), 2),
                'price_std': round(cat_data['price_own'].std(), 2),
                'min_price': round(cat_data['price_own'].min(), 2),
                'max_price': round(cat_data['price_own'].max(), 2),
                'avg_quantity': round(cat_data['quantity_sold'].mean(), 0),
                'total_revenue': round(cat_data['revenue'].sum(), 2)
            }
            
            # Basic elasticity calculation
            if len(cat_data) > 5:  # Need sufficient data points
                price_changes = cat_data['price_change'].dropna()
                quantity_changes = cat_data.groupby('product_category')['quantity_sold'].pct_change().dropna()
                
                if len(price_changes) > 0 and len(quantity_changes) > 0:
                    # Simple elasticity approximation
                    elasticity = (quantity_changes / price_changes).replace([np.inf, -np.inf], np.nan).mean()
                    summary['elasticity_insights'][category] = {
                        'estimated_elasticity': round(elasticity, 2) if not np.isnan(elasticity) else 'N/A',
                        'price_volatility': round(price_changes.std(), 3),
                        'demand_volatility': round(quantity_changes.std(), 3)
                    }
        
        return summary