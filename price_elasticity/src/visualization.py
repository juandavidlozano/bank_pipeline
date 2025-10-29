"""
Advanced Data Visualization Module for Price Elasticity Analysis

This module creates professional charts, heatmaps, interactive plots, and dashboards
for price elasticity insights. Includes publication-ready visualizations and 
business intelligence graphics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for professional visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ElasticityVisualizer:
    """
    Professional visualization suite for elasticity analysis
    """
    
    def __init__(self, output_dir='outputs/charts/'):
        """
        Initialize visualizer with output directory
        
        Args:
            output_dir (str): Directory to save charts
        """
        self.output_dir = output_dir
        self.figure_size = (12, 8)
        self.color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # Create output directory
        import os
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_demand_curves(self, demand_data: Dict, save_path: str = None) -> None:
        """
        Create demand curve visualization by product category
        
        Args:
            demand_data (dict): Demand curve data from DataProcessor
            save_path (str): Path to save the plot
        """
        fig, axes = plt.subplots(1, len(demand_data), figsize=(15, 5))
        if len(demand_data) == 1:
            axes = [axes]
            
        fig.suptitle('Price-Demand Curves by Product Category', fontsize=16, fontweight='bold')
        
        for idx, (category, data) in enumerate(demand_data.items()):
            ax = axes[idx]
            
            # Plot demand curve
            ax.plot(data['prices'], data['quantities'], 
                   linewidth=3, color=self.color_palette[idx % len(self.color_palette)],
                   label=f'{category} (E = {data["elasticity_coef"]:.2f})')
            
            # Add styling
            ax.set_xlabel('Price ($)', fontsize=12)
            ax.set_ylabel('Quantity Demanded', fontsize=12)
            ax.set_title(f'{category} Segment\nR² = {data["r_squared"]:.3f}', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add elasticity annotation
            mid_idx = len(data['prices']) // 2
            ax.annotate(f'Elasticity: {data["elasticity_coef"]:.2f}',
                       xy=(data['prices'][mid_idx], data['quantities'][mid_idx]),
                       xytext=(10, 20), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f'{self.output_dir}demand_curves.png', dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def create_elasticity_heatmap(self, elasticity_results: Dict, save_path: str = None) -> None:
        """
        Create elasticity heatmap showing own-price and cross-price elasticities
        
        Args:
            elasticity_results (dict): Results from elasticity analysis
            save_path (str): Path to save the plot
        """
        # Prepare data for heatmap
        categories = list(elasticity_results.keys())
        metrics = ['Own-Price', 'Competitor A', 'Competitor B', 'R-squared', 'Observations']
        
        heatmap_data = []
        
        for category in categories:
            results = elasticity_results[category]['log_linear_model']
            row = [
                results['own_price_elasticity'],
                results.get('cross_elasticity_comp_a', 0),
                results.get('cross_elasticity_comp_b', 0),
                results['r_squared'],
                results['n_observations'] / 1000  # Scale for better visualization
            ]
            heatmap_data.append(row)
        
        # Create DataFrame
        df_heatmap = pd.DataFrame(heatmap_data, 
                                 index=categories, 
                                 columns=metrics)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Custom colormap for elasticity values
        cmap = sns.diverging_palette(250, 10, as_cmap=True)
        
        sns.heatmap(df_heatmap, 
                   annot=True, 
                   fmt='.2f',
                   cmap=cmap,
                   center=0,
                   square=True,
                   linewidths=0.5,
                   cbar_kws={"shrink": .8},
                   ax=ax)
        
        ax.set_title('Price Elasticity Heatmap by Product Category', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Elasticity Metrics', fontsize=12)
        ax.set_ylabel('Product Categories', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f'{self.output_dir}elasticity_heatmap.png', dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def plot_price_sensitivity_analysis(self, simulation_results: Dict, save_path: str = None) -> None:
        """
        Create price sensitivity waterfall charts
        
        Args:
            simulation_results (dict): Simulation results from ElasticityAnalyzer
            save_path (str): Path to save the plot
        """
        fig, axes = plt.subplots(1, len(simulation_results), figsize=(18, 6))
        if len(simulation_results) == 1:
            axes = [axes]
            
        fig.suptitle('Revenue Impact Analysis by Price Changes', fontsize=16, fontweight='bold')
        
        for idx, (category, data) in enumerate(simulation_results.items()):
            ax = axes[idx]
            
            scenarios = list(data['scenarios'].keys())
            revenue_impacts = [data['scenarios'][s]['revenue_change_pct'] for s in scenarios]
            
            # Color bars based on positive/negative impact
            colors = ['green' if x > 0 else 'red' for x in revenue_impacts]
            
            bars = ax.bar(range(len(scenarios)), revenue_impacts, color=colors, alpha=0.7)
            
            # Add value labels on bars
            for i, (bar, impact) in enumerate(zip(bars, revenue_impacts)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height > 0 else -1),
                       f'{impact:+.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                       fontweight='bold')
            
            # Formatting
            ax.set_title(f'{category}\nElasticity: {data["elasticity_used"]:.2f}', fontsize=14)
            ax.set_xlabel('Price Change Scenarios', fontsize=12)
            ax.set_ylabel('Revenue Impact (%)', fontsize=12)
            ax.set_xticks(range(len(scenarios)))
            ax.set_xticklabels([s.replace('_', '\n') for s in scenarios], rotation=45, ha='right')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f'{self.output_dir}price_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def create_competitive_landscape(self, processed_data: pd.DataFrame, save_path: str = None) -> None:
        """
        Visualize competitive pricing landscape over time
        
        Args:
            processed_data (pd.DataFrame): Processed market data
            save_path (str): Path to save the plot
        """
        fig, axes = plt.subplots(len(processed_data['product_category'].unique()), 1, 
                                figsize=(14, 12))
        
        if len(processed_data['product_category'].unique()) == 1:
            axes = [axes]
            
        fig.suptitle('Competitive Pricing Landscape Over Time', fontsize=16, fontweight='bold')
        
        for idx, category in enumerate(processed_data['product_category'].unique()):
            ax = axes[idx]
            cat_data = processed_data[processed_data['product_category'] == category].copy()
            cat_data = cat_data.sort_values('date')
            
            # Plot price lines
            ax.plot(cat_data['date'], cat_data['price_own'], 
                   linewidth=3, label='Our Price', color='blue')
            ax.plot(cat_data['date'], cat_data['competitor_a_price'], 
                   linewidth=2, label='Competitor A', color='red', linestyle='--')
            ax.plot(cat_data['date'], cat_data['competitor_b_price'], 
                   linewidth=2, label='Competitor B', color='green', linestyle='--')
            ax.plot(cat_data['date'], cat_data['competitor_c_price'], 
                   linewidth=2, label='Competitor C', color='orange', linestyle='--')
            
            # Highlight promotional periods
            promo_periods = cat_data[cat_data['promotion_flag'] == 1]
            if not promo_periods.empty:
                ax.scatter(promo_periods['date'], promo_periods['price_own'], 
                          color='blue', s=100, marker='*', label='Promotions', zorder=5)
            
            # Formatting
            ax.set_title(f'{category} Segment Pricing', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Price ($)', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f'{self.output_dir}competitive_landscape.png', dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def plot_model_diagnostics(self, models: Dict, save_path: str = None) -> None:
        """
        Create model diagnostic plots (residuals, QQ plots, etc.)
        
        Args:
            models (dict): Trained models from ElasticityAnalyzer
            save_path (str): Path to save the plot
        """
        n_models = len(models)
        fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))
        
        if n_models == 1:
            axes = axes.reshape(-1, 1)
            
        fig.suptitle('Model Diagnostic Plots', fontsize=16, fontweight='bold')
        
        for idx, (model_name, model) in enumerate(models.items()):
            if hasattr(model, 'resid') and hasattr(model, 'fittedvalues'):
                # Residuals vs Fitted
                ax1 = axes[0, idx]
                ax1.scatter(model.fittedvalues, model.resid, alpha=0.6)
                ax1.axhline(y=0, color='red', linestyle='--')
                ax1.set_xlabel('Fitted Values')
                ax1.set_ylabel('Residuals')
                ax1.set_title(f'{model_name}\nResiduals vs Fitted')
                ax1.grid(True, alpha=0.3)
                
                # Q-Q Plot
                ax2 = axes[1, idx]
                from scipy import stats
                stats.probplot(model.resid, dist="norm", plot=ax2)
                ax2.set_title(f'{model_name}\nQ-Q Plot')
                ax2.grid(True, alpha=0.3)
            else:
                # Placeholder for models without residuals
                axes[0, idx].text(0.5, 0.5, f'{model_name}\nDiagnostics\nNot Available', 
                                 ha='center', va='center', transform=axes[0, idx].transAxes)
                axes[1, idx].text(0.5, 0.5, 'Model Type\nNot Supported', 
                                 ha='center', va='center', transform=axes[1, idx].transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f'{self.output_dir}model_diagnostics.png', dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def create_executive_dashboard(self, full_results: Dict, save_path: str = None) -> None:
        """
        Create comprehensive executive dashboard
        
        Args:
            full_results (dict): Complete analysis results
            save_path (str): Path to save the plot
        """
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Price Elasticity Executive Dashboard', fontsize=20, fontweight='bold', y=0.95)
        
        # 1. Elasticity Overview (top left)
        ax1 = fig.add_subplot(gs[0, :2])
        categories = list(full_results['category_results'].keys())
        elasticities = [full_results['category_results'][cat]['recommended_elasticity'] 
                       for cat in categories]
        
        bars = ax1.barh(categories, elasticities, color=self.color_palette[:len(categories)])
        ax1.set_xlabel('Price Elasticity')
        ax1.set_title('Price Elasticity by Category', fontweight='bold')
        ax1.axvline(x=-1, color='red', linestyle='--', alpha=0.5, label='Unit Elastic')
        
        # Add value labels
        for i, (bar, elasticity) in enumerate(zip(bars, elasticities)):
            ax1.text(elasticity - 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{elasticity:.2f}', ha='right', va='center', fontweight='bold')
        
        # 2. Model Fit Quality (top right)
        ax2 = fig.add_subplot(gs[0, 2:])
        r_squareds = [full_results['category_results'][cat]['log_linear_model']['r_squared'] 
                     for cat in categories]
        
        colors = ['green' if r2 > 0.8 else 'orange' if r2 > 0.6 else 'red' for r2 in r_squareds]
        bars2 = ax2.bar(categories, r_squareds, color=colors, alpha=0.7)
        ax2.set_ylabel('R-squared')
        ax2.set_title('Model Fit Quality', fontweight='bold')
        ax2.set_ylim(0, 1)
        
        # Add threshold lines
        ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Strong Fit')
        ax2.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Moderate Fit')
        
        # 3. Revenue Impact Matrix (middle)
        ax3 = fig.add_subplot(gs[1, :])
        
        scenarios = ['Price_Increase_5pct', 'Price_Increase_10pct', 
                    'Price_Decrease_5pct', 'Price_Decrease_10pct']
        scenario_labels = ['+5%', '+10%', '-5%', '-10%']
        
        impact_matrix = []
        for category in categories:
            row = []
            for scenario in scenarios:
                if category in full_results['simulation_results']:
                    impact = full_results['simulation_results'][category]['scenarios'][scenario]['revenue_change_pct']
                    row.append(impact)
                else:
                    row.append(0)
            impact_matrix.append(row)
        
        im = ax3.imshow(impact_matrix, cmap='RdYlGn', aspect='auto', vmin=-20, vmax=20)
        ax3.set_xticks(range(len(scenario_labels)))
        ax3.set_xticklabels(scenario_labels)
        ax3.set_yticks(range(len(categories)))
        ax3.set_yticklabels(categories)
        ax3.set_xlabel('Price Change Scenarios')
        ax3.set_title('Revenue Impact Matrix (%)', fontweight='bold')
        
        # Add text annotations
        for i in range(len(categories)):
            for j in range(len(scenarios)):
                ax3.text(j, i, f'{impact_matrix[i][j]:.1f}%', 
                        ha='center', va='center', fontweight='bold',
                        color='white' if abs(impact_matrix[i][j]) > 10 else 'black')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
        cbar.set_label('Revenue Impact (%)', rotation=270, labelpad=15)
        
        # 4. Key Insights (bottom)
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        insights_text = "KEY INSIGHTS:\n\n"
        
        # Most/least sensitive categories
        min_elasticity_cat = min(categories, key=lambda x: full_results['category_results'][x]['recommended_elasticity'])
        max_elasticity_cat = max(categories, key=lambda x: full_results['category_results'][x]['recommended_elasticity'])
        
        insights_text += f"• Most Price Sensitive: {max_elasticity_cat} (E = {full_results['category_results'][max_elasticity_cat]['recommended_elasticity']:.2f})\n"
        insights_text += f"• Least Price Sensitive: {min_elasticity_cat} (E = {full_results['category_results'][min_elasticity_cat]['recommended_elasticity']:.2f})\n\n"
        
        # Revenue optimization opportunities
        best_increase_cat = None
        best_increase_impact = -float('inf')
        
        for category in categories:
            if category in full_results['simulation_results']:
                impact = full_results['simulation_results'][category]['scenarios']['Price_Increase_5pct']['revenue_change_pct']
                if impact > best_increase_impact:
                    best_increase_impact = impact
                    best_increase_cat = category
        
        if best_increase_cat:
            insights_text += f"• Best Price Increase Opportunity: {best_increase_cat} (+{best_increase_impact:.1f}% revenue for +5% price)\n"
        
        insights_text += f"• Analysis Date: {full_results['summary']['analysis_date']}\n"
        insights_text += f"• Total Observations: {full_results['summary']['total_observations']:,}"
        
        ax4.text(0.05, 0.95, insights_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f'{self.output_dir}executive_dashboard.png', dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def export_charts_summary(self) -> str:
        """
        Generate summary of all created visualizations
        
        Returns:
            str: Summary text of generated charts
        """
        import os
        
        chart_files = [f for f in os.listdir(self.output_dir) if f.endswith('.png')]
        
        summary = "📊 VISUALIZATION SUMMARY\n"
        summary += "=" * 25 + "\n\n"
        summary += f"Output Directory: {self.output_dir}\n"
        summary += f"Total Charts Generated: {len(chart_files)}\n\n"
        
        summary += "Generated Visualizations:\n"
        for i, chart in enumerate(chart_files, 1):
            summary += f"{i}. {chart}\n"
        
        summary += "\n🎯 Chart Descriptions:\n"
        summary += "• demand_curves.png: Price-demand relationships by category\n"
        summary += "• elasticity_heatmap.png: Elasticity coefficients matrix\n"
        summary += "• price_sensitivity_analysis.png: Revenue impact scenarios\n"
        summary += "• competitive_landscape.png: Competitive pricing over time\n"
        summary += "• model_diagnostics.png: Statistical model validation\n"
        summary += "• executive_dashboard.png: Comprehensive business overview\n"
        
        print(summary)
        return summary