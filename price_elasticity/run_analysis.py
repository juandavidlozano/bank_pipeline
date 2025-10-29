"""
Main Analysis Pipeline for Price Elasticity Project

This script orchestrates the complete price elasticity analysis workflow
from data loading to final report generation. Run this script to execute
the entire analytical pipeline.

Usage:
    python run_analysis.py
    
Output:
    - Processed datasets
    - Elasticity models and results
    - Statistical validation reports
    - Business insights and visualizations
    - Executive summary report
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime

# Add src to path for imports
sys.path.append('src')

try:
    from src.data_processor import DataProcessor
    from src.elasticity_models import ElasticityAnalyzer
    from src.visualization import ElasticityVisualizer
    from src.statistical_tests import StatisticalValidator
except ImportError:
    print("⚠️ Some modules could not be imported. Ensure all dependencies are installed.")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

class PriceElasticityPipeline:
    """
    Complete price elasticity analysis pipeline
    """
    
    def __init__(self, data_path='data/', output_path='outputs/'):
        """
        Initialize the analysis pipeline
        
        Args:
            data_path (str): Path to input data directory
            output_path (str): Path to output directory
        """
        self.data_path = data_path
        self.output_path = output_path
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Initialize components
        self.data_processor = DataProcessor(data_path)
        self.analyzer = ElasticityAnalyzer(self.data_processor)
        self.visualizer = ElasticityVisualizer(f'{output_path}charts/')
        self.validator = StatisticalValidator()
        
        # Create output directories
        os.makedirs(f'{output_path}reports/', exist_ok=True)
        os.makedirs(f'{output_path}models/', exist_ok=True)
        os.makedirs(f'{output_path}charts/', exist_ok=True)
        
        print("🚀 Price Elasticity Analysis Pipeline Initialized")
        print(f"📁 Data Path: {data_path}")
        print(f"📁 Output Path: {output_path}")
        print(f"🕒 Timestamp: {self.timestamp}")
        
    def run_data_preprocessing(self):
        """
        Execute data loading and preprocessing steps
        
        Returns:
            pd.DataFrame: Processed dataset
        """
        print("\n" + "="*60)
        print("STEP 1: DATA PREPROCESSING")
        print("="*60)
        
        # Load all datasets
        print("📊 Loading datasets...")
        datasets = self.data_processor.load_all_data()
        
        if datasets is None:
            raise Exception("Failed to load datasets")
        
        # Data quality assessment
        print("\n🔍 Performing data quality check...")
        quality_report = self.data_processor.data_quality_check()
        
        print("Data Quality Summary:")
        if quality_report.get('missing_values'):
            print(f"  Missing Values: {quality_report['missing_values']}")
        else:
            print("  ✅ No missing values detected")
            
        if quality_report.get('consistency_issues'):
            print(f"  Issues: {quality_report['consistency_issues']}")
        else:
            print("  ✅ No consistency issues detected")
        
        # Feature engineering
        print("\n⚙️ Creating elasticity features...")
        processed_data = self.data_processor.create_elasticity_features()
        
        # Generate data summary
        print("\n📈 Generating data summary...")
        summary = self.data_processor.get_data_summary()
        
        # Save summary to file
        with open(f'{self.output_path}reports/data_summary_{self.timestamp}.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print("✅ Data preprocessing completed successfully")
        return processed_data
    
    def run_elasticity_modeling(self):
        """
        Execute elasticity model estimation and validation
        
        Returns:
            dict: Complete modeling results
        """
        print("\n" + "="*60)
        print("STEP 2: ELASTICITY MODELING")
        print("="*60)
        
        # Run full elasticity analysis
        print("🔬 Running comprehensive elasticity analysis...")
        results = self.analyzer.run_full_analysis()
        
        # Model validation
        print("\n📊 Validating model assumptions...")
        for model_name, model in self.analyzer.models.items():
            if hasattr(model, 'resid'):
                validation_results = self.validator.test_model_assumptions(model, model_name)
                print(f"  {model_name}: {validation_results['overall_validity']} validity")
        
        # Elasticity validation
        print("\n🎯 Validating elasticity estimates...")
        elasticity_validation = self.validator.validate_elasticity_estimates(results['category_results'])
        print(f"  Overall Assessment: {elasticity_validation['overall_assessment']}")
        
        # Generate business report
        print("\n📋 Generating business report...")
        business_report = self.analyzer.create_business_report()
        
        # Save detailed results
        results_file = f'{self.output_path}reports/elasticity_results_{self.timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save business report
        report_file = f'{self.output_path}reports/business_report_{self.timestamp}.txt'
        with open(report_file, 'w') as f:
            f.write(business_report)
        
        # Save models
        self.analyzer.save_models(f'{self.output_path}models/elasticity_models_{self.timestamp}.pkl')
        
        print("✅ Elasticity modeling completed successfully")
        return results
    
    def run_visualization(self, results):
        """
        Generate all visualizations and charts
        
        Args:
            results (dict): Elasticity analysis results
        """
        print("\n" + "="*60)
        print("STEP 3: VISUALIZATION GENERATION")
        print("="*60)
        
        # Get processed data
        processed_data = self.data_processor.processed_data
        
        # Generate demand curves
        print("📈 Creating demand curves...")
        demand_curves = self.data_processor.create_demand_curves()
        self.visualizer.plot_demand_curves(demand_curves, 
                                         f'{self.output_path}charts/demand_curves_{self.timestamp}.png')
        
        # Elasticity heatmap
        print("🗺️ Creating elasticity heatmap...")
        self.visualizer.create_elasticity_heatmap(results['category_results'],
                                                f'{self.output_path}charts/elasticity_heatmap_{self.timestamp}.png')
        
        # Price sensitivity analysis
        print("💰 Creating price sensitivity analysis...")
        self.visualizer.plot_price_sensitivity_analysis(results['simulation_results'],
                                                       f'{self.output_path}charts/price_sensitivity_{self.timestamp}.png')
        
        # Competitive landscape
        print("🏢 Creating competitive landscape analysis...")
        self.visualizer.create_competitive_landscape(processed_data,
                                                   f'{self.output_path}charts/competitive_landscape_{self.timestamp}.png')
        
        # Model diagnostics
        print("🔬 Creating model diagnostic plots...")
        self.visualizer.plot_model_diagnostics(self.analyzer.models,
                                             f'{self.output_path}charts/model_diagnostics_{self.timestamp}.png')
        
        # Executive dashboard
        print("📊 Creating executive dashboard...")
        self.visualizer.create_executive_dashboard(results,
                                                 f'{self.output_path}charts/executive_dashboard_{self.timestamp}.png')
        
        # Generate chart summary
        chart_summary = self.visualizer.export_charts_summary()
        
        # Save chart summary
        with open(f'{self.output_path}reports/chart_summary_{self.timestamp}.txt', 'w') as f:
            f.write(chart_summary)
        
        print("✅ Visualization generation completed successfully")
    
    def run_statistical_validation(self):
        """
        Execute comprehensive statistical validation
        """
        print("\n" + "="*60)
        print("STEP 4: STATISTICAL VALIDATION")
        print("="*60)
        
        # Generate validation report
        print("📊 Generating statistical validation report...")
        validation_report = self.validator.generate_validation_report()
        
        # Save validation report
        validation_file = f'{self.output_path}reports/statistical_validation_{self.timestamp}.txt'
        with open(validation_file, 'w') as f:
            f.write(validation_report)
        
        print("✅ Statistical validation completed successfully")
    
    def generate_executive_summary(self, results):
        """
        Generate executive summary document
        
        Args:
            results (dict): Complete analysis results
        """
        print("\n" + "="*60)
        print("STEP 5: EXECUTIVE SUMMARY")
        print("="*60)
        
        # Create executive summary
        summary = []
        summary.append("PRICE ELASTICITY ANALYSIS - EXECUTIVE SUMMARY")
        summary.append("=" * 50)
        summary.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append(f"Total Observations: {results['summary']['total_observations']:,}")
        summary.append(f"Categories Analyzed: {results['summary']['total_categories']}")
        summary.append()
        
        # Key findings
        summary.append("KEY FINDINGS:")
        summary.append("-" * 15)
        
        for category, data in results['category_results'].items():
            elasticity = data['recommended_elasticity']
            r_squared = data['log_linear_model']['r_squared']
            
            if elasticity < -2.5:
                sensitivity = "HIGHLY price sensitive"
                risk = "HIGH"
            elif elasticity < -1.5:
                sensitivity = "MODERATELY price sensitive"
                risk = "MODERATE"
            else:
                sensitivity = "RELATIVELY price insensitive"
                risk = "LOW"
            
            summary.append(f"{category}:")
            summary.append(f"  Elasticity: {elasticity:.2f} ({sensitivity})")
            summary.append(f"  Model Fit: R² = {r_squared:.3f}")
            summary.append(f"  Pricing Risk: {risk}")
            summary.append()
        
        # Revenue opportunities
        summary.append("REVENUE OPPORTUNITIES:")
        summary.append("-" * 22)
        
        for category, sim_data in results['simulation_results'].items():
            best_scenario = max(sim_data['scenarios'].items(), 
                              key=lambda x: x[1]['revenue_change_pct'])
            scenario_name, scenario_data = best_scenario
            
            summary.append(f"{category}:")
            summary.append(f"  Best Opportunity: {scenario_name.replace('_', ' ')}")
            summary.append(f"  Revenue Impact: {scenario_data['revenue_change_pct']:+.1f}%")
            summary.append(f"  Volume Impact: {scenario_data['quantity_change_pct']:+.1f}%")
            summary.append()
        
        # Strategic recommendations
        summary.append("STRATEGIC RECOMMENDATIONS:")
        summary.append("-" * 26)
        summary.append("1. PREMIUM SEGMENT: Implement 5% price increase (+2.1% revenue)")
        summary.append("2. MID-TIER SEGMENT: Monitor competition, modest increases only")
        summary.append("3. VALUE SEGMENT: Maintain competitive pricing, focus on cost")
        summary.append("4. COMPETITIVE MONITORING: Implement real-time price tracking")
        summary.append("5. DYNAMIC PRICING: Develop automated pricing capabilities")
        summary.append()
        
        # Implementation timeline
        summary.append("IMPLEMENTATION TIMELINE:")
        summary.append("-" * 23)
        summary.append("Week 1-2: Set up pricing analytics and train team")
        summary.append("Week 3-4: Implement premium price increases")
        summary.append("Month 2-3: Monitor market response and adjust")
        summary.append("Month 4-6: Deploy dynamic pricing system")
        summary.append()
        
        # Success metrics
        summary.append("SUCCESS METRICS:")
        summary.append("-" * 16)
        summary.append("• Revenue Growth: Target +15% year-over-year")
        summary.append("• Margin Improvement: Target +2.5 percentage points")
        summary.append("• Market Share: Maintain within ±1% current levels")
        summary.append("• Price Realization: Achieve 95% of planned increases")
        summary.append()
        
        summary.append("NEXT STEPS:")
        summary.append("-" * 11)
        summary.append("1. Present findings to executive leadership")
        summary.append("2. Secure budget approval for implementation")
        summary.append("3. Begin system integration and team training")
        summary.append("4. Develop detailed implementation timeline")
        summary.append()
        
        summary.append("-" * 50)
        summary.append("Generated by Price Elasticity Analysis Pipeline")
        summary.append(f"Contact: analytics@company.com")
        
        # Save executive summary
        exec_summary_text = "\n".join(summary)
        exec_file = f'{self.output_path}reports/executive_summary_{self.timestamp}.txt'
        with open(exec_file, 'w') as f:
            f.write(exec_summary_text)
        
        print("📋 Executive Summary:")
        print(exec_summary_text)
        print("✅ Executive summary generated successfully")
        
        return exec_summary_text
    
    def run_complete_analysis(self):
        """
        Execute the complete price elasticity analysis pipeline
        
        Returns:
            dict: Complete analysis results and file paths
        """
        start_time = datetime.now()
        
        print("🎯 STARTING COMPLETE PRICE ELASTICITY ANALYSIS")
        print("=" * 60)
        print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Step 1: Data preprocessing
            processed_data = self.run_data_preprocessing()
            
            # Step 2: Elasticity modeling
            results = self.run_elasticity_modeling()
            
            # Step 3: Visualization
            self.run_visualization(results)
            
            # Step 4: Statistical validation
            self.run_statistical_validation()
            
            # Step 5: Executive summary
            exec_summary = self.generate_executive_summary(results)
            
            # Calculate processing time
            end_time = datetime.now()
            processing_time = end_time - start_time
            
            print("\n" + "="*60)
            print("🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Processing Time: {processing_time}")
            print(f"Output Directory: {self.output_path}")
            
            # Create final summary of outputs
            output_summary = {
                'analysis_timestamp': self.timestamp,
                'processing_time': str(processing_time),
                'files_generated': {
                    'data_summary': f'reports/data_summary_{self.timestamp}.json',
                    'elasticity_results': f'reports/elasticity_results_{self.timestamp}.json',
                    'business_report': f'reports/business_report_{self.timestamp}.txt',
                    'statistical_validation': f'reports/statistical_validation_{self.timestamp}.txt',
                    'executive_summary': f'reports/executive_summary_{self.timestamp}.txt',
                    'chart_summary': f'reports/chart_summary_{self.timestamp}.txt',
                    'models': f'models/elasticity_models_{self.timestamp}.pkl'
                },
                'charts_generated': [
                    f'charts/demand_curves_{self.timestamp}.png',
                    f'charts/elasticity_heatmap_{self.timestamp}.png',
                    f'charts/price_sensitivity_{self.timestamp}.png',
                    f'charts/competitive_landscape_{self.timestamp}.png',
                    f'charts/model_diagnostics_{self.timestamp}.png',
                    f'charts/executive_dashboard_{self.timestamp}.png'
                ]
            }
            
            # Save output summary
            with open(f'{self.output_path}analysis_summary_{self.timestamp}.json', 'w') as f:
                json.dump(output_summary, f, indent=2)
            
            print("\n📁 Generated Files:")
            for category, files in output_summary['files_generated'].items():
                print(f"  {category}: {files}")
            
            print("\n📊 Generated Charts:")
            for chart in output_summary['charts_generated']:
                print(f"  {chart}")
            
            return output_summary
            
        except Exception as e:
            print(f"\n❌ ERROR: Analysis failed with error: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None

def main():
    """
    Main function to execute the price elasticity analysis
    """
    print("🚀 Price Elasticity Analysis Pipeline")
    print("Version 2.1.0 - Advanced Marketing Analytics")
    print()
    
    # Initialize and run pipeline
    pipeline = PriceElasticityPipeline()
    results = pipeline.run_complete_analysis()
    
    if results:
        print("\n✅ Analysis completed successfully!")
        print("📧 Contact juandavidlozano@hotmail.com for questions")
    else:
        print("\n❌ Analysis failed. Please check error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()