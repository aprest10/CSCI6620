import pandas as pd
import numpy as np
import re
import os
import glob

class SparseMatrixAnalysis:
    def __init__(self, df=None):
        self.raw_df = None
        self.ranked_df = None
        self.optimal_df = None
        self.performance_analysis = None
        
        if df is not None:
            self.raw_df = df.copy()
        
        if self.raw_df is not None:
            self._process_data()
    
    def _process_data(self):
        if self.raw_df is None:
            raise ValueError("No data loaded, pass in a DataFrame.")
            
        self.extract_matrix_info()
        self.rank_formats()
        self.analyze_format_performance()
        
        self.optimal_df = self.ranked_df[self.ranked_df['optimal']].copy()
    
    def extract_matrix_info(self):
        if self.raw_df is None:
            raise ValueError("No data loaded, pass in a DataFrame.")
        
        pattern = r'.*?matrix_size(\d+)_([a-zA-Z]+)_.*?\.mtx'
        
        matrix_sizes = []
        matrix_types = []
        
        for filename in self.raw_df['filename']:
            match = re.search(pattern, filename)
            if match:
                size = int(match.group(1))
                matrix_type = match.group(2).lower()
                matrix_sizes.append(size)
                matrix_types.append(matrix_type)
            else:
                matrix_sizes.append(np.nan)
                matrix_types.append('unknown')
        
        self.raw_df['matrix_size'] = matrix_sizes
        self.raw_df['matrix_type'] = matrix_types
        
        self.raw_df = self.raw_df.dropna(subset=['matrix_size'])
        self.raw_df = self.raw_df[self.raw_df['matrix_type'] != 'unknown']
        
        return self
    
    def rank_formats(self):
        if self.raw_df is None:
            raise ValueError("No data loaded, pass in a DataFrame.")
        
        df = self.raw_df.copy()
        
        df['matrix_id'] = df['filename'].str.extract(r'(matrix_size\d+_[\w]+)')
        
        required_metrics = ['conversion_time', 'memory_usage_mb', 'matmul_time_ms', 'gpu_utilization']
        for metric in required_metrics:
            if metric not in df.columns:
                raise ValueError(f"Missing required metric column: {metric}")
        
        for metric in required_metrics:
            df[f'{metric}_rank'] = df.groupby('matrix_id')[metric].rank(ascending=True)
        
        df['overall_rank'] = df[[f'{metric}_rank' for metric in required_metrics]].mean(axis=1)
        
        idx = df.groupby('matrix_id')['overall_rank'].idxmin()
        df['optimal'] = False
        df.loc[idx, 'optimal'] = True
        
        self.ranked_df = df
        return self
    
    def analyze_format_performance(self):
        if self.ranked_df is None:
            raise ValueError("Ranked data not available. Call rank_formats() first.")
            
        df = self.ranked_df
        metrics = ['conversion_time', 'memory_usage_mb', 'matmul_time_ms', 'gpu_utilization']
        format_performance = {}
        
        for format_name in df['format'].unique():
            format_df = df[df['format'] == format_name]
            
            format_ranks = {
                f'{metric}_rank': format_df[f'{metric}_rank'].mean()
                for metric in metrics
            }
            
            format_metrics = {
                f'{metric}_mean': format_df[metric].mean()
                for metric in metrics
            }
            
            format_performance[format_name] = {
                **format_ranks,
                **format_metrics,
                'optimal_count': format_df['optimal'].sum(),
                'total_count': len(format_df)
            }
        
        perf_df = pd.DataFrame(format_performance).T
        
        perf_df['overall_rank'] = perf_df[[f'{metric}_rank' for metric in metrics]].mean(axis=1)
        perf_df = perf_df.sort_values('overall_rank')
        
        self.performance_analysis = perf_df
        return perf_df
    
    def get_optimal_formats(self):
        if self.optimal_df is None:
            raise ValueError("Optimal formats not available. Process data first.")
        return self.optimal_df.copy()
    
    def print_analysis_summary(self):
        if self.performance_analysis is None:
            raise ValueError("Performance analysis not available. Process data first.")
            
        print("\n======= FORMAT PERFORMANCE ANALYSIS =======")
        print("Average Ranks (lower is better):")
        rank_cols = [col for col in self.performance_analysis.columns if col.endswith('_rank')]
        print(self.performance_analysis[rank_cols])
        
        print("\nAverage Metric Values:")
        metric_cols = [col for col in self.performance_analysis.columns if col.endswith('_mean')]
        print(self.performance_analysis[metric_cols])
        
        print("\nOptimal Format Counts:")
        print(self.performance_analysis[['optimal_count', 'total_count']])
        print("===========================================")