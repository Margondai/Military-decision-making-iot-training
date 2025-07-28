#!/usr/bin/env python3
"""
Adaptive Simulation-Based Training for Military Decision-Making: 
Leveraging IoT-Derived Cognitive and Emotional Feedback

Authors: Anamaria Acevedo Diaz, Ancuta Margondai, et al.
Institution: University of Central Florida
Conference: MODSIM World 2025
"""

import numpy as np
import pandas as pd
import mne
import neurokit2 as nk
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pygame
import asyncio
import platform
from scipy import signal
import os
import sys
import random
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns


# Statistical Analysis Class and Function
class StatisticalAnalyzer:
    def __init__(self):
        self.results = {}

    def compute_descriptive_stats(self, data, label):
        """Compute descriptive statistics for APA reporting"""
        stats_dict = {
            'mean': np.mean(data),
            'std': np.std(data, ddof=1),
            'median': np.median(data),
            'min': np.min(data),
            'max': np.max(data),
            'n': len(data)
        }
        self.results[f'{label}_descriptives'] = stats_dict
        return stats_dict

    def independent_samples_ttest(self, group1, group2, label):
        """Perform independent samples t-test with effect size"""
        # Check for normality (Shapiro-Wilk test)
        _, p_norm1 = stats.shapiro(group1) if len(group1) < 5000 else (None, 0.05)
        _, p_norm2 = stats.shapiro(group2) if len(group2) < 5000 else (None, 0.05)

        # If data is normal, use t-test; otherwise, use Mann-Whitney U
        if p_norm1 > 0.05 and p_norm2 > 0.05:
            t_stat, p_value = ttest_ind(group1, group2)
            test_used = "Independent samples t-test"
        else:
            t_stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
            test_used = "Mann-Whitney U test"

        # Calculate Cohen's d (effect size)
        pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) +
                              (len(group2) - 1) * np.var(group2, ddof=1)) /
                             (len(group1) + len(group2) - 2))
        cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std

        # Effect size interpretation
        if abs(cohens_d) < 0.2:
            effect_size = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_size = "small"
        elif abs(cohens_d) < 0.8:
            effect_size = "medium"
        else:
            effect_size = "large"

        result = {
            'test_used': test_used,
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'effect_size': effect_size,
            'significant': p_value < 0.05,
            'group1_stats': self.compute_descriptive_stats(group1, f'{label}_group1'),
            'group2_stats': self.compute_descriptive_stats(group2, f'{label}_group2')
        }

        self.results[f'{label}_comparison'] = result
        return result

    def correlation_analysis(self, x, y, labels):
        """Perform correlation analysis"""
        # Pearson correlation
        r_pearson, p_pearson = stats.pearsonr(x, y)

        # Spearman correlation (non-parametric)
        r_spearman, p_spearman = stats.spearmanr(x, y)

        result = {
            'pearson_r': r_pearson,
            'pearson_p': p_pearson,
            'spearman_r': r_spearman,
            'spearman_p': p_spearman,
            'pearson_significant': p_pearson < 0.05,
            'spearman_significant': p_spearman < 0.05
        }

        self.results[f'correlation_{labels[0]}_{labels[1]}'] = result
        return result

    def generate_apa_report(self):
        """Generate APA-style statistical reporting"""
        print("\n" + "=" * 60)
        print("STATISTICAL ANALYSIS REPORT (APA FORMAT)")
        print("=" * 60)

        for key, result in self.results.items():
            if 'comparison' in key:
                self.report_comparison_apa(key, result)
            elif 'correlation' in key:
                self.report_correlation_apa(key, result)

    def report_comparison_apa(self, key, result):
        """Report comparison results in APA format"""
        print(f"\n{key.replace('_', ' ').title()}:")
        print("-" * 40)

        g1_stats = result['group1_stats']
        g2_stats = result['group2_stats']

        print(f"Group 1: M = {g1_stats['mean']:.3f}, SD = {g1_stats['std']:.3f}, n = {g1_stats['n']}")
        print(f"Group 2: M = {g2_stats['mean']:.3f}, SD = {g2_stats['std']:.3f}, n = {g2_stats['n']}")

        if 't-test' in result['test_used']:
            print(f"\n{result['test_used']}: t({g1_stats['n'] + g2_stats['n'] - 2}) = {result['t_statistic']:.3f}, "
                  f"p = {result['p_value']:.3f}")
        else:
            print(f"\n{result['test_used']}: U = {result['t_statistic']:.3f}, p = {result['p_value']:.3f}")

        print(f"Cohen's d = {result['cohens_d']:.3f} ({result['effect_size']} effect)")

        if result['significant']:
            print("Result: Statistically significant difference found.")
        else:
            print("Result: No statistically significant difference found.")

    def report_correlation_apa(self, key, result):
        """Report correlation results in APA format"""
        print(f"\n{key.replace('_', ' ').title()}:")
        print("-" * 40)

        print(f"Pearson r = {result['pearson_r']:.3f}, p = {result['pearson_p']:.3f}")
        print(f"Spearman ρ = {result['spearman_r']:.3f}, p = {result['spearman_p']:.3f}")

        if result['pearson_significant']:
            print("Pearson correlation: Statistically significant")
        if result['spearman_significant']:
            print("Spearman correlation: Statistically significant")


def analyze_training_results(vr_adaptive, vr_static, tracker):
    """Main analysis function to add to your existing code"""

    analyzer = StatisticalAnalyzer()

    # Extract data for analysis
    adaptive_decisions = np.array(vr_adaptive.decisions)
    static_decisions = np.array(vr_static.decisions)

    adaptive_times = np.array(vr_adaptive.task_times)
    static_times = np.array(vr_static.task_times)

    adaptive_pre_rmssd = np.array([x for x in vr_adaptive.pre_recovery_rmssd if isinstance(x, (int, float))])
    static_pre_rmssd = np.array([x for x in vr_static.pre_recovery_rmssd if isinstance(x, (int, float))])

    adaptive_post_rmssd = np.array([x for x in vr_adaptive.post_recovery_rmssd if isinstance(x, (int, float))])
    static_post_rmssd = np.array([x for x in vr_static.post_recovery_rmssd if isinstance(x, (int, float))])

    print("\n" + "=" * 60)
    print("COMPREHENSIVE STATISTICAL ANALYSIS")
    print("=" * 60)

    # 1. Decision Accuracy Comparison
    print("\n1. DECISION ACCURACY ANALYSIS")
    decision_result = analyzer.independent_samples_ttest(
        adaptive_decisions, static_decisions, "decision_accuracy"
    )

    # 2. Task Completion Time Comparison
    print("\n2. TASK COMPLETION TIME ANALYSIS")
    time_result = analyzer.independent_samples_ttest(
        adaptive_times, static_times, "task_completion_time"
    )

    # 3. Physiological Recovery Analysis
    if len(adaptive_pre_rmssd) > 0 and len(static_pre_rmssd) > 0:
        print("\n3. PRE-TASK RMSSD ANALYSIS")
        pre_rmssd_result = analyzer.independent_samples_ttest(
            adaptive_pre_rmssd, static_pre_rmssd, "pre_task_rmssd"
        )

    if len(adaptive_post_rmssd) > 0 and len(static_post_rmssd) > 0:
        print("\n4. POST-TASK RMSSD ANALYSIS")
        post_rmssd_result = analyzer.independent_samples_ttest(
            adaptive_post_rmssd, static_post_rmssd, "post_task_rmssd"
        )

    # 5. Within-group recovery analysis
    if len(adaptive_pre_rmssd) > 0 and len(adaptive_post_rmssd) > 0:
        print("\n5. ADAPTIVE MODE RECOVERY ANALYSIS")
        adaptive_recovery_result = analyzer.independent_samples_ttest(
            adaptive_pre_rmssd, adaptive_post_rmssd, "adaptive_recovery"
        )

    if len(static_pre_rmssd) > 0 and len(static_post_rmssd) > 0:
        print("\n6. STATIC MODE RECOVERY ANALYSIS")
        static_recovery_result = analyzer.independent_samples_ttest(
            static_pre_rmssd, static_post_rmssd, "static_recovery"
        )

    # 6. Correlation analyses
    if len(adaptive_decisions) > 0 and len(adaptive_times) > 0:
        print("\n7. CORRELATION: DECISION ACCURACY vs TASK TIME (ADAPTIVE)")
        corr_result1 = analyzer.correlation_analysis(
            adaptive_decisions, adaptive_times, ["accuracy", "time"]
        )

    if len(static_decisions) > 0 and len(static_times) > 0:
        print("\n8. CORRELATION: DECISION ACCURACY vs TASK TIME (STATIC)")
        corr_result2 = analyzer.correlation_analysis(
            static_decisions, static_times, ["accuracy", "time"]
        )

    # Generate comprehensive APA report
    analyzer.generate_apa_report()

    # Calculate additional metrics for reporting
    print("\n" + "=" * 60)
    print("ADDITIONAL METRICS FOR APA REPORTING")
    print("=" * 60)

    # Confidence intervals (95%)
    from scipy.stats import t

    def calculate_ci(data, confidence=0.95):
        n = len(data)
        mean = np.mean(data)
        sem = stats.sem(data)
        h = sem * t.ppf((1 + confidence) / 2., n - 1)
        return mean - h, mean + h

    # Decision accuracy CIs
    adaptive_acc_ci = calculate_ci(adaptive_decisions)
    static_acc_ci = calculate_ci(static_decisions)

    print(f"\nDecision Accuracy 95% Confidence Intervals:")
    print(f"Adaptive: [{adaptive_acc_ci[0]:.3f}, {adaptive_acc_ci[1]:.3f}]")
    print(f"Static: [{static_acc_ci[0]:.3f}, {static_acc_ci[1]:.3f}]")

    # Task time CIs
    adaptive_time_ci = calculate_ci(adaptive_times)
    static_time_ci = calculate_ci(static_times)

    print(f"\nTask Completion Time 95% Confidence Intervals:")
    print(f"Adaptive: [{adaptive_time_ci[0]:.3f}, {adaptive_time_ci[1]:.3f}] seconds")
    print(f"Static: [{static_time_ci[0]:.3f}, {static_time_ci[1]:.3f}] seconds")

    return analyzer.results


# Function to load a random EEG CSV file from the preprocessed directory
def load_random_eeg_csv(folder_path):
    """Load a random EEG CSV file from the given directory"""
    # Use simulated data if no directory provided or directory doesn't exist
    if not folder_path or not os.path.exists(folder_path):
        print("❌ EEG directory not found. Using simulated EEG patterns.")
        return None, "simulated"
    
    eeg_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    if not eeg_files:
        print(f"❌ No EEG CSV files found in {folder_path}. Using simulated EEG patterns.")
        return None, "simulated"
    
    selected_file = random.choice(eeg_files)
    full_path = os.path.join(folder_path, selected_file)
    subject_id = selected_file.split('_')[0]  # Extract subject ID (e.g., sub-001)
    return full_path, subject_id


class MarkovChain:
    def __init__(self, states, transition_matrix):
        self.states = states
        self.transition_matrix = np.array(transition_matrix)
        self.current_state = np.random.choice(states)

    def next_state(self):
        probs = self.transition_matrix[self.states.index(self.current_state)]
        self.current_state = np.random.choice(self.states, p=probs)
        return self.current_state


cognitive_load_mc = MarkovChain(
    states=['low', 'medium', 'high'],
    transition_matrix=[[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.1, 0.4, 0.5]]
)
individual_mc = MarkovChain(
    states=['resilient', 'average', 'stress_prone'],
    transition_matrix=[[0.8, 0.15, 0.05], [0.1, 0.8, 0.1], [0.05, 0.15, 0.8]]
)


class LearningTracker:
    def __init__(self):
        self.performance_history_adaptive = []
        self.performance_history_static = []
        self.retention_scores_adaptive = []
        self.retention_scores_static = []
        self.cognitive_failures_adaptive = []
        self.cognitive_failures_static = []

    def track_skill_acquisition(self, decision_accuracy, is_adaptive=True):
        if is_adaptive:
            self.performance_history_adaptive.append(decision_accuracy)
        else:
            self.performance_history_static.append(decision_accuracy)
        if len(self.performance_history_adaptive) > 1 and len(self.performance_history_static) > 1:
            # Avoid division by zero
            if self.performance_history_adaptive[0] != 0:
                adaptive_improvement = (self.performance_history_adaptive[-1] - self.performance_history_adaptive[0]) / \
                                       self.performance_history_adaptive[0] * 100
            else:
                adaptive_improvement = 0
            if self.performance_history_static[0] != 0:
                static_improvement = (self.performance_history_static[-1] - self.performance_history_static[0]) / \
                                     self.performance_history_static[0] * 100
            else:
                static_improvement = 0
            return adaptive_improvement, static_improvement
        return None, None

    def retention_assessment(self, decision_accuracy, scenario_id, is_adaptive=True):
        scenario_performance = (scenario_id, decision_accuracy)
        if is_adaptive:
            self.retention_scores_adaptive.append(scenario_performance)
        else:
            self.retention_scores_static.append(scenario_performance)
        adaptive_scores = {sid: acc for sid, acc in self.retention_scores_adaptive}
        static_scores = {sid: acc for sid, acc in self.retention_scores_static}
        adaptive_retention = []
        static_retention = []
        for sid, acc in adaptive_scores.items():
            previous_attempts = [score for s_id, score in self.retention_scores_adaptive if s_id == sid]
            if len(previous_attempts) > 1:
                if previous_attempts[0] != 0:
                    improvement = (previous_attempts[-1] - previous_attempts[0]) / previous_attempts[0] * 100
                else:
                    improvement = 0
                adaptive_retention.append(improvement)
        for sid, acc in static_scores.items():
            previous_attempts = [score for s_id, score in self.retention_scores_static if s_id == sid]
            if len(previous_attempts) > 1:
                if previous_attempts[0] != 0:
                    improvement = (previous_attempts[-1] - previous_attempts[0]) / previous_attempts[0] * 100
                else:
                    improvement = 0
                static_retention.append(improvement)
        return np.mean(adaptive_retention) if adaptive_retention else None, np.mean(
            static_retention) if static_retention else None

    def detect_cognitive_failures(self, decision_accuracy, cognitive_load, stress, is_adaptive=True):
        if (cognitive_load > 0.1 or stress > 0.1) and decision_accuracy < 0.5:
            failure = True
        else:
            failure = False
        if is_adaptive:
            self.cognitive_failures_adaptive.append(failure)
        else:
            self.cognitive_failures_static.append(failure)
        adaptive_failure_rate = np.mean(self.cognitive_failures_adaptive) if self.cognitive_failures_adaptive else 0
        static_failure_rate = np.mean(self.cognitive_failures_static) if self.cognitive_failures_static else 0
        return adaptive_failure_rate * 100, static_failure_rate * 100


def load_real_eeg_features(file_path, duration=120, fs=256):
    try:
        df = pd.read_csv(file_path)
        eeg_data = df[['T7', 'F8', 'Cz', 'P4']].values.T
        ch_names = ['T7', 'F8', 'Cz', 'P4']
        sfreq = 200
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(eeg_data, info)
        raw.filter(1, 40, fir_design='firwin')
        raw.resample(fs)
        data, times = raw.get_data(return_times=True)
        data = data[:, :int(duration * fs)]
        times = times[:int(duration * fs)]
        psd, freqs = mne.time_frequency.psd_array_welch(data, sfreq=fs, fmin=1, fmax=40, n_fft=2048)
        delta_idx = (freqs >= 1) & (freqs <= 4)
        theta_idx = (freqs >= 4) & (freqs <= 7)
        alpha_idx = (freqs >= 8) & (freqs <= 12)
        beta_idx = (freqs >= 13) & (freqs <= 30)
        gamma_idx = (freqs >= 30) & (freqs <= 40)
        delta_power = np.mean(np.mean(psd[:, delta_idx], axis=1))
        theta_power = np.mean(np.mean(psd[:, theta_idx], axis=1))
        alpha_power = np.mean(np.mean(psd[:, alpha_idx], axis=1))
        beta_power = np.mean(np.mean(psd[:, beta_idx], axis=1))
        gamma_power = np.mean(np.mean(psd[:, gamma_idx], axis=1))
        tbr = np.log1p(theta_power / np.clip(beta_power, 1e-6, None))
        alpha_theta_ratio = np.log1p(alpha_power / np.clip(theta_power, 1e-6, None))
        gamma_theta_ratio = np.log1p(gamma_power / np.clip(theta_power, 1e-6, None))
        f8_alpha = np.mean(psd[1, alpha_idx])
        t7_alpha = np.mean(psd[0, alpha_idx])
        frontal_asymmetry = np.log(f8_alpha) - np.log(t7_alpha)
        cz_data = data[2, :]
        connectivity_measure = 0
        for ch_idx in [0, 1, 3]:
            coherence = np.corrcoef(cz_data, data[ch_idx, :])[0, 1]
            connectivity_measure += coherence
        connectivity_measure /= 3

        def spectral_entropy(psd_channel):
            psd_norm = psd_channel / np.sum(psd_channel)
            psd_norm = psd_norm[psd_norm > 0]
            return -np.sum(psd_norm * np.log(psd_norm))

        avg_spectral_entropy = np.mean([spectral_entropy(psd[i, :]) for i in range(4)])
        feature_names = [
            'delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power',
            'tbr', 'alpha_theta_ratio', 'gamma_theta_ratio',
            'frontal_asymmetry', 'connectivity', 'spectral_entropy'
        ]
        features = []
        base_values = [delta_power, theta_power, alpha_power, beta_power, gamma_power,
                       tbr, alpha_theta_ratio, gamma_theta_ratio,
                       frontal_asymmetry, connectivity_measure, avg_spectral_entropy]
        for i, base_val in enumerate(base_values):
            temporal_variation = np.random.normal(0, abs(base_val) * 0.1, int(duration * fs))
            feature_series = base_val + temporal_variation
            features.append(feature_series)
        return np.stack(features, axis=1), feature_names
    except Exception as e:
        print(f"Error loading real EEG data: {e}. Using simulated EEG features.")
        return simulate_eeg_features(duration, fs)


def simulate_eeg_features(duration=120, fs=256):
    t = np.linspace(0, duration, int(duration * fs))
    theta_power = 5.0 + 1.0 * np.sin(2 * np.pi * 0.1 * t) + np.random.normal(0, 0.3, len(t))
    alpha_power = 3.0 + 0.4 * np.cos(2 * np.pi * 0.08 * t) + np.random.normal(0, 0.15, len(t))
    tbr = np.log1p(theta_power / np.clip(alpha_power, 1e-6, None))
    delta_power = 2.0 + 0.3 * np.sin(2 * np.pi * 0.05 * t) + np.random.normal(0, 0.1, len(t))
    beta_power = 1.2 + 0.3 * np.sin(2 * np.pi * 0.15 * t) + np.random.normal(0, 0.1, len(t))
    gamma_power = 0.8 + 0.2 * np.sin(2 * np.pi * 0.2 * t) + np.random.normal(0, 0.05, len(t))
    alpha_theta_ratio = np.log1p(alpha_power / np.clip(theta_power, 1e-6, None))
    gamma_theta_ratio = np.log1p(gamma_power / np.clip(theta_power, 1e-6, None))
    frontal_asymmetry = -0.15 + 0.05 * np.sin(2 * np.pi * 0.03 * t) + np.random.normal(0, 0.02, len(t))
    connectivity = 0.3 + 0.1 * np.sin(2 * np.pi * 0.02 * t) + np.random.normal(0, 0.05, len(t))
    spectral_entropy = 2.8 + 0.2 * np.sin(2 * np.pi * 0.04 * t) + np.random.normal(0, 0.1, len(t))
    feature_names = [
        'delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power',
        'tbr', 'alpha_theta_ratio', 'gamma_theta_ratio',
        'frontal_asymmetry', 'connectivity', 'spectral_entropy'
    ]
    return np.stack([
        delta_power, theta_power, alpha_power, beta_power, gamma_power,
        tbr, alpha_theta_ratio, gamma_theta_ratio,
        frontal_asymmetry, connectivity, spectral_entropy
    ], axis=1), feature_names


def fallback_simulated_features(duration=120, fs=128):
    t = np.linspace(0, duration, int(duration * fs))
    features = []
    unnormalized_rmssd = []
    cognitive_states = []
    individual_states = []
    eeg_features, eeg_feature_names = simulate_eeg_features(duration, fs)
    for _ in range(len(t)):
        cog_state = cognitive_load_mc.next_state()
        ind_state = individual_mc.next_state()
        cognitive_states.append(cog_state)
        individual_states.append(ind_state)
        if ind_state == 'resilient':
            rmssd = 40 + np.random.normal(0, 2)
            gsr_base = 0.8 + np.random.normal(0, 0.1)
            hr = 70 + np.random.normal(0, 2)
            temp = 36.5 + np.random.normal(0, 0.5)
            acc = 1.0 + np.random.normal(0, 0.1)
        elif ind_state == 'average':
            rmssd = 30 + np.random.normal(0, 2)
            gsr_base = 1.0 + np.random.normal(0, 0.1)
            hr = 80 + np.random.normal(0, 2)
            temp = 37.0 + np.random.normal(0, 0.5)
            acc = 1.5 + np.random.normal(0, 0.1)
        else:
            rmssd = 20 + np.random.normal(0, 2)
            gsr_base = 1.3 + np.random.normal(0, 0.15)
            hr = 90 + np.random.normal(0, 2)
            temp = 37.5 + np.random.normal(0, 0.5)
            acc = 2.0 + np.random.normal(0, 0.1)
        gsr = gsr_base + (2.0 if np.random.rand() > 0.95 else 0.0)
        feature_row = list(eeg_features[_, :]) + [rmssd, gsr, hr, temp, acc]
        features.append(feature_row)
        unnormalized_rmssd.append(rmssd)
    features = np.array(features)
    unnormalized_rmssd = np.array(unnormalized_rmssd)
    feature_names = eeg_feature_names + ['rmssd', 'gsr', 'hr', 'temp', 'acc']
    return features, unnormalized_rmssd, feature_names, cognitive_states, individual_states


def load_wearable_features(wearable_dir=None, eeg_file_path=None, duration=120, fs=128):
    """
    Load wearable features - uses simulated data if directories not available
    """
    try:
        # Always try to load EEG features first
        eeg_features, eeg_feature_names = load_real_eeg_features(eeg_file_path, duration=duration, fs=fs) if eeg_file_path else simulate_eeg_features(duration, fs)
        
        # Use simulated wearable data if directory not provided or doesn't exist
        if not wearable_dir or not os.path.exists(wearable_dir):
            print("Wearable data directory not found. Using simulated physiological data.")
            return fallback_simulated_features(duration, fs)
        
        # Try to load real wearable data (original code logic)
        ibi_path = os.path.join(wearable_dir, 'IBI.csv')
        if not os.path.exists(ibi_path):
            print("IBI.csv not found. Using simulated physiological data.")
            return fallback_simulated_features(duration, fs)
            
        ibi_df = pd.read_csv(ibi_path, names=['timestamp', 'ibi'], skiprows=1)
        print("IBI Data Sample:\n", ibi_df.head())
        ibi_df['timestamp'] = pd.to_numeric(ibi_df['timestamp'], errors='coerce')
        ibi_df['ibi'] = pd.to_numeric(ibi_df['ibi'], errors='coerce')
        ibi_df.dropna(inplace=True)
        if ibi_df.empty:
            raise ValueError("IBI.csv contains no valid numeric data after parsing.")
        ibi_times = ibi_df['timestamp'].values
        ibi_values = ibi_df['ibi'].values
        rpeaks = np.cumsum(ibi_values) * fs
        rpeaks = rpeaks[rpeaks < duration * fs]
        ecg_signals = nk.ecg_simulate(duration=duration, sampling_rate=fs, rpeaks=rpeaks.astype(int))
        ecg_processed, info = nk.ecg_process(ecg_signals, sampling_rate=fs)
        hrv = nk.hrv_time(info, sampling_rate=fs)
        rmssd = hrv['HRV_RMSSD'].iloc[0] if not hrv.empty else 30.0
        
        # Load other wearable data files with error handling
        eda_path = os.path.join(wearable_dir, 'EDA.csv')
        hr_path = os.path.join(wearable_dir, 'HR.csv')
        temp_path = os.path.join(wearable_dir, 'TEMP.csv')
        acc_path = os.path.join(wearable_dir, 'ACC.csv')
        
        # Check if all required files exist
        required_files = [eda_path, hr_path, temp_path, acc_path]
        for file_path in required_files:
            if not os.path.exists(file_path):
                print(f"Required file {file_path} not found. Using simulated data.")
                return fallback_simulated_features(duration, fs)
        
        # Load EDA data
        with open(eda_path, 'r') as f:
            lines = f.readlines()
            eda_start_time = float(lines[0].strip())
            eda_fs = float(lines[1].strip())
        eda_df = pd.read_csv(eda_path, skiprows=2, names=['eda'])
        eda_df['eda'] = pd.to_numeric(eda_df['eda'], errors='coerce')
        eda_df.dropna(inplace=True)
        if eda_df.empty:
            raise ValueError("EDA.csv contains no valid numeric data after parsing.")
        eda_interp = signal.resample(eda_df['eda'].values, int(duration * fs))
        gsr_clean = nk.signal_sanitize(eda_interp)
        gsr_mean = np.mean(gsr_clean)
        
        # Load HR data
        with open(hr_path, 'r') as f:
            lines = f.readlines()
            hr_start_time = float(lines[0].strip())
            hr_fs = float(lines[1].strip())
        hr_df = pd.read_csv(hr_path, skiprows=2, names=['heart_rate'])
        hr_df['heart_rate'] = pd.to_numeric(hr_df['heart_rate'], errors='coerce')
        hr_df.dropna(inplace=True)
        if hr_df.empty:
            raise ValueError("HR.csv contains no valid numeric data after parsing.")
        hr_interp = signal.resample(hr_df['heart_rate'].values, int(duration * fs))
        hr_mean = np.mean(hr_interp)
        
        # Load TEMP data
        with open(temp_path, 'r') as f:
            lines = f.readlines()
            temp_start_time = float(lines[0].strip())
            temp_fs = float(lines[1].strip())
        temp_df = pd.read_csv(temp_path, skiprows=2, names=['temp'])
        temp_df['temp'] = pd.to_numeric(temp_df['temp'], errors='coerce')
        temp_df.dropna(inplace=True)
        if temp_df.empty:
            raise ValueError("TEMP.csv contains no valid numeric data after parsing.")
        temp_interp = signal.resample(temp_df['temp'].values, int(duration * fs))
        temp_mean = np.mean(temp_interp)
        
        # Load ACC data
        with open(acc_path, 'r') as f:
            lines = f.readlines()
            acc_start_time = float(lines[0].strip().split(',')[0])
            acc_fs = float(lines[1].strip().split(',')[0])
        acc_df = pd.read_csv(acc_path, skiprows=2, names=['x', 'y', 'z'])
        acc_df[['x', 'y', 'z']] = acc_df[['x', 'y', 'z']].apply(pd.to_numeric, errors='coerce')
        acc_df.dropna(inplace=True)
        if acc_df.empty:
            raise ValueError("ACC.csv contains no valid numeric data after parsing.")
        acc_x = signal.resample(acc_df['x'].values, int(duration * fs))
        acc_y = signal.resample(acc_df['y'].values, int(duration * fs))
        acc_z = signal.resample(acc_df['z'].values, int(duration * fs))
        acc_magnitude = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
        acc_mean = np.mean(acc_magnitude)
        
        # Generate time series data
        rmssd_series = rmssd + np.random.normal(0, rmssd * 0.1, int(duration * fs))
        gsr_series = gsr_mean + np.random.normal(0, gsr_mean * 0.1, int(duration * fs))
        hr_series = hr_mean + np.random.normal(0, hr_mean * 0.1, int(duration * fs))
        temp_series = temp_mean + np.random.normal(0, temp_mean * 0.05, int(duration * fs))
        acc_series = acc_mean + np.random.normal(0, acc_mean * 0.1, int(duration * fs))
        
        samples = int(duration * fs)
        features = []
        unnormalized_rmssd = []
        cognitive_states = []
        individual_states = []
        
        for _ in range(samples):
            cog_state = cognitive_load_mc.next_state()
            ind_state = individual_mc.next_state()
            cognitive_states.append(cog_state)
            individual_states.append(ind_state)
            
            if ind_state == 'resilient':
                rmssd_adj = rmssd_series[_] * 1.2 + np.random.normal(0, 2)
                gsr_adj = gsr_series[_] * 0.8 + np.random.normal(0, 0.1)
                hr_adj = hr_series[_] * 0.9 + np.random.normal(0, 2)
                temp_adj = temp_series[_] * 0.95 + np.random.normal(0, 0.5)
                acc_adj = acc_series[_] * 0.9 + np.random.normal(0, 0.1)
            elif ind_state == 'average':
                rmssd_adj = rmssd_series[_] * 1.0 + np.random.normal(0, 2)
                gsr_adj = gsr_series[_] * 1.0 + np.random.normal(0, 0.1)
                hr_adj = hr_series[_] * 1.0 + np.random.normal(0, 2)
                temp_adj = temp_series[_] * 1.0 + np.random.normal(0, 0.5)
                acc_adj = acc_series[_] * 1.0 + np.random.normal(0, 0.1)
            else:
                rmssd_adj = rmssd_series[_] * 0.8 + np.random.normal(0, 2)
                gsr_adj = gsr_series[_] * 1.2 + np.random.normal(0, 0.15)
                hr_adj = hr_series[_] * 1.1 + np.random.normal(0, 2)
                temp_adj = temp_series[_] * 1.05 + np.random.normal(0, 0.5)
                acc_adj = acc_series[_] * 1.1 + np.random.normal(0, 0.1)
                
            gsr_adj += 2.0 if np.random.rand() > 0.95 else 0.0
            feature_row = list(eeg_features[_, :]) + [rmssd_adj, gsr_adj, hr_adj, temp_adj, acc_adj]
            features.append(feature_row)
            unnormalized_rmssd.append(rmssd_series[_])
            
        features = np.array(features)
        unnormalized_rmssd = np.array(unnormalized_rmssd)
        feature_names = eeg_feature_names + ['rmssd', 'gsr', 'hr', 'temp', 'acc']
        return features, unnormalized_rmssd, feature_names, cognitive_states, individual_states
        
    except Exception as e:
        print(f"Error loading wearable data: {e}. Using fully simulated data.")
        return fallback_simulated_features(duration, fs)


def generate_military_labels(features):
    tbr = features[:, 5]
    rmssd = features[:, 11]
    gsr = features[:, 12]
    hr = features[:, 13]
    temp = features[:, 14]
    acc = features[:, 15]
    cognitive_load = 1 / (1 + np.exp(-(tbr - 0.5)))
    stress = 1 / (1 + np.exp(-(-rmssd / 30 + gsr - 1.0 + hr / 80 + temp / 37 + acc / 1.5)))
    return np.stack([cognitive_load, stress], axis=1)


def prepare_training_data(features, unnormalized_rmssd, labels, window_size=10 * 128):
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    X, y, unnormalized_rmssd_windows = [], [], []
    step_size = window_size // 10  # Changed from // 3 to // 10 to increase iterations
    for i in range(0, len(features) - window_size, step_size):
        X.append(features[i:i + window_size])
        y.append(labels[i + window_size - 1])
        unnormalized_rmssd_windows.append(unnormalized_rmssd[i + window_size - 1])
    return np.array(X), np.array(y), np.array(unnormalized_rmssd_windows), scaler


class AdaptiveMilitarySystem:
    def __init__(self):
        self.actions = [
            'deploy_air_support', 'hold_position',
            'reduce_time_pressure', 'increase_time_pressure',
            'reduce_comms_noise', 'increase_comms_noise'
        ]
        self.model = RandomForestClassifier(n_estimators=100)
        self.scaler = None

    def train(self, X, y):
        X_flat = X.reshape(X.shape[0], -1)
        self.model.fit(X_flat, (y > 0.5).astype(int))

    def select_action(self, state, is_adaptive=True):
        if not is_adaptive:
            return np.random.choice(['deploy_air_support', 'hold_position'])
        cognitive_load, stress = state
        if cognitive_load > 0.1:
            return 'hold_position'
        elif stress > 0.1:
            return 'reduce_time_pressure'
        elif cognitive_load < -0.1:
            return 'deploy_air_support'
        elif stress < -0.1:
            return 'increase_time_pressure'
        return np.random.choice(['reduce_comms_noise', 'increase_comms_noise'])


class MilitaryVRInterface:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Military Decision-Making IoT Training")
        self.font = pygame.font.SysFont('arial', 20)
        self.current_tactic = "hold_position"
        self.time_pressure = 0.5
        self.comms_noise_level = 0.5
        self.clock = pygame.time.Clock()
        self.running = True
        self.decisions = []
        self.task_times = []
        self.pre_recovery_rmssd = []
        self.post_recovery_rmssd = []
        self.iteration = 0
        self.max_iterations = 359
        self.scenario_ids = []

    def update_environment(self, action, state, decision_accuracy, task_time, rmssd, cog_state, ind_state, scenario_id):
        cognitive_load, stress = state
        self.iteration += 1
        self.decisions.append(decision_accuracy)
        self.task_times.append(task_time)
        self.scenario_ids.append(scenario_id)

        if action == 'deploy_air_support':
            self.current_tactic = "Deploy Air Support"
        elif action == 'hold_position':
            self.current_tactic = "Hold Position"
        elif action == 'reduce_time_pressure':
            self.time_pressure = max(0.1, self.time_pressure - 0.1)
        elif action == 'increase_time_pressure':
            self.time_pressure = min(1.0, self.time_pressure + 0.1)
        elif action == 'reduce_comms_noise':
            self.comms_noise_level = max(0.1, self.comms_noise_level - 0.2)
        elif action == 'increase_comms_noise':
            self.comms_noise_level = min(1.0, self.comms_noise_level + 0.2)

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.running = False
                return False

        self.screen.fill((255, 255, 255))
        progress = min(100, int((self.iteration / self.max_iterations) * 100))
        texts = [
            f'Current Tactic: {self.current_tactic}',
            f'Time Pressure: {self.time_pressure:.2f}',
            f'Comms Noise Level: {self.comms_noise_level:.2f}',
            f'Cognitive Load: {cognitive_load:.2f} ({cog_state})',
            f'Stress: {stress:.2f} ({ind_state})',
            f'Decision Accuracy: {decision_accuracy:.2f}',
            f'Task Time: {task_time:.2f}s',
            f'RMSSD (Pre-Recovery): {rmssd:.2f}',
            f'Scenario ID: {scenario_id}',
            f'Progress: {progress}% (Iteration {self.iteration}/{self.max_iterations})',
            'Press ESC to quit'
        ]
        for i, text in enumerate(texts):
            self.screen.blit(self.font.render(text, True, (0, 0, 0)), (50, 50 + i * 30))

        pygame.display.flip()
        self.clock.tick(30)
        return True

    def recovery_game(self, unnormalized_rmssd):
        self.pre_recovery_rmssd.append(unnormalized_rmssd)
        score = 0
        start_time = pygame.time.get_ticks()
        running = True
        pre_rmssd = unnormalized_rmssd
        
        while running and (pygame.time.get_ticks() - start_time) < 30000:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    score += 1
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
            self.screen.fill((200, 200, 200))
            self.screen.blit(self.font.render(f'Target Game Score: {score}', True, (0, 0, 0)), (50, 50))
            pygame.display.flip()
            self.clock.tick(30)
            
        post_rmssd = pre_rmssd * 1.1
        self.post_recovery_rmssd.append(post_rmssd)
        return score, pre_rmssd, post_rmssd

    def quit(self):
        pygame.quit()


async def main():
    print("=== Adaptive Military Decision-Making Training with IoT Feedback ===")

    # Generic directory paths - will use simulated data if not available
    eeg_dir = "data/eeg"  # Users can create this directory and add EEG CSV files
    wearable_dir = "data/wearable"  # Users can create this directory and add wearable sensor data

    # Load a random EEG file for Adaptive mode
    eeg_path_adaptive, subject_id_adaptive = load_random_eeg_csv(eeg_dir)
    if eeg_path_adaptive:
        print(f"🧠 Adaptive Mode - Using EEG from subject: {subject_id_adaptive} at {eeg_path_adaptive}")
    else:
        print(f"🧠 Adaptive Mode - Using simulated EEG patterns")

    print("Running Adaptive Mode...")
    features, unnormalized_rmssd, feature_names, cog_states, ind_states = load_wearable_features(
        wearable_dir=wearable_dir, eeg_file_path=eeg_path_adaptive)
    labels = generate_military_labels(features)
    X, y, unnormalized_rmssd_windows, scaler = prepare_training_data(features, unnormalized_rmssd, labels)

    adaptive_system = AdaptiveMilitarySystem()
    adaptive_system.train(X, y)
    adaptive_system.scaler = scaler

    tracker = LearningTracker()
    vr_adaptive = MilitaryVRInterface()

    print("Starting real-time training simulation (Adaptive Mode)...")
    window_size = 10 * 128
    window_indices = range(len(unnormalized_rmssd_windows))
    scenario_cycle = list(range(5)) * (len(window_indices) // 5 + 1)
    scenario_cycle = scenario_cycle[:len(window_indices)]

    for idx in window_indices:
        i = idx * 128
        if not vr_adaptive.running:
            break

        window = features[i:i + window_size]
        window_scaled = scaler.transform(window)
        state = np.mean(window_scaled, axis=0)[:2]
        action = adaptive_system.select_action(state, is_adaptive=True)

        decision_accuracy = np.random.uniform(0.6, 1.0) if state[0] < 0.1 else np.random.uniform(0.3, 0.7)
        task_time = np.random.uniform(2, 5) if state[1] < 0.1 else np.random.uniform(5, 8)
        rmssd = unnormalized_rmssd_windows[idx]

        cog_state = cog_states[i + window_size - 1]
        ind_state = ind_states[i + window_size - 1]
        scenario_id = scenario_cycle[idx]

        print(
            f"State: Cognitive Load={state[0]:.2f} ({cog_state}), Stress={state[1]:.2f} ({ind_state}), Action={action}")

        if not vr_adaptive.update_environment(action, state, decision_accuracy, task_time, rmssd, cog_state, ind_state,
                                              scenario_id):
            break

        adaptive_improvement, _ = tracker.track_skill_acquisition(decision_accuracy, is_adaptive=True)
        adaptive_retention, _ = tracker.retention_assessment(decision_accuracy, scenario_id, is_adaptive=True)
        adaptive_failure_rate, _ = tracker.detect_cognitive_failures(decision_accuracy, state[0], state[1],
                                                                     is_adaptive=True)

        if idx % 10 == 0 and idx > 0:
            print("Starting recovery game...")
            score, pre_rmssd, post_rmssd = vr_adaptive.recovery_game(rmssd)
            print(f"Recovery Game Score: {score}")
            print(f"Pre-Recovery RMSSD: {pre_rmssd:.2f}, Post-Recovery RMSSD: {post_rmssd:.2f}")

        await asyncio.sleep(0.5)

    avg_accuracy_adaptive = np.mean(vr_adaptive.decisions) if vr_adaptive.decisions else 0
    avg_task_time_adaptive = np.mean(vr_adaptive.task_times) if vr_adaptive.task_times else 0
    avg_pre_rmssd_adaptive = np.mean(
        [rmssd for rmssd in vr_adaptive.pre_recovery_rmssd if isinstance(rmssd, (int, float))])
    avg_post_rmssd_adaptive = np.mean(
        [rmssd for rmssd in vr_adaptive.post_recovery_rmssd if isinstance(rmssd, (int, float))])

    print("\nRunning Static Mode...")
    # Load a random EEG file for Static mode (can be different from Adaptive mode)
    eeg_path_static, subject_id_static = load_random_eeg_csv(eeg_dir)
    if eeg_path_static:
        print(f"🧠 Static Mode - Using EEG from subject: {subject_id_static} at {eeg_path_static}")
    else:
        print(f"🧠 Static Mode - Using simulated EEG patterns")
        
    features, unnormalized_rmssd, feature_names, cog_states, ind_states = load_wearable_features(
        wearable_dir=wearable_dir, eeg_file_path=eeg_path_static)
    labels = generate_military_labels(features)
    X, y, unnormalized_rmssd_windows, scaler = prepare_training_data(features, unnormalized_rmssd, labels)

    adaptive_system = AdaptiveMilitarySystem()
    adaptive_system.train(X, y)
    adaptive_system.scaler = scaler

    vr_static = MilitaryVRInterface()

    print("Starting real-time training simulation (Static Mode)...")
    for idx in window_indices:
        i = idx * 128
        if not vr_static.running:
            break

        window = features[i:i + window_size]
        window_scaled = scaler.transform(window)
        state = np.mean(window_scaled, axis=0)[:2]
        action = adaptive_system.select_action(state, is_adaptive=False)

        decision_accuracy = np.random.uniform(0.6, 1.0) if state[0] < 0.1 else np.random.uniform(0.3, 0.7)
        task_time = np.random.uniform(2, 5) if state[1] < 0.1 else np.random.uniform(5, 8)
        rmssd = unnormalized_rmssd_windows[idx]

        cog_state = cog_states[i + window_size - 1]
        ind_state = ind_states[i + window_size - 1]
        scenario_id = scenario_cycle[idx]

        print(
            f"State: Cognitive Load={state[0]:.2f} ({cog_state}), Stress={state[1]:.2f} ({ind_state}), Action={action}")

        if not vr_static.update_environment(action, state, decision_accuracy, task_time, rmssd, cog_state, ind_state,
                                            scenario_id):
            break

        _, static_improvement = tracker.track_skill_acquisition(decision_accuracy, is_adaptive=False)
        _, static_retention = tracker.retention_assessment(decision_accuracy, scenario_id, is_adaptive=False)
        _, static_failure_rate = tracker.detect_cognitive_failures(decision_accuracy, state[0], state[1],
                                                                   is_adaptive=False)

        if idx % 10 == 0 and idx > 0:
            print("Starting recovery game...")
            score, pre_rmssd, post_rmssd = vr_static.recovery_game(rmssd)
            print(f"Recovery Game Score: {score}")
            print(f"Pre-Recovery RMSSD: {pre_rmssd:.2f}, Post-Recovery RMSSD: {post_rmssd:.2f}")

        await asyncio.sleep(0.5)

    avg_accuracy_static = np.mean(vr_static.decisions) if vr_static.decisions else 0
    avg_task_time_static = np.mean(vr_static.task_times) if vr_static.task_times else 0
    avg_pre_rmssd_static = np.mean(
        [rmssd for rmssd in vr_static.pre_recovery_rmssd if isinstance(rmssd, (int, float))])
    avg_post_rmssd_static = np.mean(
        [rmssd for rmssd in vr_static.post_recovery_rmssd if isinstance(rmssd, (int, float))])

    # Run comprehensive statistical analysis
    print("\n" + "=" * 60)
    print("RUNNING COMPREHENSIVE STATISTICAL ANALYSIS...")
    print("=" * 60)

    statistical_results = analyze_training_results(vr_adaptive, vr_static, tracker)

    print("\nFinal Performance Metrics:")
    print(f"Adaptive Mode - Average Decision Accuracy: {avg_accuracy_adaptive:.2f}")
    print(f"Static Mode - Average Decision Accuracy: {avg_accuracy_static:.2f}")
    print(f"Adaptive Mode - Average Task Completion Time: {avg_task_time_adaptive:.2f}s")
    print(f"Static Mode - Average Task Completion Time: {avg_task_time_static:.2f}s")
    print(
        f"Adaptive Mode - Post-Task Physiological Recovery (RMSSD): Pre={avg_pre_rmssd_adaptive:.2f}, Post={avg_post_rmssd_adaptive:.2f}")
    print(
        f"Static Mode - Post-Task Physiological Recovery (RMSSD): Pre={avg_pre_rmssd_static:.2f}, Post={avg_post_rmssd_static:.2f}")
    print(f"Learning Improvement - Adaptive: {adaptive_improvement if adaptive_improvement is not None else 'N/A'}%")
    print(f"Learning Improvement - Static: {static_improvement if static_improvement is not None else 'N/A'}%")
    print(f"Retention Improvement - Adaptive: {adaptive_retention if adaptive_retention is not None else 'N/A'}%")
    print(f"Retention Improvement - Static: {static_retention if static_retention is not None else 'N/A'}%")
    print(
        f"Cognitive Failure Rate - Adaptive: {adaptive_failure_rate if adaptive_failure_rate is not None else 'N/A'}%")
    print(f"Cognitive Failure Rate - Static: {static_failure_rate if static_failure_rate is not None else 'N/A'}%")

    cog_counts = pd.Series(cog_states).value_counts()
    ind_counts = pd.Series(ind_states).value_counts()
    print("\nCognitive Load Diversity:")
    print(cog_counts)
    print("\nIndividual Profile Diversity:")
    print(ind_counts)

    vr_adaptive.quit()
    vr_static.quit()


if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())
