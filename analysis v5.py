"""
Analysis script for evaluating XI VOCAL volumetric ultrasound measurements.
This script performs statistical analyses to answer research questions about accuracy,
optimal slice number, system comparison, inter-observer variability, precision, and bias.

Results are saved to both results.txt (human-readable) and results.csv (structured format).

Author: Grant Delanoy
Date: April 2025
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import redirect_stdout
import os
from statsmodels.stats.anova import AnovaRM
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pingouin as pg

# Constants
TRUE_VOLUME = 172  # True target volume in mL
SYSTEMS = ["W10", "Z20"]
SLICE_NUMBERS = [5, 10, 15, 20]
OUTPUT_DIR = "output"

# Global list to store results for CSV export
results_list = []

def setup_output_directory():
    """Create the output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def create_tee_output():
    """
    Create a Tee object to redirect print statements to both terminal and a file.
    Returns the Tee object and file handle.
    """
    results_file = os.path.join(OUTPUT_DIR, "results.txt")
    f = open(results_file, "w")
    
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for file in self.files:
                file.write(obj)
                file.flush()
        def flush(self):
            for file in self.files:
                file.flush()
    
    return Tee(sys.stdout, f), f

def save_results_to_csv():
    """Save collected results to results.csv."""
    df = pd.DataFrame(results_list)
    df.to_csv(os.path.join(OUTPUT_DIR, "results.csv"), index=False)

def load_data():
    """Load the data from the Excel file and return the DataFrame."""
    return pd.read_excel("data.xlsx")

def descriptive_statistics(data, tee):
    """Calculate and save descriptive statistics by System, Slices, and Observer."""
    summary = data.groupby(["System", "Slices", "Observer"])["Volume"].agg(["mean", "std", "count"])
    print("\nDescriptive Statistics by System, Slices, and Observer:", file=tee)
    print(summary, file=tee)
    summary.to_csv(os.path.join(OUTPUT_DIR, "descriptive_stats.csv"))

    # Add to results_list
    for (system, slices, observer), row in summary.iterrows():
        results_list.append({
            "Analysis": "Descriptive Statistics",
            "Group": f"System: {system}, Slices: {slices}, Observer: {observer}",
            "Metric": "Mean Volume (mL)",
            "Value": f"{row['mean']:.3f}",
            "Interpretation": ""
        })
        results_list.append({
            "Analysis": "Descriptive Statistics",
            "Group": f"System: {system}, Slices: {slices}, Observer: {observer}",
            "Metric": "Standard Deviation (mL)",
            "Value": f"{row['std']:.3f}",
            "Interpretation": ""
        })
        results_list.append({
            "Analysis": "Descriptive Statistics",
            "Group": f"System: {system}, Slices: {slices}, Observer: {observer}",
            "Metric": "Count",
            "Value": f"{int(row['count'])}",
            "Interpretation": ""
        })

def calculate_error_metrics(data, tee):
    """Calculate MAE and percent error from the true volume."""
    data["MAE"] = abs(data["Volume"] - TRUE_VOLUME)
    data["Percent_Error"] = (abs(data["Volume"] - TRUE_VOLUME) / TRUE_VOLUME) * 100
    
    mae_summary = data.groupby(["System", "Slices"])["MAE"].mean()
    percent_error_summary = data.groupby(["System", "Slices"])["Percent_Error"].mean()
    
    print("\nMean Absolute Error (MAE) by System and Slices:", file=tee)
    print(mae_summary, file=tee)
    print("\nPercent Error by System and Slices:", file=tee)
    print(percent_error_summary, file=tee)
    
    mae_summary.to_csv(os.path.join(OUTPUT_DIR, "mae_summary.csv"))
    percent_error_summary.to_csv(os.path.join(OUTPUT_DIR, "percent_error_summary.csv"))

    # Add to results_list
    for (system, slices), mae in mae_summary.items():
        results_list.append({
            "Analysis": "Error Metrics",
            "Group": f"System: {system}, Slices: {slices}",
            "Metric": "Mean Absolute Error (mL)",
            "Value": f"{mae:.3f}",
            "Interpretation": ""
        })
    for (system, slices), pe in percent_error_summary.items():
        results_list.append({
            "Analysis": "Error Metrics",
            "Group": f"System: {system}, Slices: {slices}",
            "Metric": "Percent Error (%)",
            "Value": f"{pe:.3f}",
            "Interpretation": ""
        })

def normality_testing(data, tee):
    """Perform Shapiro-Wilk test for normality on each group."""
    groups = data.groupby(["System", "Slices", "Observer"])
    normality_results = {}
    
    print("\nShapiro-Wilk Normality Test Results (p-values):", file=tee)
    for name, group in groups:
        stat, p = stats.shapiro(group["Volume"])
        normality_results[name] = p
        print(f"Group {name}: p-value = {p:.4f}", file=tee)
        print(f"  -> {'Normal' if p > 0.05 else 'Not Normal'} (p {'>' if p > 0.05 else '<='} 0.05)", file=tee)

        # Add to results_list
        results_list.append({
            "Analysis": "Normality Testing (Shapiro-Wilk)",
            "Group": f"System: {name[0]}, Slices: {name[1]}, Observer: {name[2]}",
            "Metric": "p-value",
            "Value": f"{p:.4f}",
            "Interpretation": "Normal" if p > 0.05 else "Not Normal"
        })
    
    return normality_results

def accuracy_analysis(data, normality_results, tee):
    """Compare each group's mean volume to the true volume using t-tests or Wilcoxon tests."""
    groups = data.groupby(["System", "Slices", "Observer"])
    
    print("\nAccuracy Analysis (Comparison to True Volume):", file=tee)
    for name, group in groups:
        mean_volume = group["Volume"].mean()
        std_volume = group["Volume"].std()
        if normality_results[name] > 0.05:
            t_stat, p_val = stats.ttest_1samp(group["Volume"], TRUE_VOLUME)
            test_name = "t-test"
            cohen_d = (mean_volume - TRUE_VOLUME) / std_volume
        else:
            stat, p_val = stats.wilcoxon(group["Volume"] - TRUE_VOLUME)
            test_name = "Wilcoxon"
            cohen_d = None
        
        print(f"\nGroup {name}:", file=tee)
        print(f"  Mean Volume = {mean_volume:.3f} mL", file=tee)
        print(f"  {test_name} p-value = {p_val:.4f}", file=tee)
        print(f"  -> {'Significantly different' if p_val < 0.05 else 'Not significantly different'} from true volume (p {'<' if p_val < 0.05 else '>='} 0.05)", file=tee)
        if cohen_d is not None:
            print(f"  Cohen's d = {cohen_d:.3f}", file=tee)

        # Add to results_list
        results_list.append({
            "Analysis": "Accuracy Analysis",
            "Group": f"System: {name[0]}, Slices: {name[1]}, Observer: {name[2]}",
            "Metric": "Mean Volume (mL)",
            "Value": f"{mean_volume:.3f}",
            "Interpretation": ""
        })
        results_list.append({
            "Analysis": "Accuracy Analysis",
            "Group": f"System: {name[0]}, Slices: {name[1]}, Observer: {name[2]}",
            "Metric": f"{test_name} p-value",
            "Value": f"{p_val:.4f}",
            "Interpretation": "Significantly different" if p_val < 0.05 else "Not significantly different"
        })
        if cohen_d is not None:
            results_list.append({
                "Analysis": "Accuracy Analysis",
                "Group": f"System: {name[0]}, Slices: {name[1]}, Observer: {name[2]}",
                "Metric": "Cohen's d",
                "Value": f"{cohen_d:.3f}",
                "Interpretation": ""
            })

def optimal_slice_analysis(data, tee):
    """Perform Repeated Measures ANOVA with post-hoc tests to assess optimal slice number."""
    data["Subject"] = data["Image"].astype(str) + "_" + data["Observer"].astype(str)
    
    print("\nOptimal Slice Number Analysis (Repeated Measures ANOVA):", file=tee)
    for system in SYSTEMS:
        system_data = data[data["System"] == system]
        anova = AnovaRM(system_data, depvar="Volume", subject="Subject", within=["Slices"])
        anova_results = anova.fit()
        print(f"\nSystem: {system}", file=tee)
        print(anova_results, file=tee)

        p_value = anova_results.anova_table["Pr > F"][0]
        # Add to results_list
        results_list.append({
            "Analysis": "Optimal Slice Analysis (ANOVA)",
            "Group": f"System: {system}",
            "Metric": "p-value",
            "Value": f"{p_value:.4f}",
            "Interpretation": "Significant" if p_value < 0.05 else "Not significant"
        })

        # Post-hoc pairwise t-tests with Bonferroni correction if ANOVA is significant
        if p_value < 0.05:
            print("\nPost-hoc Pairwise Comparisons (Bonferroni corrected):", file=tee)
            slice_numbers = [5, 10, 15, 20]
            comparisons = [(s1, s2) for i, s1 in enumerate(slice_numbers) for s2 in slice_numbers[i+1:]]
            alpha = 0.05 / len(comparisons)  # Bonferroni correction
            for s1, s2 in comparisons:
                group1 = system_data[system_data["Slices"] == s1]["Volume"]
                group2 = system_data[system_data["Slices"] == s2]["Volume"]
                t_stat, p_val = stats.ttest_rel(group1, group2)
                print(f"Slices {s1} vs {s2}: p-value = {p_val:.4f}", file=tee)
                print(f"  -> {'Significant' if p_val < alpha else 'Not significant'} (p {'<' if p_val < alpha else '>='} {alpha:.4f})", file=tee)

                # Add to results_list
                results_list.append({
                    "Analysis": "Optimal Slice Analysis (Post-hoc)",
                    "Group": f"System: {system}, Slices {s1} vs {s2}",
                    "Metric": "p-value",
                    "Value": f"{p_val:.4f}",
                    "Interpretation": f"Significant (Bonferroni α = {alpha:.4f})" if p_val < alpha else f"Not significant (Bonferroni α = {alpha:.4f})"
                })

def precision_analysis(data, tee):
    """Calculate Coefficient of Variation (CV) and perform Levene's test for variances."""
    # CV calculation
    cv_summary = data.groupby(["System", "Slices"])["Volume"].agg(
        lambda x: (np.std(x) / np.mean(x)) * 100
    ).rename("CV (%)")
    
    print("\nPrecision Analysis (Coefficient of Variation):", file=tee)
    print(cv_summary, file=tee)
    cv_summary.to_csv(os.path.join(OUTPUT_DIR, "cv_summary.csv"))
    
    # Add to results_list
    for (system, slices), cv in cv_summary.items():
        results_list.append({
            "Analysis": "Precision Analysis",
            "Group": f"System: {system}, Slices: {slices}",
            "Metric": "Coefficient of Variation (%)",
            "Value": f"{cv:.3f}",
            "Interpretation": ""
        })
    
    # Levene's test across slice numbers
    print("\nLevene's Test for Variances Across Slice Numbers:", file=tee)
    for system in SYSTEMS:
        system_data = data[data["System"] == system]
        slices_groups = [system_data[system_data["Slices"] == s]["Volume"] for s in SLICE_NUMBERS]
        stat, p = stats.levene(*slices_groups)
        print(f"System {system}:", file=tee)
        print(f"  Levene's test p-value = {p:.4f}", file=tee)
        print(f"  -> Variances are {'significantly different' if p < 0.05 else 'not significantly different'} across slice numbers (p {'<' if p < 0.05 else '>='} 0.05)", file=tee)

        # Add to results_list
        results_list.append({
            "Analysis": "Precision Analysis (Levene's Test Across Slices)",
            "Group": f"System: {system}",
            "Metric": "p-value",
            "Value": f"{p:.4f}",
            "Interpretation": "Variances significantly different" if p < 0.05 else "Variances not significantly different"
        })
    
    # Levene's test between systems
    print("\nLevene's Test for Variances Between Systems (by Slice Number):", file=tee)
    for slices in SLICE_NUMBERS:
        w10_volumes = data[(data["System"] == "W10") & (data["Slices"] == slices)]["Volume"]
        z20_volumes = data[(data["System"] == "Z20") & (data["Slices"] == slices)]["Volume"]
        stat, p = stats.levene(w10_volumes, z20_volumes)
        print(f"Slices {slices}:", file=tee)
        print(f"  Levene's test p-value = {p:.4f}", file=tee)
        print(f"  -> Variances are {'significantly different' if p < 0.05 else 'not significantly different'} between W10 and Z20 (p {'<' if p < 0.05 else '>='} 0.05)", file=tee)

        # Add to results_list
        results_list.append({
            "Analysis": "Precision Analysis (Levene's Test Between Systems)",
            "Group": f"Slices: {slices}",
            "Metric": "p-value",
            "Value": f"{p:.4f}",
            "Interpretation": "Variances significantly different" if p < 0.05 else "Variances not significantly different"
        })

def system_comparison(data, tee):
    """Perform two-way ANOVA to compare systems and calculate overall MAE and CV."""
    # Two-way ANOVA
    model = ols("Volume ~ C(System) + C(Slices) + C(System):C(Slices)", data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    print("\nSystem Comparison (Two-way ANOVA):", file=tee)
    print(anova_table, file=tee)
    anova_table.to_csv(os.path.join(OUTPUT_DIR, "system_comparison_anova.csv"))
    
    # Add to results_list
    for index, row in anova_table.iterrows():
        if index != "Residual":
            results_list.append({
                "Analysis": "System Comparison (Two-way ANOVA)",
                "Group": index,
                "Metric": "p-value",
                "Value": f"{row['PR(>F)']:.4f}",
                "Interpretation": "Significant" if row['PR(>F)'] < 0.05 else "Not significant"
            })
    
    # MAE comparison
    mae_system = data.groupby("System")["MAE"].mean()
    print("\nMAE Comparison Between Systems:", file=tee)
    print(mae_system, file=tee)
    mae_system.to_csv(os.path.join(OUTPUT_DIR, "mae_system.csv"))
    
    # Add to results_list
    for system, mae in mae_system.items():
        results_list.append({
            "Analysis": "System Comparison",
            "Group": f"System: {system}",
            "Metric": "Mean Absolute Error (mL)",
            "Value": f"{mae:.3f}",
            "Interpretation": ""
        })
    
    # CV comparison
    cv_system = data.groupby("System")["Volume"].agg(
        lambda x: (np.std(x) / np.mean(x)) * 100
    ).rename("CV (%)")
    print("\nCV Comparison Between Systems:", file=tee)
    print(cv_system, file=tee)
    cv_system.to_csv(os.path.join(OUTPUT_DIR, "cv_system.csv"))
    
    # Add to results_list
    for system, cv in cv_system.items():
        results_list.append({
            "Analysis": "System Comparison",
            "Group": f"System: {system}",
            "Metric": "Coefficient of Variation (%)",
            "Value": f"{cv:.3f}",
            "Interpretation": ""
        })

def inter_observer_variability(data, tee):
    """Calculate ICC and generate Bland-Altman plots with detailed limits of agreement."""
    # Set font to Arial for all plot elements
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial']
    
    icc_data = data.pivot_table(
        index=["System", "Slices", "Image"],
        columns="Observer",
        values="Volume"
    ).reset_index()
    icc_data = icc_data.rename(columns={1: "Observer_1_Volume", 2: "Observer_2_Volume"})
    
    # ICC calculation
    print("\nInter-Observer Variability Analysis:", file=tee)
    print("\nIntraclass Correlation Coefficient (ICC):", file=tee)
    for system in SYSTEMS:
        for slices in SLICE_NUMBERS:
            subset = icc_data[(icc_data["System"] == system) & (icc_data["Slices"] == slices)]
            icc = pg.intraclass_corr(
                data=subset.melt(id_vars=["System", "Slices", "Image"],
                                 value_vars=["Observer_1_Volume", "Observer_2_Volume"],
                                 var_name="Observer",
                                 value_name="Volume"),
                targets="Image",
                raters="Observer",
                ratings="Volume"
            ).set_index("Type").loc["ICC2", "ICC"]
            print(f"System {system}, Slices {slices}: ICC = {icc:.3f}", file=tee)
            if icc < 0.5:
                print("  -> Poor agreement", file=tee)
                interp = "Poor agreement"
            elif icc < 0.75:
                print("  -> Moderate agreement", file=tee)
                interp = "Moderate agreement"
            elif icc < 0.9:
                print("  -> Good agreement", file=tee)
                interp = "Good agreement"
            else:
                print("  -> Excellent agreement", file=tee)
                interp = "Excellent agreement"

            # Add to results_list
            results_list.append({
                "Analysis": "Inter-Observer Variability (ICC)",
                "Group": f"System: {system}, Slices: {slices}",
                "Metric": "ICC",
                "Value": f"{icc:.3f}",
                "Interpretation": interp
            })
    
    # Bland-Altman plots with superimposed systems by slice number
    print("\nBland-Altman Analysis:", file=tee)
    
    # Create a 2x2 subplot grid for the four slice numbers
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()  # Flatten the 2x2 array for easier indexing
    slice_numbers = SLICE_NUMBERS  # [5, 10, 15, 20]
    
    # Grayscale colors for systems (matching boxplots)
    colors = {"W10": "0.05", "Z20": "0.4"}  # W10: dark gray, Z20: light gray
    
    for idx, slices in enumerate(slice_numbers):
        ax = axes[idx]
        # Store handles and labels for the legend
        legend_handles = []
        legend_labels = []
        
        for system in SYSTEMS:
            subset = icc_data[(icc_data["System"] == system) & (icc_data["Slices"] == slices)]
            mean_vol = (subset["Observer_1_Volume"] + subset["Observer_2_Volume"]) / 2
            diff_vol = subset["Observer_1_Volume"] - subset["Observer_2_Volume"]
            mean_diff = diff_vol.mean()
            sd_diff = diff_vol.std()
            loa_upper = mean_diff + 1.96 * sd_diff
            loa_lower = mean_diff - 1.96 * sd_diff
            loa_width = loa_upper - loa_lower
            
            # Plot scatter points for this system
            scatter = ax.scatter(mean_vol, diff_vol, color=colors[system], alpha=0.5)
            # Plot mean difference and limits of agreement
            ax.axhline(mean_diff, color=colors[system], linestyle="--", alpha=0.7)
            ax.axhline(loa_upper, color=colors[system], linestyle=":", alpha=0.5)
            ax.axhline(loa_lower, color=colors[system], linestyle=":", alpha=0.5)
            
            # Add to legend with bold system name
            legend_handles.append(scatter)
            legend_labels.append(f"$\\mathbf{{{system}}}$: Mean Diff = {mean_diff:.2f}, \n±1.96 SD = [{loa_lower:.2f}, {loa_upper:.2f}]")
            
            # Print detailed results (same as before)
            print(f"\nSystem {system}, Slices {slices}:", file=tee)
            print(f"  Mean Difference (Observer 1 - Observer 2) = {mean_diff:.3f} mL", file=tee)
            print(f"  Standard Deviation of Differences = {sd_diff:.3f} mL", file=tee)
            print(f"  95% Limits of Agreement = [{loa_lower:.3f}, {loa_upper:.3f}] mL", file=tee)
            print(f"  Width of Limits of Agreement = {loa_width:.3f} mL", file=tee)
            
            # Add to results_list (same as before)
            results_list.append({
                "Analysis": "Inter-Observer Variability (Bland-Altman)",
                "Group": f"System: {system}, Slices: {slices}",
                "Metric": "Mean Difference (mL)",
                "Value": f"{mean_diff:.3f}",
                "Interpretation": ""
            })
            results_list.append({
                "Analysis": "Inter-Observer Variability (Bland-Altman)",
                "Group": f"System: {system}, Slices: {slices}",
                "Metric": "Standard Deviation of Differences (mL)",
                "Value": f"{sd_diff:.3f}",
                "Interpretation": ""
            })
            results_list.append({
                "Analysis": "Inter-Observer Variability (Bland-Altman)",
                "Group": f"System: {system}, Slices: {slices}",
                "Metric": "95% Limits of Agreement (mL)",
                "Value": f"[{loa_lower:.3f}, {loa_upper:.3f}]",
                "Interpretation": ""
            })
            results_list.append({
                "Analysis": "Inter-Observer Variability (Bland-Altman)",
                "Group": f"System: {system}, Slices: {slices}",
                "Metric": "Width of Limits of Agreement (mL)",
                "Value": f"{loa_width:.3f}",
                "Interpretation": ""
            })
        
        # Customize each subplot
        ax.set_title(f"{slices} Slices")
        ax.set_xlabel("Mean Volume (mL)")
        ax.set_ylabel("Difference (Obs 1 - Obs 2) (mL)")
        # Custom x-axis for 10 Slices plot (index 1)
        if slices == 10:
            ax.set_xticks(np.arange(165, 186, 5))  # 165 to 185, step by 5
        # Add custom legend with LoA information and adjusted font size
        ax.legend(handles=legend_handles, labels=legend_labels, loc="lower right", fontsize=7.5)
    
    # Adjust layout to minimize whitespace
    plt.tight_layout()
    # Save the combined figure
    plt.savefig(os.path.join(OUTPUT_DIR, "bland_altman_combined.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved combined Bland-Altman plot: bland_altman_combined.png", file=tee)

def bias_assessment(data, tee):
    """Assess systematic bias with ANOVA, post-hoc tests for Slices:Observer, and boxplots."""
    # Set font to Arial for all plot elements
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial']
    
    data["Difference"] = data["Volume"] - TRUE_VOLUME
    
    # Mean difference
    diff_summary = data.groupby(["System", "Slices", "Observer"])["Difference"].mean()
    print("\nBias Assessment:", file=tee)
    print("\nMean Difference from True Volume (Measured - True):", file=tee)
    print(diff_summary, file=tee)
    diff_summary.to_csv(os.path.join(OUTPUT_DIR, "difference_summary.csv"))
    
    # Add to results_list
    for (system, slices, observer), diff in diff_summary.items():
        results_list.append({
            "Analysis": "Bias Assessment",
            "Group": f"System: {system}, Slices: {slices}, Observer: {observer}",
            "Metric": "Mean Difference (mL)",
            "Value": f"{diff:.3f}",
            "Interpretation": ""
        })
    
    # Two-way ANOVA
    model = ols("Difference ~ C(System) + C(Slices) + C(Observer) + C(System):C(Slices) + C(System):C(Observer) + C(Slices):C(Observer)", data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print("\nTwo-way ANOVA on Differences (Testing for Systematic Bias):", file=tee)
    print(anova_table, file=tee)
    anova_table.to_csv(os.path.join(OUTPUT_DIR, "bias_anova.csv"))
    
    # Add to results_list
    for index, row in anova_table.iterrows():
        if index != "Residual":
            results_list.append({
                "Analysis": "Bias Assessment (Two-way ANOVA)",
                "Group": index,
                "Metric": "p-value",
                "Value": f"{row['PR(>F)']:.4f}",
                "Interpretation": "Significant" if row['PR(>F)'] < 0.05 else "Not significant"
            })
    
    # Post-hoc tests for Slices:Observer interaction
    print("\nPost-hoc Tests for Slices:Observer Interaction:", file=tee)
    for slices in SLICE_NUMBERS:
        slice_data = data[data["Slices"] == slices]
        obs1_volumes = slice_data[slice_data["Observer"] == 1]["Difference"]
        obs2_volumes = slice_data[slice_data["Observer"] == 2]["Difference"]
        t_stat, p_val = stats.ttest_ind(obs1_volumes, obs2_volumes)
        print(f"Slices {slices}: Observer 1 vs Observer 2", file=tee)
        print(f"  t-test p-value = {p_val:.4f}", file=tee)
        print(f"  -> {'Significant difference' if p_val < 0.05 else 'No significant difference'} (p {'<' if p_val < 0.05 else '>='} 0.05)", file=tee)

        # Add to results_list
        results_list.append({
            "Analysis": "Bias Assessment (Post-hoc)",
            "Group": f"Slices: {slices}, Observer 1 vs Observer 2",
            "Metric": "t-test p-value",
            "Value": f"{p_val:.4f}",
            "Interpretation": "Significant difference" if p_val < 0.05 else "No significant difference"
        })
    
    # Boxplots with vertically stacked subplots, independent x-axes
    print("\nGenerating Boxplots for Differences...", file=tee)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'hspace': 0.2})
    
    # Boxplot for System vs. Slices (Figure 4a)
    sns.boxplot(x="Slices", y="Difference", hue="System", data=data, palette=["0.3", "0.7"], ax=ax1)
    ax1.axhline(0, color="black", linestyle="--", label="No Bias (Difference = 0)")
    ax1.set_xlabel("Number of Slices")  # X-axis label for top plot
    ax1.set_ylabel("Difference from True Volume (mL)")
    ax1.set_title("Difference from True Volume by System and Slices")
    ax1.legend(title="System", loc="upper right", fontsize=7.5)
    
    # Boxplot for Observer vs. Slices (Figure 4b)
    sns.boxplot(x="Slices", y="Difference", hue="Observer", data=data, palette=["0.4", "0.6"], ax=ax2)
    ax2.axhline(0, color="black", linestyle="--", label="No Bias (Difference = 0)")
    ax2.set_xlabel("Number of Slices")  # X-axis label for bottom plot
    ax2.set_ylabel("Difference from True Volume (mL)")
    ax2.set_title("Difference from True Volume by Observer and Slices")
    ax2.legend(title="Observer", loc="upper right", fontsize=7.5)
    
    # Adjust layout to minimize whitespace
    plt.tight_layout()
    # Save the combined figure
    plt.savefig(os.path.join(OUTPUT_DIR, "boxplot_diff_combined.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved combined boxplot: boxplot_diff_combined.png", file=tee)

def main():
    """Main function to run all analyses."""
    setup_output_directory()
    tee, f = create_tee_output()
    
    try:
        # Load data
        data = load_data()
        
        # Run analyses
        descriptive_statistics(data, tee)
        calculate_error_metrics(data, tee)
        normality_results = normality_testing(data, tee)
        accuracy_analysis(data, normality_results, tee)
        optimal_slice_analysis(data, tee)
        precision_analysis(data, tee)
        system_comparison(data, tee)
        inter_observer_variability(data, tee)
        bias_assessment(data, tee)
        
        # Save all results to CSV
        save_results_to_csv()
    
    finally:
        f.close()

if __name__ == "__main__":
    import sys
    main()