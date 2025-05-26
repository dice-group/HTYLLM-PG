import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict
import os

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create plots directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Load the results data
with open("results_custom_gpt.json", "r") as f:
    results = json.load(f)

# Extract language codes, scripts, and scores
languages = list(results.keys())
scores = list(results.values())

# Calculate average
average_score = np.mean(scores)
print(f"Average score across all languages: {average_score:.3f}")

# Parse language codes to extract script information
def parse_language_code(lang_code):
    """Parse language_script format to extract language and script"""
    parts = lang_code.split('_')
    if len(parts) >= 2:
        return parts[0], parts[1]
    return parts[0], 'Unknown'

# Group by script
script_groups = defaultdict(list)
for lang_code, score in results.items():
    lang, script = parse_language_code(lang_code)
    script_groups[script].append((lang_code, score))

# Sort languages by score for better visualization
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
sorted_languages, sorted_scores = zip(*sorted_results)

# Create multiple visualizations

# 1. Horizontal bar chart of all languages
plt.figure(figsize=(14, 20))
y_pos = np.arange(len(sorted_languages))
bars = plt.barh(y_pos, sorted_scores, alpha=0.8)

# Color bars based on score (gradient from red to green)
colors = plt.cm.RdYlGn([score/max(sorted_scores) for score in sorted_scores])
for bar, color in zip(bars, colors):
    bar.set_color(color)

plt.yticks(y_pos, sorted_languages, fontsize=8)
plt.xlabel('Score', fontsize=12)
plt.title('Language Performance Results (Sorted by Score)', fontsize=14, fontweight='bold')
plt.axvline(x=average_score, color='red', linestyle='--', linewidth=2, 
           label=f'Average: {average_score:.3f}')
plt.legend()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig("plots/language_performance_sorted.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Distribution histogram
plt.figure(figsize=(12, 8))
plt.hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(x=average_score, color='red', linestyle='--', linewidth=2, 
           label=f'Average: {average_score:.3f}')
plt.xlabel('Score', fontsize=12)
plt.ylabel('Number of Languages', fontsize=12)
plt.title('Distribution of Language Performance Scores', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plots/score_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. Box plot by script
script_scores = {}
for script, lang_scores in script_groups.items():
    script_scores[script] = [score for _, score in lang_scores]

# Only include scripts with multiple languages for meaningful box plots
scripts_with_multiple = {k: v for k, v in script_scores.items() if len(v) > 1}

if scripts_with_multiple:
    plt.figure(figsize=(14, 8))
    script_names = list(scripts_with_multiple.keys())
    script_data = [scripts_with_multiple[script] for script in script_names]
    
    box_plot = plt.boxplot(script_data, labels=script_names, patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(script_names)))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.axhline(y=average_score, color='red', linestyle='--', linewidth=2, 
               label=f'Overall Average: {average_score:.3f}')
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Script', fontsize=12)
    plt.title('Performance Distribution by Script (Scripts with Multiple Languages)', 
              fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/performance_by_script.png", dpi=300, bbox_inches='tight')
    plt.close()

# 4. Script averages bar chart
script_averages = {}
script_counts = {}
for script, lang_scores in script_groups.items():
    scores_only = [score for _, score in lang_scores]
    script_averages[script] = np.mean(scores_only)
    script_counts[script] = len(scores_only)

# Sort scripts by average score
sorted_scripts = sorted(script_averages.items(), key=lambda x: x[1], reverse=True)
script_names, script_avg_scores = zip(*sorted_scripts)

plt.figure(figsize=(14, 8))
bars = plt.bar(range(len(script_names)), script_avg_scores, alpha=0.8)

# Color bars based on score
colors = plt.cm.RdYlGn([score/max(script_avg_scores) for score in script_avg_scores])
for bar, color in zip(bars, colors):
    bar.set_color(color)

# Add count labels on bars
for i, (script, avg_score) in enumerate(sorted_scripts):
    count = script_counts[script]
    plt.text(i, avg_score + 0.005, f'n={count}', ha='center', va='bottom', fontsize=9)

plt.xticks(range(len(script_names)), script_names, rotation=45, ha='right')
plt.ylabel('Average Score', fontsize=12)
plt.xlabel('Script', fontsize=12)
plt.title('Average Performance by Script', fontsize=14, fontweight='bold')
plt.axhline(y=average_score, color='red', linestyle='--', linewidth=2, 
           label=f'Overall Average: {average_score:.3f}')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("plots/script_averages.png", dpi=300, bbox_inches='tight')
plt.close()

# 5. Summary statistics
print("\n=== SUMMARY STATISTICS ===")
print(f"Total languages: {len(results)}")
print(f"Average score: {average_score:.3f}")
print(f"Median score: {np.median(scores):.3f}")
print(f"Standard deviation: {np.std(scores):.3f}")
print(f"Min score: {min(scores):.3f} ({sorted_results[-1][0]})")
print(f"Max score: {max(scores):.3f} ({sorted_results[0][0]})")

print(f"\n=== TOP 10 LANGUAGES ===")
for i, (lang, score) in enumerate(sorted_results[:10]):
    print(f"{i+1:2d}. {lang}: {score:.3f}")

print(f"\n=== BOTTOM 10 LANGUAGES ===")
for i, (lang, score) in enumerate(sorted_results[-10:]):
    print(f"{len(sorted_results)-9+i:2d}. {lang}: {score:.3f}")

print(f"\n=== SCRIPT STATISTICS ===")
print(f"Number of scripts: {len(script_groups)}")
for script, avg_score in sorted_scripts:
    count = script_counts[script]
    print(f"{script}: {avg_score:.3f} (n={count})")

print("\nAll plots saved to 'plots' directory!") 