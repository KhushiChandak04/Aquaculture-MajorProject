"""Generate dataset summaries for Productivity, Sustainability, and Genomic tracks."""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def generate_track_summaries():
    """Generate summaries for all three tracks."""
    
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / "processed" / "final_dataset.csv"
    
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    
    # Create target labels (all three tracks use production-based quantile split)
    target = pd.qcut(df["production"], q=3, labels=["Low", "Medium", "High"])
    
    # Generate summary data
    summaries = {}
    
    # 1. PRODUCTIVITY TRACK
    productivity_X = df.drop(columns=[c for c in [
        "production", "Production_Category", "disease_risk_target", 
        "disease_risk_score", "genomic_Disease_Risk_global_mode"
    ] if c in df.columns], errors="ignore")
    
    X_train, X_test, y_train, y_test = train_test_split(
        productivity_X, target, test_size=0.2, random_state=42, stratify=target
    )
    
    train_dist = y_train.value_counts().sort_index()
    test_dist = y_test.value_counts().sort_index()
    
    summaries['Productivity'] = {
        'features': len(productivity_X.columns),
        'feature_names': productivity_X.columns.tolist(),
        'total_samples': len(y_train) + len(y_test),
        'train_samples': len(y_train),
        'test_samples': len(y_test),
        'train_dist': train_dist,
        'test_dist': test_dist,
    }
    
    # 2. SUSTAINABILITY TRACK
    sustainability_features = [
        "country", "year", "temperature_celsius", "precip_mm", "humidity",
        "water_Salinity (ppt)", "water_pH", "water_SecchiDepth (m)",
        "water_WaterDepth (m)", "water_WaterTemp (C)", "water_AirTemp (C)",
        "genomic_GC_Content_global_mean", "genomic_AT_Content_global_mean",
        "genomic_Mutation_Flag_global_mean",
    ]
    sustainability_features = [c for c in sustainability_features if c in df.columns]
    
    sustainability_X = df[sustainability_features].copy()
    
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        sustainability_X, target, test_size=0.2, random_state=42, stratify=target
    )
    
    train_dist_s = y_train_s.value_counts().sort_index()
    test_dist_s = y_test_s.value_counts().sort_index()
    
    summaries['Sustainability'] = {
        'features': len(sustainability_X.columns),
        'feature_names': sustainability_X.columns.tolist(),
        'total_samples': len(y_train_s) + len(y_test_s),
        'train_samples': len(y_train_s),
        'test_samples': len(y_test_s),
        'train_dist': train_dist_s,
        'test_dist': test_dist_s,
    }
    
    # 3. GENOMIC TRACK
    genomic_cols = [c for c in df.columns if c.startswith("genomic_")]
    context_cols = [c for c in ["country", "year", "temperature_celsius", "precip_mm", "humidity"] if c in df.columns]
    genomic_features = genomic_cols + context_cols
    
    genomic_X = df[genomic_features].copy()
    
    X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(
        genomic_X, target, test_size=0.2, random_state=42, stratify=target
    )
    
    train_dist_g = y_train_g.value_counts().sort_index()
    test_dist_g = y_test_g.value_counts().sort_index()
    
    summaries['Genomic'] = {
        'features': len(genomic_X.columns),
        'feature_names': genomic_X.columns.tolist(),
        'total_samples': len(y_train_g) + len(y_test_g),
        'train_samples': len(y_train_g),
        'test_samples': len(y_test_g),
        'train_dist': train_dist_g,
        'test_dist': test_dist_g,
    }
    
    return summaries


def create_summary_table(summaries):
    """Create a markdown table summarizing all tracks."""
    
    lines = []
    lines.append("# Dataset Summary by Track\n")
    
    for track_name, track_data in summaries.items():
        lines.append(f"## {track_name} Track\n")
        lines.append(f"**Features:** {track_data['features']}")
        lines.append(f"\n**Total Samples:** {track_data['total_samples']}\n")
        
        # Create distribution table
        table_rows = [
            "| Sr. No | Class Label | Training Split | Validation Split |",
            "|--------|-------------|----------------|------------------|",
        ]
        
        for idx, class_label in enumerate(["Low", "Medium", "High"], 1):
            train_count = track_data['train_dist'].get(class_label, 0)
            test_count = track_data['test_dist'].get(class_label, 0)
            table_rows.append(f"| {idx} | {class_label} | {train_count} | {test_count} |")
        
        lines.extend(table_rows)
        lines.append("")
        
        # Summary statistics
        lines.append(f"**Training Samples:** {track_data['train_samples']}")
        lines.append(f"\n**Validation Samples:** {track_data['test_samples']}\n")
        
        # Feature list
        lines.append(f"**Features ({track_data['features']}):**")
        for feat in track_data['feature_names']:
            lines.append(f"- {feat}")
        lines.append("")
    
    return "\n".join(lines)


def main():
    """Generate and save summaries."""
    repo_root = Path(__file__).resolve().parents[1]
    
    summaries = generate_track_summaries()
    
    if not summaries:
        print("Failed to generate summaries")
        return
    
    summary_table = create_summary_table(summaries)
    
    output_path = repo_root / "results" / "track_dataset_summaries.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary_table)
    
    print(f"[OK] Saved track summaries to: {output_path}")
    
    # Also create a CSV version with stats
    stats_data = []
    for track_name, track_data in summaries.items():
        for class_label in ["Low", "Medium", "High"]:
            train_count = track_data['train_dist'].get(class_label, 0)
            test_count = track_data['test_dist'].get(class_label, 0)
            stats_data.append({
                'Track': track_name,
                'Class': class_label,
                'Training_Count': train_count,
                'Training_Percentage': round(100 * train_count / track_data['train_samples'], 2),
                'Validation_Count': test_count,
                'Validation_Percentage': round(100 * test_count / track_data['test_samples'], 2),
                'Total_Count': train_count + test_count,
                'Total_Percentage': round(100 * (train_count + test_count) / track_data['total_samples'], 2),
            })
    
    stats_df = pd.DataFrame(stats_data)
    csv_path = repo_root / "results" / "track_dataset_statistics.csv"
    stats_df.to_csv(csv_path, index=False)
    
    print(f"[OK] Saved track statistics to: {csv_path}")
    
    # Print to console
    print("\n" + "="*80)
    print(summary_table)
    print("\n" + stats_df.to_string(index=False))


if __name__ == "__main__":
    main()
