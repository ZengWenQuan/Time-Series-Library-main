
import pandas as pd
import numpy as np
import sys

def analyze_csv(file_path):
    """
    对CSV文件进行详细分析，检查可能导致NaN的常见问题。
    """
    print(f"\n{'='*60}")
    print(f"--- Analyzing File: {file_path} ---")
    print(f"{ '='*60}")
    
    try:
        # 使用pandas加载数据，将第一行设为表头，第一列设为索引
        df = pd.read_csv(file_path, header=0, index_col=0, low_memory=False)
        print(f"\n[INFO] Successfully loaded data with shape: {df.shape}. The first column has been set as the index and is excluded from analysis.")

        # --- 0. 检查非数值列 ---
        print("\n" + "-"*20 + " 0. Non-Numeric Data Check " + "-"*20)
        numeric_df = df.apply(pd.to_numeric, errors='coerce')
        non_numeric_cols = df.columns[numeric_df.isnull().any()].tolist()

        if non_numeric_cols:
            print(f"[!!!] CRITICAL: Found {len(non_numeric_cols)} columns containing non-numeric data.")
            print(f"This is a major issue. First 5 affected column indices: {non_numeric_cols[:5]}")
            # 显示一些问题样本
            for col in non_numeric_cols[:5]:
                print(f"  - Sample of non-numeric data in column {col}: {df[col][pd.to_numeric(df[col], errors='coerce').isnull()].iloc[0]}")
            # 用转换后的数值DataFrame进行后续分析
            df = numeric_df
        else:
            print("[OK] Passed: All columns appear to be numeric.")

        # --- 1. 检查NaN或Inf值 ---
        print("\n" + "-"*20 + " 1. NaN/Inf Check " + "-"*20)
        nan_count = df.isnull().sum().sum()
        inf_count = np.isinf(df.values).sum()

        if nan_count > 0 or inf_count > 0:
            print(f"[!!!] CRITICAL: Found {nan_count} NaN values and {inf_count} Inf values.")
            # 定位NaN所在的行和列
            if nan_count > 0:
                nan_locations = df.isnull().stack()
                print("First 5 NaN locations (row, col):")
                print(nan_locations[nan_locations].head())
        else:
            print("[OK] Passed: No NaN or Inf values found in the dataset.")

        # --- 2. 描述性统计分析 ---
        print("\n" + "-"*20 + " 2. Descriptive Statistics " + "-"*20)
        stats = df.describe()
        print(stats)

        # 检查极端值
        max_val = stats.loc['max'].max()
        min_val = stats.loc['min'].min()
        mean_of_means = stats.loc['mean'].mean()
        std_of_stds = stats.loc['std'].mean()
        
        print(f"\nOverall Max Value: {max_val:.4f}")
        print(f"Overall Min Value: {min_val:.4f}")
        print(f"Average of Column Means: {mean_of_means:.4f}")
        print(f"Average of Column Stds: {std_of_stds:.4f}")

        if max_val > 1e7 or min_val < -1e7:
            print("[WARN] Potential Issue: Data contains very large or small values, which could cause numerical instability.")
        else:
            print("[OK] Passed: Data range seems reasonable.")

        # --- 3. 零方差列检查 ---
        print("\n" + "-"*20 + " 3. Zero Variance Check " + "-"*20)
        # 计算标准差，并允许小的浮点误差
        zero_variance_cols = df.columns[df.std(axis=0) < 1e-9]
        
        if len(zero_variance_cols) > 0:
            print(f"[!!!] CRITICAL: Found {len(zero_variance_cols)} columns with zero (or near-zero) variance.")
            print("These columns are constant and will cause division by zero in normalization layers.")
            print(f"First 5 zero-variance column indices: {zero_variance_cols.tolist()[:5]}")
            # 显示这些列的恒定值
            for col_idx in zero_variance_cols[:5]:
                 print(f"  - Column {col_idx} has a constant value of: {df[col_idx].iloc[0]}")
        else:
            print("[OK] Passed: No columns with zero variance found.")

    except FileNotFoundError:
        print(f"\n[ERROR] File not found: {file_path}")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred during analysis: {e}")
    finally:
        print(f"\n--- End of analysis for: {file_path} ---\\n")


if __name__ == "__main__":
    # 检查文件路径参数
    if len(sys.argv) != 2:
        print("Usage: python3 analyze_data.py <path_to_csv_file>")
        sys.exit(1)
    
    file_to_analyze = sys.argv[1]
    analyze_csv(file_to_analyze)
