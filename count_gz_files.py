
import os

# --- 请在这里修改为您要统计的文件夹路径 ---
parent_directory_path = "YOUR_FOLDER_PATH_HERE"
# -----------------------------------------

def count_gz_files_in_subdirs(parent_dir):
    """
    统计指定父文件夹下所有直接子文件夹中包含的 .gz 文件数量。

    Args:
        parent_dir (str): 要统计的父文件夹路径。
    """
    # 检查路径是否存在且是否为文件夹
    if not os.path.isdir(parent_dir):
        print(f"错误：提供的路径 '{parent_dir}' 不是一个有效的文件夹。")
        return

    print(f"正在统计文件夹 '{parent_dir}' 的子文件夹中的 .gz 文件...")
    print("-" * 30)

    subdir_gz_counts = {}
    total_gz_count = 0

    # 遍历父文件夹下的所有项目
    for name in sorted(os.listdir(parent_dir)):
        full_path = os.path.join(parent_dir, name)
        
        # 检查是否是文件夹
        if os.path.isdir(full_path):
            try:
                # 统计该子文件夹下的 .gz 文件数量
                gz_files = [f for f in os.listdir(full_path) 
                            if f.endswith('.gz') and os.path.isfile(os.path.join(full_path, f))]
                count = len(gz_files)
                subdir_gz_counts[name] = count
                total_gz_count += count
            except OSError as e:
                print(f"无法访问子文件夹 '{name}' 或其内容: {e}")


    # --- 生成并打印总结报告 ---
    print("【总结报告】")
    if not subdir_gz_counts:
        print("未在任何子文件夹中找到 .gz 文件或没有子文件夹。")
    else:
        for subdir, count in subdir_gz_counts.items():
            print(f"- 子文件夹 '{subdir}': {count} 个 .gz 文件")

    print("-" * 30)
    print(f"总计: {total_gz_count} 个 .gz 文件")

if __name__ == "__main__":
    # 检查用户是否修改了路径
    if parent_directory_path == "YOUR_FOLDER_PATH_HERE":
        print("请先修改脚本中的 'parent_directory_path' 变量，指向您要统计的文件夹。")
    else:
        count_gz_files_in_subdirs(parent_directory_path)
