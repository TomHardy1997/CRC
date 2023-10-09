import os
import pandas as pd
from sklearn.utils import shuffle

# 指定数据集的根目录
root_dir = 'copy'

# 创建一个空的列表，用于存储数据
data_list = []

# 定义一个函数，用于处理每个类别文件夹
def process_class_folder(class_folder):
    class_folder_path = os.path.join(root_dir, class_folder)
    if os.path.isdir(class_folder_path):
        for data_file in os.listdir(class_folder_path):
            if data_file.endswith('.jpg') or data_file.endswith('.tif'):
                data_file_path = os.path.join(class_folder_path, data_file)
                data_list.append({'path': data_file_path, 'class': class_folder})

# 遍历每个类别文件夹
class_folders = os.listdir(root_dir)
for class_folder in class_folders:
    process_class_folder(class_folder)

# 将数据列表转换为 DataFrame
df = pd.DataFrame(data_list)

# 打乱 DataFrame 中的数据
df = shuffle(df).reset_index(drop=True)

# 打印打乱后的 DataFrame 的前几行
print(df.head())

# 可以将 DataFrame 写入 CSV 文件，如果需要
df.to_csv('shuffled_data.csv', index=False)
