import os
import cv2
import lmdb
import numpy as np
from tqdm import tqdm

def create_lmdb_dataset(input_folder, output_lmdb):
    # 打开 LMDB 数据库
    env = lmdb.open(output_lmdb, map_size=int(1e12))

    # 遍历类别文件夹
    for class_name in os.listdir(input_folder):
        class_folder = os.path.join(input_folder, class_name)

        # 检查是否是文件夹
        if not os.path.isdir(class_folder):
            continue

        # 获取类别名称
        class_id = class_name.encode('ascii')

        # 创建一个事务
        with env.begin(write=True) as txn:
            # 遍历每个图像文件
            for image_name in tqdm(os.listdir(class_folder), desc=class_name):
                image_path = os.path.join(class_folder, image_name)
                with open(image_path, 'rb') as f:
                    image_bytes = f.read()

                # 构建LMDB键
                key = f"{class_name}_{image_name}".encode('ascii')

                # 将图像数据存储到LMDB数据库
                txn.put(key, image_bytes)

    # 关闭 LMDB 数据库
    env.close()

# 指定输入文件夹（包含不同类别的图像）和输出的 LMDB 文件夹
input_folder = 'copy'  # 根据您的目录结构调整
output_lmdb = 'lmdb_dataset'  # 输出 LMDB 数据库文件夹

# 创建 LMDB 数据集
create_lmdb_dataset(input_folder, output_lmdb)
