import os
import shutil

# 文件路径
train_txt = '/data1/samgrasp/dataset/cornell_affga/train-test/train-test-cornell/image-wise-train.txt'
test_txt = '/data1/samgrasp/dataset/cornell_affga/train-test/train-test-cornell/image-wise-test.txt'
dataset_dir = '/data1/samgrasp/dataset/cornell_adapt/Training'

# 目标文件夹
train_output_dir = 'train_data'
test_output_dir = 'test_data'

# 确保目标文件夹存在
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(test_output_dir, exist_ok=True)

# 函数：根据文件列表将文件复制到目标文件夹
def copy_files(file_list, output_dir):
    for file_name in file_list:
        for suffix in ['grasp.mat', 'Label.txt', 'r.png']:
            source_file = os.path.join(dataset_dir, f'{file_name}{suffix}')
            if os.path.exists(source_file):
                shutil.copy(source_file, output_dir)
            else:
                print(f'文件 {source_file} 不存在')

# 读取train和test txt文件，获取样本名列表
with open(train_txt, 'r') as f:
    train_samples = f.read().splitlines()

with open(test_txt, 'r') as f:
    test_samples = f.read().splitlines()

# 复制文件到各自的目标文件夹
copy_files(train_samples, train_output_dir)
copy_files(test_samples, test_output_dir)

print("数据集分拣完成。")

