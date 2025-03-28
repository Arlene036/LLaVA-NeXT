import json
from collections import defaultdict
import random

# 读取JSON文件
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# 保存JSON文件
def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def extract_validation_set(data, val_ratio=0.1):
    # 按scene分组
    scene_groups = defaultdict(list)
    
    # 遍历所有样本，根据video字段分组
    for item in data:
        scene = item['video'].split('/')[0]  # 获取scene名称，如"scene0010_00"
        scene_groups[scene].append(item)
    
    # 随机选择10%的scene作为验证集
    all_scenes = list(scene_groups.keys())
    num_val_scenes = max(1, int(len(all_scenes) * val_ratio))
    val_scenes = set(random.sample(all_scenes, num_val_scenes))
    
    # 分离验证集数据
    val_data = []
    train_data = []
    
    for scene, items in scene_groups.items():
        if scene in val_scenes:
            # 验证集场景
            for item in items:
                item['metadata']['split'] = 'val'
                val_data.append(item)
        else:
            # 训练集场景
            for item in items:
                item['metadata']['split'] = 'train'
                train_data.append(item)
    
    # 按sample_id排序
    val_data.sort(key=lambda x: x['sample_id'])
    train_data.sort(key=lambda x: x['sample_id'])
    
    return train_data, val_data

def main():
    # 设置随机种子以确保结果可复现
    random.seed(42)
    
    # 输入和输出文件路径
    input_file = 'scanqa_map_small.json'
    train_output_file = 'scanqa_map_small_train.json'
    val_output_file = 'scanqa_map_small_val.json'
    
    # 读取数据
    data = load_json(input_file)
    
    # 分离训练集和验证集
    train_data, val_data = extract_validation_set(data)
    
    # 保存训练集和验证集
    save_json(train_data, train_output_file)
    save_json(val_data, val_output_file)
    
    # 打印统计信息
    print(f"处理完成！")
    print(f"训练集样本数: {len(train_data)}")
    print(f"验证集样本数: {len(val_data)}")
    print(f"验证集场景数: {len(set(item['video'].split('/')[0] for item in val_data))}")

if __name__ == '__main__':
    main()