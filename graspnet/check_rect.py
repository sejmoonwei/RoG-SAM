import os

def find_incomplete_scenes(base_dir):
    incomplete_scenes = []

    # 获取所有以 'scene_' 开头的文件夹
    scenes = [d for d in os.listdir(base_dir)
              if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('scene_')]

    for scene in sorted(scenes):
        scene_path = os.path.join(base_dir, scene)
        incomplete = False  # 标记当前场景是否数据不完整

        for sensor in ['kinect', 'realsense']:
            sensor_path = os.path.join(scene_path, sensor)
            if not os.path.exists(sensor_path):
                print(f"场景 '{scene}' 缺少传感器文件夹 '{sensor}'。")
                incomplete = True
                continue  # 继续检查下一个传感器

            # 生成期望的文件名集合
            expected_files = {f"{i:04d}.npy" for i in range(256)}
            # 获取实际存在的文件名集合
            actual_files = set(os.listdir(sensor_path))

            missing_files = expected_files - actual_files
            if missing_files:
                print(f"场景 '{scene}' 的 '{sensor}' 中缺少文件：{sorted(missing_files)}")
                incomplete = True

        if incomplete:
            incomplete_scenes.append(scene)

    # 打印所有数据不完整的场景
    print("\n数据不完整的场景：")
    for scene in incomplete_scenes:
        print(scene)

if __name__ == '__main__':
    base_dir = '/data/myp/grasp/ori/graspnet/rect_labels'  # 请将此处替换为您的主文件夹路径
    find_incomplete_scenes(base_dir)
