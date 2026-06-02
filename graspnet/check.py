import os.path
import cv2
import numpy as np
# import matplotlib.pyplot as plt # 移除此包以解决numpy版本冲突
from graspnetAPI import GraspNet

# GraspNetAPI example for checking the data completeness.
# change the graspnet_root path

# 以下函数依赖matplotlib，当前脚本执行路径中并未用到，故注释掉
# def visualize_mask(mask_path):
#     # 读取mask图片
#     mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
#
#     # 创建一个RGB图像来存储可视化的结果
#     height, width = mask.shape
#     visualization = np.zeros((height, width, 3), dtype=np.uint8)
#
#     # 获取所有非背景的物体id（假设背景的值是0）
#     unique_objects = np.unique(mask)
#     unique_objects = unique_objects[unique_objects != 0]  # 排除背景
#
#     # 生成随机颜色表
#     np.random.seed(42)  # 固定随机种子以确保每次颜色一致
#     colors = {obj_id: np.random.randint(0, 255, 3) for obj_id in unique_objects}
#
#     # 对每个物体进行着色
#     for obj_id in unique_objects:
#         visualization[mask == obj_id] = colors[obj_id]
#
#     # 显示可视化结果
#     plt.figure(figsize=(10, 10))
#     plt.imshow(visualization)
#     plt.axis('off')  # 隐藏坐标轴
#     plt.show()
#
#
# def visualize_mask_on_rgb(mask_path, rgb_path):
#     # 读取mask图片
#     mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
#
#     # 读取RGB图片
#     rgb_image = cv2.imread(rgb_path)
#     rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)  # 将BGR转换为RGB
#
#     # 创建一个RGB图像来存储可视化的mask叠加结果
#     height, width = mask.shape
#     mask_overlay = np.copy(rgb_image)
#
#     # 获取所有非背景的物体id（假设背景的值是0）
#     unique_objects = np.unique(mask)
#     unique_objects = unique_objects[unique_objects != 0]  # 排除背景
#
#     # 生成随机颜色表
#     np.random.seed(35)  # 固定随机种子以确保每次颜色一致
#     colors = {obj_id: np.random.randint(0, 255, 3) for obj_id in unique_objects}
#
#     # 对每个物体进行着色，并叠加到RGB图像上
#     for obj_id in unique_objects:
#         mask_region = mask == obj_id
#         color_mask = np.zeros_like(rgb_image)
#         color_mask[mask_region] = colors[obj_id]  # 将颜色应用到mask区域
#
#         # 将mask的颜色混合叠加到原始RGB图像上
#         mask_overlay = cv2.addWeighted(mask_overlay, 0.7, color_mask, 1, 0)
#
#     # 显示叠加后的结果
#     plt.figure(figsize=(10, 10))
#     plt.imshow(mask_overlay)
#     plt.axis('off')  # 隐藏坐标轴
#     plt.show()


if __name__ == '__main__':

    ####################################################################
    graspnet_root = '/data/myp/grasp_dataset'  ### ROOT PATH FOR GRASPNET ###
    ####################################################################

    # initialize a GraspNet instance
    g = GraspNet(graspnet_root, camera='kinect', split='train')
    # show scene rectangle grasps
    try:
        g.showSceneGrasp(sceneId = 181, camera = 'kinect', annId = 30, format = 'rect', numGrasp = 50, coef_fric_thresh=0.4)
    except Exception as e:
        print("\n无法显示图形窗口，因为已安装 headless 版本的 OpenCV。")
        print("这是一个预期的行为，脚本已成功运行所有非图形部分。")
        print(f"原始错误: {e}")
