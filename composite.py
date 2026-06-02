import matplotlib.pyplot as plt
import numpy as np
import cv2

def combine_and_overlay_heatmaps(image_path, heatmap_paths, output_path, cmap='jet', vmin=0.05, vmax=0.85):
    # 读取 RGB 图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将 BGR 转换为 RGB

    # 初始化一个与热度图相同大小的空白数组
    combined_heatmap = None

    # 逐个读取并累加热度图
    for idx, heatmap_path in enumerate(heatmap_paths):
        heatmap = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图
        heatmap = heatmap.astype(np.float32)  # 确保是浮点类型

        # 对 heatmap5 进行放大处理
        if idx == 5:
            maxva = heatmap.max()
            thershold = 0.9 * maxva
            heatmap[heatmap < thershold] *= 0.9
            # heatmap = heatmap ** 1.3
            heatmap *= 2.4

        if combined_heatmap is None:
            combined_heatmap = heatmap
        else:
            combined_heatmap += heatmap

    # 归一化合成的热度图到0-255
    combined_heatmap = np.clip(combined_heatmap, 0, 255).astype(np.uint8)

    # 创建一个新的 figure
    plt.figure(figsize=(10, 10))

    # 显示 RGB 图像
    plt.imshow(image, alpha=1)

    # 将合成的热度图叠加到图像上
    plt.imshow(combined_heatmap, cmap=cmap, alpha=0.5, vmin=vmin*255, vmax=vmax*255)

    # 隐藏坐标轴
    plt.axis('off')

    # 保存最终图像
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()


# 示例用法
if __name__ == "__main__":
    # 图片路径
    image_path = '/data1/samgrasp/dataset/test/show_real/rgb1_2.png'
    heatmap_paths = [f'/data1/samgrasp/dataset/test/show_real/rawheatmap{i}.png' for i in range(10)]
    output_path = './png/output_image.png'

    # 叠加热度图到RGB图像
    combine_and_overlay_heatmaps(image_path, heatmap_paths, output_path)











