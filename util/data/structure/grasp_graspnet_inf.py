# -*- coding: UTF-8 -*-
"""===============================================
@Author : wangdx
@Date   : 2020/9/1 21:37
==============================================="""
import mmcv
import numpy as np
import cv2
import math
import torch
import scipy.io as scio
# from mmcv.ops.roi_align import roi_align
import random
def calcAngle2(angle):
    """
    根据给定的angle计算与之反向的angle
    :param angle: 弧度
    :return: 弧度
    """
    return angle + math.pi - int((angle + math.pi) // (2 * math.pi)) * 2 * math.pi

def drawGrasp(img, label, offset, interval=20):
    """
    绘制抓取标签
        label: (4, h, w)
        offset: (row, col)
    :return:
    """

    grasp_confidence = label[0, :, :]   # 抓取置信度
    grasp_mode = label[1, :, :]         # 抓取模式 0-无约束抓取 1-单向抓取 2-对称抓取
    grasp_angle = label[2, :, :]        # 抓取角
    grasp_width = label[3, :, :]        # 抓取宽度

    # 绘制抓取点
    grasp_point_rows, grasp_point_cols = np.where(grasp_confidence > 0)
    grasp_point_rows = grasp_point_rows + offset[0]
    grasp_point_cols = grasp_point_cols + offset[1]
    img[grasp_point_rows, grasp_point_cols, :] = [0, 255, 0]

    # 绘制抓取角和抓取宽度
    n = 0
    for i, _ in enumerate(grasp_point_rows):
        n += 1
        if n % interval != 0:
            continue
        row, col = grasp_point_rows[i] - offset[0], grasp_point_cols[i] - offset[1]
        width = grasp_width[row, col] * 150. / 2
        angle = grasp_angle[row, col]   # 弧度
        mode = grasp_mode[row, col]

        row, col = row + offset[0], col + offset[1]

        if mode == 0.:      # 无约束抓取
            cv2.circle(img, (col, row), int(width), (255, 245, 0), 1)

        elif mode == 1.:    # 单向抓取
            k = math.tan(angle)

            if k == 0:
                dx = width
                dy = 0
            else:
                dx = k / abs(k) * width / pow(k ** 2 + 1, 0.5)
                dy = k * dx

            if angle < math.pi:
                cv2.line(img, (col, row), (int(col + dx), int(row - dy)), (255, 245, 0), 1)
            else:
                cv2.line(img, (col, row), (int(col - dx), int(row + dy)), (255, 245, 0), 1)

        elif mode == 2.:    # 对称抓取
            angle2 = calcAngle2(angle)
            k = math.tan(angle)

            if k == 0:
                dx = width
                dy = 0
            else:
                dx = k / abs(k) * width / pow(k ** 2 + 1, 0.5)
                dy = k * dx

            if angle < math.pi:
                cv2.line(img, (col, row), (int(col + dx), int(row - dy)), (255, 245, 0), 1)
            else:
                cv2.line(img, (col, row), (int(col - dx), int(row + dy)), (255, 245, 0), 1)

            if angle2 < math.pi:
                cv2.line(img, (col, row), (int(col + dx), int(row - dy)), (255, 245, 0), 1)
            else:
                cv2.line(img, (col, row), (int(col - dx), int(row + dy)), (255, 245, 0), 1)

        else:
            raise ValueError

    return img

def drawGrasp1(img, grasp):
    """
    绘制抓取标签
        grasp: [row, col, angle, width]
    :return:
    """

    row, col = int(grasp[0]), int(grasp[1])
    cv2.circle(img, (int(grasp[1]), int(grasp[0])), 2, (0, 255, 0), -1)
    angle = grasp[2]   # 弧度
    width = grasp[3] / 2

    k = math.tan(angle)

    if k == 0:
        dx = width
        dy = 0
    else:
        dx = k / abs(k) * width / pow(k ** 2 + 1, 0.5)
        dy = k * dx

    if angle < math.pi:
        cv2.line(img, (col, row), (int(col + dx), int(row - dy)), (255, 245, 0), 1)
    else:
        cv2.line(img, (col, row), (int(col - dx), int(row + dy)), (255, 245, 0), 1)


    return img


def imrotate(img,
             angle,
             center=None,
             scale=1.0,
             flag=cv2.INTER_NEAREST,
             border_value=0,
             auto_bound=False):
    """Rotate an image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees, positive values mean
            clockwise rotation.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If not specified, the center of the image will be
            used.
        scale (float): Isotropic scale factor.
        border_value (int): Border value.
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image.

    Returns:
        ndarray: The rotated image.
    """
    if center is not None and auto_bound:
        raise ValueError('`auto_bound` conflicts with `center`')
    h, w = img.shape[:2]
    if center is None:
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
    assert isinstance(center, tuple)

    matrix = cv2.getRotationMatrix2D(center, -angle, scale)
    if auto_bound:
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = h * sin + w * cos
        new_h = h * cos + w * sin
        matrix[0, 2] += (new_w - w) * 0.5
        matrix[1, 2] += (new_h - h) * 0.5
        w = int(np.round(new_w))
        h = int(np.round(new_h))
    rotated = cv2.warpAffine(img, matrix, (w, h), flags=flag, borderValue=border_value)
    return rotated

# 函数计算中心点
def calculate_center(points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    center_x = np.mean(x_coords)
    center_y = np.mean(y_coords)
    return (center_x, center_y)

# 函数计算长度（假设第一个和第二个点为长边）
def calculate_length(points):
    p1, p2 = points[1], points[2]
    length = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return length

# 函数计算抓取方向与y轴的夹角
def calculate_angle(points):
    p1, p2 = points[0], points[1]
    delta_x = p2[0] - p1[0]
    delta_y = p2[1] - p1[1]
    angle = math.atan2(delta_y, delta_x)
    # 将角度转换为与y轴的夹角
    angle_with_y = angle - (math.pi / 2)
    # 将角度转换为度数
    angle_with_y_degrees = math.degrees(angle_with_y)
    return angle_with_y_degrees


def get_grasp(grasp_rects):
    # 收集所有抓取表示
    grasp_representations = []
    for rect in grasp_rects:
        center = calculate_center(rect)
        length = calculate_length(rect)
        angle = calculate_angle(rect)
        grasp_representations.append((center[0], center[1], angle, length))

    return grasp_representations


def compute_grasp_rectangles(rect_np):
    rectangles = []

    for grasp in rect_np:
        xcenter, ycenter, xright, yright, height, _, _ = grasp

        # 计算矩形的宽度
        width = 2 * np.sqrt((xright - xcenter) ** 2 + (yright - ycenter) ** 2)

        # 计算旋转角度
        angle = np.arctan2(yright - ycenter, xright - xcenter)

        # 计算未旋转前的四个顶点的相对坐标
        dx = width / 2
        dy = height / 2
        corners = [
            (-dx, dy),  # 左下
            (-dx, -dy),  # 左上
            (dx, -dy),  # 右上
            (dx, dy),  # 右下
        ]

        # 旋转并平移四个顶点到实际位置
        rotated_corners = [
            (xcenter + x * np.cos(angle) - y * np.sin(angle),
             ycenter + x * np.sin(angle) + y * np.cos(angle))
            for (x, y) in corners
        ]

        rectangles.append(rotated_corners)

    return rectangles


def gen_mat(label_file,H=720, W=1280):
    # label_file = '/data/myp/grasp_dataset/scenes/scene_0000/kinect/rect/0000.npy'
    rect_np = np.load(label_file) #8056,7
    boxes_list = compute_grasp_rectangles(rect_np)
    # get pos, cls, nagle, width
    grasp_label = get_grasp(boxes_list)
    label_mat = np.zeros((4, H, W), dtype=np.float64) #pos cls angle width

    for label in grasp_label:
        row = int(float(label[1]))
        col = int(float(label[0]))

        size = 2 #train 3 test 1
        startx = max(0,row-size)
        endx = min(1280, row+size+1)
        starty = max(0,col-size)
        endy = min(720, col+size+1)
        label_mat[0,startx:endx,starty:endy] = 1  # 设置抓取点

        label_mat[1, startx:endx, starty:endy] = 3. # class
        # label_mat[1, row, col] = 3.
        theta = float(label[2])  #
        angle = - theta / 180.0 * np.pi
        if angle < -3.14 or angle > 6.28:
            raise ValueError('invalid angle:{}'.format(angle))
        elif angle < 0:
            angle += 3.14
        elif angle > 3.14:
            angle -= 3.14


        label_mat[2,startx:endx,starty:endy] = angle  # 设置抓取点 #same pos with multi angle   not considered yet

        # label_mat[2, row, col] = angle
        label_mat[3,startx:endx,starty:endy] = float(label[-1])  # / 200.  # 设置抓取宽度
    # 0:position 1:label 2:angle 3:w
    #---------------vis------------------#
    # png_file = label_file.replace("rect", "rgb").replace(".npy", ".png")
    # im = cv2.imread(png_file)
    # mask = torch.from_numpy(label_mat[0]==1)
    # indices = torch.nonzero(mask, as_tuple=False)
    # list_of_coord = [tuple(idx) for idx in indices ]
    # n = 0
    # for coord in list_of_coord:
    #     n += 1
    #     if n%50==0:
    #         y,x = coord
    #         x = int(x)
    #         y = int(y)
    #         w = float(label_mat[3][y,x]) # width
    #         angle1 = label_mat[2][y,x]
    #         if angle1 < 3.14:
    #             angle2 = angle + 3.14
    #         else:
    #             angle2 = angle - 3.14
    #         k = math.tan(angle1)
    #
    #         if k == 0:
    #             dx = w
    #             dy = 0
    #         else:
    #             dx = k / abs(k) * w / pow(k ** 2 + 1, 0.5)
    #             dy = k * dx
    #
    #         if angle1 < math.pi:
    #             cv2.line(im, (x, y), (int(x + dx), int(y - dy)), (0, 255, 0), 1)
    #         else:
    #             cv2.line(im, (x, y), (int(x - dx), int(y + dy)), (0, 255, 0), 1)
    #
    #         if angle2 < math.pi:
    #             cv2.line(im, (x, y), (int(x + dx), int(y - dy)), (0, 255, 0), 1)
    #         else:
    #             cv2.line(im, (x, y), (int(x - dx), int(y + dy)), (0, 255, 0), 1)
    #
    #         cv2.circle(im, (x, y), 1, (255, 0, 0), 1)
    #
    # cv2.imshow('img',im)
    # cv2.waitKey(0)
    # cv2.destroyALLWindows()
    # #-------------------vis---------------------------------#
    return label_mat




class GraspMat:
    def __init__(self, rect_path, ins_mask_path):
        #(4,1280,720)
        grasp_list = []
        box_mask_list = []
        grasp = gen_mat(rect_path) #4,720,1280
        instance_msk_ori = cv2.imread(ins_mask_path, cv2.IMREAD_UNCHANGED)
        ins_value = np.unique(instance_msk_ori) #[0,1,2]
        ins_value = [x for x in ins_value if x != 0]#[1,2]
        for value in ins_value:
            instance_msk = instance_msk_ori.copy()
            positions = (instance_msk == value) #720,1280 array
            expanded_position = np.expand_dims(positions,axis=0) #1,720,1280
            expanded_position = np.repeat(expanded_position, grasp.shape[0], axis=0) #4,720,1280
            grasp_ins = np.where(expanded_position, grasp, 0) #4,720,1280
            grasp_list.append(self.preprocess_image(grasp_ins))
            instance_msk[instance_msk != value] = 0  # 720,1280
            instance_msk = np.expand_dims(instance_msk, axis=0)  # 1,720,1280
            box_mask_list.append(self.preprocess_image(instance_msk))
        self.grasps = grasp_list #4,512,512
        self.box_masks = box_mask_list #1,512,512

    @staticmethod
    def preprocess_image(img, target_size=(512, 512)):
        c, h, w = img.shape  # 假设输入为 (通道, 高度, 宽度)
        target_w, target_h = target_size

        # 初始化处理后的图像数组
        processed_img = np.zeros((c, target_h, target_w), dtype=img.dtype)

        for i in range(c):
            # 计算调整比例
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            # 调整图像大小
            resized_img = cv2.resize(img[i], (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            # 计算填充
            delta_w = target_w - new_w
            delta_h = target_h - new_h
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)

            # 填充图像
            padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

            # 将处理后的单通道图像放回多通道数组中
            processed_img[i] = padded_img

        return processed_img

    def height(self):
        return self.grasp.shape[1]

    def width(self):
        return self.grasp.shape[2]

    def crop(self, bbox):
        """
        裁剪 self.grasp

        args:
            bbox: list(x1, y1, x2, y2)
        """
        for i in range(len(self.grasps)):
            self.grasps[i] = self.grasps[i][:, bbox[1]:bbox[3], bbox[0]:bbox[2]]
            self.box_masks[i] = self.box_masks[i][:, bbox[1]:bbox[3], bbox[0]:bbox[2]]






    def _flipAngle(self, angle_mat, confidence_mat):
        """
        水平翻转angle
        Args:
            angle_mat: (h, w) 弧度
            confidence_mat: (h, w) 抓取置信度
        Returns:
        """
        # 全部水平翻转
        angle_out = (angle_mat // math.pi) * 2 * math.pi + math.pi - angle_mat
        # 将非抓取区域的抓取角置0
        angle_out = angle_out * confidence_mat
        # 所有角度对2π求余
        angle_out = angle_out % (2 * math.pi)

        return angle_out


    def _decode(self, mat, angle_cls):
        """
        解析 grasp_mat
        Args:
            mat: np.ndarray (4, h, w)
            angle_cls: 抓取角类别数，36/72/120

        Returns:
                (1 + angle_cls + 1, h, w)  float
        """
        h, w = mat.shape[1:]
        grasp_confidence = mat[0, :, :]
        grasp_mode = mat[1, :, :]
        grasp_angle = mat[2, :, :]
        grasp_width = mat[3, :, :]

        angle_mat = np.zeros((angle_cls, h, w), dtype=np.float64)     # -1:不属于抓取点
        grasp_point = np.where(grasp_confidence > 0)
        for i, _ in enumerate(grasp_point[0]):
            row, col = grasp_point[0][i], grasp_point[1][i]
            angle = grasp_angle[row, col]  # 弧度
            mode = grasp_mode[row, col]

            angle_mat[-1, row, col] = 0.

            if mode == 1.:  # 无约束抓取
                angle_mat[:, row, col] = 1.
            elif mode == 2.:  # 单向抓取
                angle1 = int(angle / (2 * np.pi) * angle_cls)  # 弧度转类别
                angle_mat[angle1, row, col] = 1.
            elif mode == 3.:  # 对称抓取
                angle1 = int(angle / (2 * np.pi) * angle_cls)  # 弧度转类别
                angle2 = angle + np.pi - int((angle + np.pi) // (2 * np.pi)) * 2 * np.pi
                angle2 = int(angle2 / (2 * np.pi) * angle_cls)  # 弧度转类别
                angle_mat[angle1, row, col] = 1.
                angle_mat[angle2, row, col] = 1.
            else:
                print('mode error')
                raise ValueError

        grasp_confidence = np.expand_dims(grasp_confidence, axis=0)
        grasp_width = np.expand_dims(grasp_width, axis=0)
        ret_mat = np.zeros(shape=(122, 500, 500))   # 122
        ret_mat[0, :, :] = grasp_confidence
        ret_mat[1:-1, :, :] = angle_mat
        ret_mat[-1, :, :] = grasp_width / 100. #200 to 100

        # print(np.unique(grasp_width))

        return ret_mat


    def decode(self, angle_cls):
        """
        (4, H, W) -> (angle_cls+2, H, W)
        """
        for i in range(len(self.grasps)):
            self.grasps[i] = self._decode(self.grasps[i], angle_cls)
