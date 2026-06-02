import cv2
import math
from skimage.draw import polygon
from skimage.feature import peak_local_max
import torch.nn.functional as F
import numpy as np


def length(pt1, pt2):
    """
    :param pt1: [row, col]
    :param pt2: [row, col]
    :return:
    """
    return pow(pow(pt1[0] - pt2[0], 2) + pow(pt1[1] - pt2[1], 2), 0.5)


def diff(k, label):
    """
    """
    clss = np.argwhere(label == 1)
    clss = np.reshape(clss, newshape=(clss.shape[0],))
    clss_list = list(clss)
    min_diff = label.shape[0] + 1

    for cls in clss_list:
        min_diff = min(min_diff, abs(cls - k))

    return min_diff, clss_list


def arg_thresh(array, thresh):
    """
    :return: array shape=(n, 2)
    """
    res = np.where(array > thresh)
    rows = np.reshape(res[0], (-1, 1))
    cols = np.reshape(res[1], (-1, 1))
    locs = np.hstack((rows, cols))
    for i in range(locs.shape[0]):
        for j in range(locs.shape[0])[i+1:]:
            if array[locs[i, 0], locs[i, 1]] < array[locs[j, 0], locs[j, 1]]:
                locs[[i, j], :] = locs[[j, i], :]

    return locs



def rect_loc(row, col, angle, height, bottom):
    """
    :return:
    """
    xo = np.cos(angle)
    yo = np.sin(angle)

    y1 = row + height / 2 * yo
    x1 = col - height / 2 * xo
    y2 = row - height / 2 * yo
    x2 = col + height / 2 * xo

    return np.array(
        [
         [y1 - bottom/2 * xo, x1 - bottom/2 * yo],
         [y2 - bottom/2 * xo, x2 - bottom/2 * yo],
         [y2 + bottom/2 * xo, x2 + bottom/2 * yo],
         [y1 + bottom/2 * xo, x1 + bottom/2 * yo],
         ]
    ).astype(np.int32)



def polygon_iou(polygon_1, polygon_2):
    """
    :param polygon_1: [[row1, col1], [row2, col2], ...]
    :return:
    """
    rr1, cc1 = polygon(polygon_2[:, 0], polygon_2[:, 1])
    rr2, cc2 = polygon(polygon_1[:, 0], polygon_1[:, 1])

    try:
        r_max = max(rr1.max(), rr2.max()) + 1
        c_max = max(cc1.max(), cc2.max()) + 1
    except ValueError:
        return 0

    canvas = np.zeros((r_max, c_max))
    canvas[rr1, cc1] += 1
    canvas[rr2, cc2] += 1
    union = np.sum(canvas > 0)
    if union == 0:
        return 0
    intersection = np.sum(canvas == 2)
    return intersection / union


def calcAngle2(angle):
    """
    """
    return angle + math.pi - int((angle + math.pi) // (2 * math.pi)) * 2 * math.pi



def evaluation(able_out, angle_out, width_out, target, angle_k=120, eval_mode='peak', angle_th=30, iou_th=0.25, bottom=30, desc='1'):
    """
    :param target: (1, 2+angle_k, 320, 320)
    :param desc:
    :return:
        2、IOU>0.25
    """



    rows = able_out.shape[0]    #
    cols = able_out.shape[1]    #

    able_target = target[0, 0, :, :].cpu().numpy()          # (256, 256)
    angles_target = target[0, 1:-1, :, :].cpu().numpy()     # (angle_k, 256, 256)
    width_target = target[0, -1, :, :].cpu().numpy() * 100.        # (256, 256)

    if able_target.max() == 0:
        return None

    threshold_abs = 0.1 # 0.3 to 0.1
    if eval_mode == 'peak':
        min_distance = 10 #30 to 10
        pred_pts = peak_local_max(able_out, num_peaks = 5, min_distance=min_distance, threshold_abs=threshold_abs)
    elif eval_mode == 'all':
        pred_pts = arg_thresh(able_out, threshold_abs)
        while pred_pts.shape[0] > 50:
            threshold_abs += 0.05
            pred_pts = arg_thresh(able_out, threshold_abs)
            if threshold_abs >= 0.95:
                break
    elif eval_mode == 'max':
        loc = np.argmax(able_out)
        row = loc // able_out.shape[0]
        col = loc % able_out.shape[0]
        pred_pts = np.array([[row, col]])

    else:
        raise ValueError(f"Invalid evaluation mode: {eval_mode}. Choose from ['peak', 'all', 'max'].")

    if desc != '1':
        return 0

    thresh = 30
    for idx in range(pred_pts.shape[0]):
        row_pred, col_pred = pred_pts[idx]
        angle_pred_cls = angle_out[row_pred, col_pred]
        width_pred = width_out[row_pred, col_pred]
        angle_pred = angle_pred_cls / angle_k * 2 * math.pi

        rect_pred = rect_loc(row_pred, col_pred, angle_pred, width_pred, bottom)

        for row in range(rows):
            for col in range(cols):
                if able_target[row, col] != 1.:
                    continue

                if length([row, col], [row_pred, col_pred]) > thresh:
                    continue
                mon_pt = True

                angle_label = angles_target[:, row, col]
                angle_diff, angle_label = diff(angle_pred_cls, angle_label)
                angle_diff = angle_diff / angle_k * 360.

                if angle_diff > angle_th:
                    continue

                width_label = width_target[row, col]
                for angle in angle_label:
                    angle = angle / angle_k * 2 * math.pi
                    rect_label = rect_loc(row, col, angle, width_label, bottom)
                    iou = polygon_iou(rect_label, rect_pred)
                    if iou >= iou_th:
                        return True

        return False


'''
python accuracy.py -net sam -mod sam_adpt -exp_name ocid_eval -image_size 512 -out_size 256 -b 1 -dataset OCID -gpu_device 0 -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -multimask 122 -prompt click box -data_path /path/to/OCID_grasp -pretrain /path/to/ocid_checkpoint.pth
'''
