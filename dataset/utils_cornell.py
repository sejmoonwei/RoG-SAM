import torch
import mmcv


def rescale_bbox(bbox, scale):
    """
    Rescale bounding box coordinates according to the scale factor.

    Args:
        bbox (tensor): Bounding box coordinates in the format [x1, y1, x2, y2].
        original_size (tuple): Original image size (width, height).
        scale (tuple or float): Scale factor. If a tuple, it should be (scale_x, scale_y).
                                If a float, it will be used for both width and height.

    Returns:
        tensor: Rescaled bounding box coordinates.
    """

    if isinstance(scale, tuple):
        scale_x, scale_y = scale
    else:
        scale_x = scale_y = scale

    bbox_rescaled = bbox.clone()
    bbox_rescaled[0] = bbox[0] * scale_x
    bbox_rescaled[1] = bbox[1] * scale_y
    bbox_rescaled[2] = bbox[2] * scale_x
    bbox_rescaled[3] = bbox[3] * scale_y

    return bbox_rescaled


def crop(bbox, crop_coords):
    """
    Crop bounding box coordinates according to the crop area.

    Args:
        bbox (tensor): Bounding box coordinates in the format [x1, y1, x2, y2].
        crop_coords (tuple): Crop area coordinates (crop_x1, crop_y1, crop_x2, crop_y2).

    Returns:
        tensor: Cropped bounding box coordinates.
    """
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_coords

    bbox_cropped = bbox.clone()
    bbox_cropped[0] = bbox[0] - crop_x1
    bbox_cropped[1] = bbox[1] - crop_y1
    bbox_cropped[2] = bbox[2] - crop_x1
    bbox_cropped[3] = bbox[3] - crop_y1

    return bbox_cropped


def flip_bbox(bbox, img_size, flip_direction='horizontal'):
    width, height = img_size

    bbox_flipped = bbox.clone()
    if flip_direction == 'horizontal':
        bbox_flipped[0] = width - bbox[2]
        bbox_flipped[2] = width - bbox[0]
    elif flip_direction == 'vertical':
        bbox_flipped[1] = height - bbox[3]
        bbox_flipped[3] = height - bbox[1]

    return bbox_flipped

