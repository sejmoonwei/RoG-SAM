# -*- coding: UTF-8 -*-
"""===============================================
@Author : wangdx
@Date   : 2020/9/1 21:37
==============================================="""
import cv2
import mmcv
import math
import numpy as np

class Image:
    """
    """
    def __init__(self, file):

        img = cv2.imread(file)
        # self.img = img
        self.img = self.preprocess_image(img)
        # cv2.imshow('crop',self.img)
        # cv2.waitKey(0)
        # cv2.destroyALLWindows()

    def height(self):
        return self.img.shape[0]

    def width(self):
        return self.img.shape[1]

    def preprocess_image(self,img, target_size=(512, 512)):
        h, w = img.shape[:2]
        target_w, target_h = target_size

        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized_img = cv2.resize(img, (new_w, new_h))

        delta_w = target_w - new_w
        delta_h = target_h - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        color = [0, 0, 0]
        padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        return padded_img

    def crop(self, size, dist=-1):
        """

        args:
            size: int
            dist: int
        return:
            crop_x1, ...
        """
        if dist > 0:
            x_offset = np.random.randint(-1 * dist, dist)
            y_offset = np.random.randint(-1 * dist, dist)
        else:
            x_offset = 0
            y_offset = 0

        crop_x1 = int((self.width() - size) / 2 + x_offset) #width 512  size500  6
        crop_y1 = int((self.height() - size) / 2 + y_offset)
        crop_x2 = crop_x1 + size
        crop_y2 = crop_y1 + size

        self.img = self.img[crop_y1:crop_y2, crop_x1:crop_x2, :]

        return crop_x1, crop_y1, crop_x2, crop_y2


    def rescale(self, scale, interpolation='bilinear'):
        self.img = mmcv.imrescale(self.img, scale, interpolation=interpolation)

    def rotate(self, rota):
        """
        """
        self.img = mmcv.imrotate(self.img, rota, border_value=(0, 0, 0))
        # cv2.imshow('crop',self.img)
        # cv2.waitKey(0)
        # cv2.destroyALLWindows()


    def flip(self, flip_direction='horizontal'):
        """See :func:`BaseInstanceMasks.flip`."""
        assert flip_direction in ('horizontal', 'vertical')

        self.img = mmcv.imflip(self.img, direction=flip_direction)


    def _Hue(self, img, bHue, gHue, rHue):
        imgB = img[:, :, 0]
        imgG = img[:, :, 1]
        imgR = img[:, :, 2]

        bAve = cv2.mean(imgB)[0] + bHue
        gAve = cv2.mean(imgG)[0] + gHue
        rAve = cv2.mean(imgR)[0] + rHue
        aveGray = (int)(bAve + gAve + rAve) / 3

        if bAve == 0:
            pass

        bCoef = aveGray / bAve
        gCoef = aveGray / gAve
        rCoef = aveGray / rAve

        imgB = np.expand_dims(np.floor((imgB * bCoef)), axis=2)
        imgG = np.expand_dims(np.floor((imgG * gCoef)), axis=2)
        imgR = np.expand_dims(np.floor((imgR * rCoef)), axis=2)

        dst = np.concatenate((imgB, imgG, imgR), axis=2)
        dst = np.clip(dst, 0, 255).astype(np.uint8)

        return dst


    def color(self, hue=10):
        """
        """

        hue = np.random.uniform(-1 * hue, hue)

        if hue == 0:
            if np.random.rand() < 0.5:
                self.img = self._Hue(self.img, hue, hue, hue)
        else:
            bHue = hue if hue > 0 else 0
            gHue = abs(hue)
            rHue = -1 * hue if hue < 0 else 0
            self.img = self._Hue(self.img, bHue, gHue, rHue)


        bright = np.random.uniform(-40, 10)
        imgZero = np.zeros(self.img.shape, self.img.dtype)
        self.img = cv2.addWeighted(self.img, 1, imgZero, 2, bright)


    def nomalise(self):
        self.img = self.img.astype(np.float32) / 255.0
        self.img -= self.img.mean()

