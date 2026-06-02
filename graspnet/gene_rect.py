import os.path
from audioop import error

import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from graspnetAPI import GraspNet
from tqdm import tqdm

NUM_PROCESS = 24

def generate_scene_rectangle_grasp(sceneId, dump_folder, camera):
    g = GraspNet(graspnet_root, camera=camera, split='all')
    objIds = g.getObjIds(sceneIds = sceneId)
    grasp_labels = g.loadGraspLabels(objIds)
    collision_labels = g.loadCollisionLabels(sceneIds = sceneId)
    scene_dir = os.path.join(dump_folder,'scene_%04d' % sceneId)
    if not os.path.exists(scene_dir):
        os.mkdir(scene_dir)
    camera_dir = os.path.join(scene_dir, camera)
    if not os.path.exists(camera_dir):
        os.mkdir(camera_dir)
    for annId in tqdm(range(256), 'Scene:{}, Camera:{}'.format(sceneId, camera)):
        _6d_grasp = g.loadGrasp(sceneId = sceneId, annId = annId, format = '6d', camera = camera, grasp_labels = grasp_labels, collision_labels = collision_labels, fric_coef_thresh = 1.0)
        rect_grasp_group = _6d_grasp.to_rect_grasp_group(camera)
        rect_grasp_group.save_npy(os.path.join(camera_dir, '%04d.npy' % annId))


if __name__ == '__main__':
    graspnet_root = '/data/myp/grasp_dataset'
    dump_folder = 'rect_labels'

    if not os.path.exists(dump_folder):
        os.mkdir(dump_folder)

    if NUM_PROCESS > 1:
        from multiprocessing import Pool

        pool = Pool(NUM_PROCESS)  # 使用 NUM_PROCESS
        for camera in ['realsense', 'kinect']:
            for sceneId in [140,142,153,155,160,161,162,174,175,176,177,186,187]:
                pool.apply_async(func=generate_scene_rectangle_grasp, args=(sceneId, dump_folder, camera))
        pool.close()
        pool.join()
    else:
        print("Running in single process mode")
        for camera in ['realsense', 'kinect']:
            for sceneId in [140,142,153,155,160,161,162,174,175,176,177,186,187]:
                generate_scene_rectangle_grasp(sceneId, dump_folder, camera)  # 执行单进程逻辑

    # from multiprocessing import Pool
    # for camera in ['realsense', 'kinect']:
    #
    #     pool = Pool(48)  # 使用 NUM_PROCESS
    #     for camera in ['realsense', 'kinect']:
    #         pool.apply_async(func=generate_scene_rectangle_grasp, args=(120, dump_folder, camera))
    #     pool.close()
    #     pool.join()
