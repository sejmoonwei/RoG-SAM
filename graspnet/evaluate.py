from graspnetAPI import GraspNet, Grasp, GraspGroup, RectGraspGroup
import cv2
import open3d as o3d
import numpy as np
from graspnetAPI.graspnet_eval import GraspNetEval
from graspnetAPI.utils.utils import batch_center_depth

# GraspNetAPI example for checking the data completeness.
# change the graspnet_root path

camera = 'kinect'
sceneId = 112
annId = 30
# file = '/data/myp/grasp_dataset/scenes/scene_0112/kinect/rect/0030.npy'
file = 'grasp_poses.npy'
####################################################################
graspnet_root = '/data/myp/grasp_dataset' # ROOT PATH FOR GRASPNET
####################################################################

# rext = np.load('/data/myp/grasp_dataset/scenes/scene_0001/kinect/rect/0000.npy')

g = GraspNet(graspnet_root, camera = camera, split = 'all')
bgr = g.loadBGR(sceneId = sceneId, camera = camera, annId = annId)
depth = g.loadDepth(sceneId = sceneId, camera = camera, annId = annId) #720,1280
rect_grasp_group = RectGraspGroup().from_npy(npy_file_path=file)



graspgroup_6d = rect_grasp_group.to_grasp_group(camera = camera, depths = depth)
# graspgroup_6d = graspgroup_6d.nms(translation_thresh=0.2, rotation_thresh = 30 / 180.0 * 3.1416)


if graspgroup_6d is not None:
    geometry = []
    geometry.append(g.loadScenePointCloud(sceneId, camera, annId))
    geometry += graspgroup_6d.to_open3d_geometry_list()
    o3d.visualization.draw_geometries(geometry)
else:
    print('No result because the depth is invalid, please try again!')




