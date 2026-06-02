import torchvision.transforms.functional as TF
from utils import *
import matplotlib.pyplot as plt
from util.data.structure.img_graspnet import Image
from util.data.structure.grasp_graspnet_inf import GraspMat, drawGrasp1
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import cv2
import matplotlib.patches as patches
from matplotlib.patches import Polygon, Rectangle
from utils import get_grasp
import time
# from Grounding_sam import Grounding

def create_star(x_center, y_center, outer_radius, inner_radius):
    star_points = []
    y_center -= 10
    for i in range(10):
        angle = i * np.pi / 5
        radius = outer_radius if i % 2 == 0 else inner_radius
        x = x_center + radius * np.cos(angle)
        y = y_center - radius * np.sin(angle)
        star_points.append((x, y))
    star = Polygon(star_points, closed=True, edgecolor='r', facecolor='yellow')
    return star



def display_image_with_point1(image, point=[0, 0], box=[0, 0, 0, 0]):
    fig, ax = plt.subplots()
    ax.imshow(image, alpha=1)

    # 创建并添加五角星
    star = create_star(point[0], point[1], 10, 5)
    ax.add_patch(star)

    # 添加边界框
    rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r',
                             facecolor='none')
    ax.add_patch(rect)

    # 创建红色矩形区域
    text_box = patches.FancyBboxPatch((box[0], box[1]-20), 60, 20, boxstyle="round,pad=0.3", linewidth=1, edgecolor='red',
                                      facecolor='red')
    # ax.add_patch(text_box)

    # 在红色矩形区域内显示白色字体
    # ax.text(box[0] + 5, box[1] + 10 -20, 'bowl', fontsize=10, color='white', verticalalignment='center')

    plt.axis("off")
    plt.savefig('/data/myp/otherdataset/dataset/Graspnet/rgb1.png', bbox_inches='tight', pad_inches=0)
    plt.close()  # 关闭图形以释放内存


def display_image_with_hm(image, point=[0, 0], box=[0, 0, 0, 0], able_pred=None):
    fig, ax = plt.subplots()
    ax.imshow(image)

    # 如果提供了able_pred，则显示热度图
    if able_pred is not None:
        show_heatmap(ax, able_pred)

    # 创建并添加五角星
    star = create_star(point[0], point[1], 10, 5)
    # ax.add_patch(star)

    # 添加边界框
    rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
    # ax.add_patch(rect)

    # 创建红色矩形区域
    text_box = patches.FancyBboxPatch((box[0], box[1]-20), 120, 20, boxstyle="round,pad=0.3", linewidth=1, edgecolor='red', facecolor='red')
    # ax.add_patch(text_box)

    # 在红色矩形区域内显示白色字体
    # ax.text(box[0] + 5, box[1] - 10, 'bowl', fontsize=10, color='white', verticalalignment='center')

    plt.axis("off")
    plt.savefig('/data/myp/otherdataset/dataset/Graspnet/rgb_heatmap.png', bbox_inches='tight', pad_inches=0)
    plt.close()  # 关闭图形以释放内存


def display_image_with_grasp(image, grasp):
    fig, ax = plt.subplots()
    ax.imshow(image)

    # show grasp
    long_edge_color = 'red'  # 绿色
    short_edge_color = 'green'  # 红色

    for pose in grasp:
        for i in range(4):
            ax.plot(pose[i][1], pose[i][0], 'o', color='blue',markersize=3)  # 显示抓取点
            pt1 = (pose[i][1], pose[i][0])
            pt2 = (pose[(i + 1) % 4][1], pose[(i + 1) % 4][0])
            color = long_edge_color if i % 2 == 0 else short_edge_color
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=color)  # 显示抓取线条

    plt.axis("off")
    plt.savefig('/data/myp/otherdataset/dataset/Graspnet/rgbgrasp.png', bbox_inches='tight', pad_inches=0)
    plt.close()  # 关闭图形以释放内存




def display_labels_with_point(data, point= [0,0], box = [0,0,0,0]):
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    titles = ['Label 1'
        # , 'Label 2'
        # , 'Label 3'
        # , 'Label 4'
              ]


    axs.imshow(data[0], cmap='jet')
    # axs.scatter(point[0], point[1], color='red', s=20)  # 在每个标签上标记点

    # 在每个子图上添加边界框
    # rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
    # add boundingbox
    # axs.add_patch(rect)

    # axs.set_title(titles[0])
    axs.axis('off')

    plt.tight_layout()
    plt.savefig('/data/myp/otherdataset/dataset/Graspnet/heatmap_label.png', bbox_inches='tight', pad_inches=0)
    # plt.show()
    plt.close()  # 关闭图形以释放内存

def numpy_to_torch(s):
    """
    numpy转tensor
    """
    if len(s.shape) == 2:
        return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
    else:
        return torch.from_numpy(s.astype(np.float32))

# def show_heatmap(ax, able_pred):
#     able_pred_np = able_pred.squeeze().numpy()
#     # able_pred_np[able_pred_np > (able_pred_np.max()*0.7)] *=  0.7
#     # 可视化为热度图
#     heatmap = ax.imshow(able_pred_np, cmap='jet', interpolation='nearest', vmin=able_pred.min(), vmax=able_pred.max() , alpha=0.5) # 0.7 for cornell
#     return heatmap


def show_heatmap(able_pred):
    able_pred_np = able_pred.squeeze().numpy()
    # 可视化为热度图
    cax = plt.imshow(able_pred_np, cmap='jet', interpolation='nearest',vmin=0.05,vmax=0.85) #0.7 for cornell
    # plt.colorbar()  # 添加颜色条
    # plt.title('Grasp Confidence Heatmap')
    plt.axis('off')
    # plt.show()
    plt.tight_layout()

    plt.savefig('/data/myp/otherdataset/dataset/Graspnet/heatmap.png', bbox_inches='tight', pad_inches=0)
    return cax


def display_labels_and_points(image, boxes_vis, points_vis):
    fig, ax = plt.subplots(dpi=500)
    ax.imshow(image, alpha=1)

    # 定义可复现的颜色列表
    num_boxes = len(boxes_vis)
    cmap = plt.get_cmap('tab20')  # 使用固定的颜色映射
    colors = cmap.colors[:num_boxes]  # 取前 num_boxes 个颜色

    # 遍历每个边界框和对应的点
    for idx, (box, point) in enumerate(zip(boxes_vis, points_vis)):
        color = colors[idx % len(colors)]  # 为每个边界框指定一个颜色

        # 绘制点（所有点颜色相同）
        star = create_star(point[0], point[1], 7, 4)
        ax.add_patch(star)

        # 绘制边界框，使用指定的颜色
        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=1,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)

    plt.axis("off")
    plt.savefig('/data/myp/otherdataset/dataset/Graspnet/box+point_vis.png', bbox_inches='tight', pad_inches=0)
    plt.close()  # 关闭图形以释放内存
    return colors  # 返回颜色列表，以便在下一个函数中使用

def display_grasps_on_img(image, grasps_vis, colors):
    fig, ax = plt.subplots(dpi=500)
    ax.imshow(image)

    short_edge_color = 'blue'  # 短边颜色固定为绿色

    # 遍历每组抓取
    for idx, grasp in enumerate(grasps_vis):
        long_edge_color = colors[idx % len(colors)]  # 长边颜色与对应的边界框颜色一致

        for pose in grasp:
            for i in range(4):
                ax.plot(
                    pose[i][1], pose[i][0],
                    'o', color='blue', markersize=2
                )  # 显示抓取点
                pt1 = (pose[i][1], pose[i][0])
                pt2 = (pose[(i + 1) % 4][1], pose[(i + 1) % 4][0])
                color = long_edge_color if i % 2 == 0 else short_edge_color
                ax.plot(
                    [pt1[0], pt2[0]],
                    [pt1[1], pt2[1]],
                    color=color
                )  # 显示抓取线条

    plt.axis("off")
    plt.savefig('/data/myp/otherdataset/dataset/Graspnet/grasps.png', bbox_inches='tight', pad_inches=0)
    plt.close()  # 关闭图形以释放内存


if __name__ == '__main__':


    transform_train = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
    ])

    args = cfg.parse_args()
    #-----------------------args--------------------------#
    args.pretrain    = './checkpoint/25kinect_11_08_11_04_31.pth' #OCIDsam902   5-80 10-90
    args.image_size  = 512
    args.mask_size   = 256
    args.out_size    = 512
    args.scene = '0112' #106 030
    args.index = '0030'
    args.camera = 'kinect' #

    args.sample_path = '/data/myp/grasp_dataset/scenes/scene_' + args.scene + '/' + args.camera
    args.img_path = os.path.join(args.sample_path,'rgb' ,args.index + '.png')
    args.anno_path = os.path.join(args.sample_path, 'rect', args.index + '.npy')
    args.ins_mask_path = os.path.join(args.sample_path, 'label', args.index + '.png')
    GPUdevice = torch.device('cuda', args.gpu_device)
    #------------------------------------------------------#


    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
    print('Network build')
    total_params = sum(p.numel() for p in net.parameters())
    print(f'total params: {total_params}')

    weights = torch.load(args.pretrain, map_location='cuda:0')['state_dict']
    net.load_state_dict(weights,strict=True)
    # for name, param in weights.items():
    #     if name in weights:
    #         net.state_dict()[name].copy_(param)





    # raw image and raters images
    image = Image(args.img_path)
    labels = GraspMat(args.anno_path, args.ins_mask_path) #随机选一个目标
    # prompt = 'orange'

    #-----argument------#
    # crop
    dist = -1 # 50
    crop_bbox = image.crop(500, dist)
    labels.crop(crop_bbox)
    labels.decode(angle_cls=120)  # 4通道标签变122
    #------------------#
    # img归一化
    image.nomalise()
    img = cv2.cvtColor(image.img,cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))  # (320, 320, 3) -> (3, 320, 320)
    img = numpy_to_torch(img)  # 3,320,320 under1
    # ---------------nomalize----------
    min_val = img.min()
    max_val = img.max()
    img = (img - min_val) / (max_val - min_val)
    # ---------------------------------#
    img = transform_train(TF.to_pil_image(img))

    # display_image(img.permute(1,2,0))

    img_in = img.unsqueeze(0).to(GPUdevice) #1,3,512,512   0-1

    # if use Grounding DINO
    # box0,pt0 = Grounding(img_in.squeeze().cpu(),args.prompt) #tensor([[ 63.8085, 134.3783, 212.4087, 256.4101]], device='cuda:0')

    #
    boxes_vis = []
    points_vis = []
    grasps_vis = []
    grasp_graspnet = []
    #-------------inference--------------#
    for i in range(len(labels.grasps)):
          # (2 + angle_k, 320, 320)
        target = numpy_to_torch(labels.grasps[i])  # 122,320,320 tensor 0 or 1
        target_msk = numpy_to_torch(labels.box_masks[i])
        conf_mask = F.interpolate(target[:1].unsqueeze(0), size=(args.mask_size, args.mask_size), mode='bilinear',
                                  align_corners=False)  #
        angle_mask = F.interpolate(target[1:-1].unsqueeze(0), size=(args.mask_size, args.mask_size), mode='nearest',
                                   ) #
        width_mask = F.interpolate(target[-1:].unsqueeze(0), size=(args.mask_size, args.mask_size), mode='nearest',
                                   ) #  1,1,512,512
        mask = torch.concat([conf_mask.squeeze(0)
                             # ,angle_mask.squeeze(0)
                             # , width_mask.squeeze(0)
                             ], dim=0)  # 122,512,512
        newsize = (args.mask_size, args.mask_size)
        click_mask_np = cv2.resize(target_msk[0].numpy().copy(), (args.image_size,args.image_size), interpolation=cv2.INTER_LINEAR)

        point_label = 1
        point_label, pt = random_click(click_mask_np, point_label)  # in: inputsize,input 0-1
        x = pt[1]
        y = pt[0]
        pt[0] = x
        pt[1] = y
        # point_label, pt_disc = random_click(np.array(target[0]) , point_label)


        pt_coord = pt.copy()
        point_label = torch.tensor([point_label])
        pt = torch.tensor([pt])
        if point_label.clone().flatten()[0] != -1:
            # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
            point_coords = pt  # 390 339
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
            labels_torch = torch.as_tensor(point_label, dtype=torch.int, device=GPUdevice)
            if (len(point_label.shape) == 1):  # only one point prompt
                # coords_torch, labels_torch, showp = coords_torch[None, :, :], labels_torch[None, :], showp[None, :, :]
                coords_torch, labels_torch = coords_torch[:, None, :], labels_torch[:, None]
            pt = (coords_torch, labels_torch)


        box_mask = torch.from_numpy(labels.box_masks[i]).unsqueeze(0)
        box_mask = F.interpolate(box_mask, size=(args.image_size, args.image_size), mode="nearest")
        box_mask = torch.as_tensor(box_mask > 0.0, dtype=torch.float32)
        x_min_cup, x_max_cup, y_min_cup, y_max_cup = random_box(box_mask)  # ymin,ymax,xmin,xmax
        xmin = y_min_cup
        ymin = x_min_cup
        xmax = y_max_cup
        ymax = x_max_cup
        box_cup = [xmin, ymin, xmax, ymax]





        # box_cup = [151., 207., 186., 246.] #pear
        box = [torch.tensor(box_cup)]
        combined_box = torch.stack(box, dim=0).to(dtype=torch.float32, device=GPUdevice)
        combined_box = combined_box[:, None, :]
        #FIX
        # box_cup = [int(box0[0][0]),int(box0[0][1]),int(box0[0][2]),int(box0[0][3])]

        strat_time = time.time()
        with torch.no_grad():
            imge = net.image_encoder(img_in)
            if args.net == 'sam' or args.net == 'mobile_sam':
                se, de = net.prompt_encoder(
                    points= pt, #pt,
                    boxes= combined_box, #combined_box,
                    masks=None,
                )
            if args.net == 'sam':
                pred, _ = net.mask_decoder(
                    image_embeddings=imge,
                    image_pe=net.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de,
                    multimask_output=(args.multimask_output > 1),
                )

            end_time = time.time()
            print(f'the inference time is : {end_time-strat_time}')

            pred = F.interpolate(pred, size=(args.out_size, args.out_size),mode='bilinear')
            pred = pred.squeeze(0)
            conf_angle = pred[:1]
            angle = pred[1:-1]
            angle_width = pred[-1:]
            able_pred = torch.sigmoid(conf_angle)
            angle_pred = torch.sigmoid(angle)
            width_pred = torch.sigmoid(angle_width)


            grasp = get_grasp(able_pred.unsqueeze(0), angle_pred.unsqueeze(0), width_pred.unsqueeze(0)) #512,512
            grasp_graspnet_ = grasp.copy()
            grasp_graspnet_[0] = grasp_graspnet_[0].astype(np.float32)
            grasp_graspnet_[0][:, 0] -= 112
            grasp_graspnet_[0][:,0] *= 2.5
            grasp_graspnet_[0][:,1] *= 2.5
            grasp_graspnet_[0] = grasp_graspnet_[0].astype(np.int32)  # 最后将结果转换为整数类型

            boxes_vis.append(box_cup)
            points_vis.append(pt_coord)
            grasps_vis.append(grasp)
            grasp_graspnet.append(grasp_graspnet_)

    colors = display_labels_and_points(img_in.squeeze(0).permute(1,2,0).cpu().numpy(),boxes_vis,points_vis)
    display_grasps_on_img(img_in.squeeze(0).permute(1,2,0).cpu().numpy(),grasps_vis, colors)


def to_graspnet_npy(grasps):
    grasp_poses = []

    for grasp in grasps:
        corners = np.array(grasp[0])

        # 计算矩形中心坐标
        center_x = np.mean(corners[:, 1])
        center_y = np.mean(corners[:, 0])

        # 计算右边中点坐标（右下角和右上角的中点）
        right_mid_x = (corners[1][1] + corners[2][1]) / 2
        right_mid_y = (corners[1][0] + corners[2][0]) / 2

        # 计算矩形的高度
        height = np.linalg.norm(corners[0] - corners[1])

        # 创建 n×7 的姿态行
        grasp_pose = [center_x, center_y, right_mid_x, right_mid_y, height, 0.8, 0]
        grasp_poses.append(grasp_pose)



    # 转换为 numpy 数组并保存为 npy 文件

    grasp_poses = np.array(grasp_poses)

    np.save("grasp_poses.npy", grasp_poses)

to_graspnet_npy(grasp_graspnet)


