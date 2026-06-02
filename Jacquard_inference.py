import torchvision.transforms.functional as TF
from utils import *
import matplotlib.pyplot as plt
from util.data.structure.img_jacquard import Image
from util.data.structure.grasp_jacquard import GraspMat, drawGrasp1
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import cv2
import matplotlib.patches as patches
from matplotlib.patches import Polygon, Rectangle
from utils import get_grasp



def create_star(x_center, y_center, outer_radius, inner_radius):
    star_points = []
    for i in range(10):
        angle = i * np.pi / 5
        radius = outer_radius if i % 2 == 0 else inner_radius
        x = x_center + radius * np.cos(angle)
        y = y_center - radius * np.sin(angle)
        star_points.append((x, y))
    star = Polygon(star_points, closed=True, edgecolor='r', facecolor='yellow')
    return star

def display_image_with_point(image, point= [0,0], box = [0,0,0,0]):
    fig, ax = plt.subplots()
    ax.imshow(image)
    # ax.scatter(point[0], point[1], color='red', s=20)  # 点的大小为100
    star = create_star(point[0], point[1], 10, 5)  # 创建五角星
    ax.add_patch(star)
    # 添加边界框
    rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    plt.axis("off")
    plt.savefig('/data1/samgrasp/dataset/test/jaq_test/show/rgb.png', bbox_inches='tight', pad_inches=0)
    # plt.show()
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
    plt.savefig('/data1/samgrasp/dataset/test/jaq_test/show/rgbgrasp.png', bbox_inches='tight', pad_inches=0)
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
    rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
    # add boundingbox
    # axs.add_patch(rect)

    # axs.set_title(titles[0])
    axs.axis('off')

    plt.tight_layout()
    plt.savefig('/data1/samgrasp/dataset/test/jaq_test/show/heatmap_label.png', bbox_inches='tight', pad_inches=0)
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



def show_heatmap(able_pred):
    able_pred_np = able_pred.squeeze().numpy()
    # 可视化为热度图
    plt.imshow(able_pred_np, cmap='jet', interpolation='nearest',vmin=0,vmax=0.7)
    # plt.colorbar()  # 添加颜色条
    # plt.title('Grasp Confidence Heatmap')
    plt.axis('off')
    # plt.show()
    plt.tight_layout()

    plt.savefig('/data1/samgrasp/dataset/test/jaq_test/show/heatmap.png', bbox_inches='tight', pad_inches=0)
    plt.close()  # 关闭图形以释放内存







transform_train = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
])

args = cfg.parse_args()
#--------------------args--------------------------#
args.pretrain    = '/data1/ori/checkpoint/jacquard_point_box.pth'
args.image_size  = 512
args.mask_size   = 256
args.out_size    = 512
args.file        = '/data1/samgrasp/dataset/Jacquard_V2/Jacquard_Dataset_0/1cfc37465809382edfd1d17b67edb09/3_1cfc37465809382edfd1d17b67edb09_RGB.png'
args.label_path  = args.file.replace('RGB.png', 'grasps.txt')
#--------------------------------------------------#
#3_8456c4a8189a91fb27eb00c151c6f711_RGB.png
GPUdevice = torch.device('cuda', args.gpu_device)
net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
print('Network build')
weights = torch.load(args.pretrain, map_location='cuda:0')['state_dict']
net.load_state_dict(weights,strict=True)



# raw image and raters images
image = Image(args.file)
label = GraspMat(args.label_path)

#-----argument------#
# crop
dist = 2  # 50
crop_bbox = image.crop(500, dist)
label.crop(crop_bbox)
label.decode(angle_cls=120)  # 4通道标签变122


# color
# image.color()
#------------------#
# img归一化
image.nomalise()

img = cv2.cvtColor(image.img,cv2.COLOR_BGR2RGB)
img = img.transpose((2, 0, 1))  # (320, 320, 3) -> (3, 320, 320)
img = numpy_to_torch(img)  # 3,320,320 under1
# ---------------nomalize----------

# 假设 img 是一个形状为 [3, 320, 320]，值范围为 -0.6 到 0.18 的张量
min_val = img.min()
max_val = img.max()

# 将图像数据归一化到 [0, 1]
img = (img - min_val) / (max_val - min_val)
# ---------------------------------#
img = transform_train(TF.to_pil_image(img))

# display_image(img.permute(1,2,0))

img_in = img.unsqueeze(0).to(GPUdevice)

#-------------get prompt--------------#
target = label.grasp  # (2 + angle_k, 320, 320)
target = numpy_to_torch(target)  # 122,320,320 tensor 0 or 1

conf_mask = F.interpolate(target[:1].unsqueeze(0), size=(args.mask_size, args.mask_size), mode='bilinear',
                          align_corners=False)  #

angle_mask = F.interpolate(target[1:-1].unsqueeze(0), size=(args.mask_size, args.mask_size), mode='nearest',
                           )  #

width_mask = F.interpolate(target[-1:].unsqueeze(0), size=(args.mask_size, args.mask_size), mode='nearest',
                           )  # 1,1,512,512

mask = torch.concat([conf_mask.squeeze(0)
                     # ,angle_mask.squeeze(0)
                     # , width_mask.squeeze(0)
                     ], dim=0)  # 122,512,512

newsize = (args.mask_size, args.mask_size)
click_mask_np = cv2.resize(target[0].numpy().copy(), (args.image_size,args.image_size), interpolation=cv2.INTER_LINEAR)
point_label, pt = random_click(click_mask_np, point_labels = 1)
x = pt[1]
y = pt[0]
pt[0] = x
pt[1] = y
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

box_mask = F.interpolate(conf_mask, size=(args.image_size, args.image_size), mode="bilinear")
box_mask = torch.as_tensor(box_mask > 0.0, dtype=torch.float32)
x_min_cup, x_max_cup, y_min_cup, y_max_cup = random_box(box_mask)  # ymin,ymax,xmin,xmax
xmin = y_min_cup
ymin = x_min_cup
xmax = y_max_cup
ymax = x_max_cup
# box_cup = [x_min_cup, x_max_cup, y_min_cup, y_max_cup]
box_cup = [xmin, ymin, xmax, ymax]
box = [torch.tensor(box_cup)]
combined_box = torch.stack(box, dim=0).to(dtype=torch.float32, device=GPUdevice)
combined_box = combined_box[:, None, :]






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

    pred = F.interpolate(pred, size=(args.out_size, args.out_size),mode='bilinear')
    pred = pred.squeeze(0)
    conf_angle = pred[:1]
    angle = pred[1:-1]
    angle_width = pred[-1:]
    able_pred = torch.sigmoid(conf_angle)
    angle_pred = torch.sigmoid(angle)
    width_pred = torch.sigmoid(angle_width)
    combined_slices = torch.cat((able_pred
                                 # , angle_pred
                                 , width_pred
                                 ), dim=0)
    show_heatmap(combined_slices[0].cpu())

    # ------visualize-------#
    grasp = get_grasp(able_pred.unsqueeze(0), angle_pred.unsqueeze(0), width_pred.unsqueeze(0))
    show_heatmap(combined_slices[0].cpu())
    # img_to_show = (image.img - image.img.min()) / (image.img.max() - image.img.min())
    display_image_with_point(img_in.squeeze(0).permute(1,2,0).cpu().numpy(),pt_coord,box_cup)  # 320,320,3   255
    display_image_with_grasp(img_in.squeeze(0).permute(1, 2, 0).cpu().numpy(), grasp)
    lbl = torch.from_numpy(label.grasp[:1]).unsqueeze(0)
    label_to_vis = F.interpolate(lbl, size=(args.image_size, args.image_size), mode="bilinear", align_corners=False)
    display_labels_with_point(label_to_vis.squeeze(0),pt_coord,box_cup)  # 4,320,320
    # ----------------------#
    # display_labels(combined_slices.cpu())
    pred_ = torch.as_tensor(combined_slices > 0.5, dtype=torch.float32).cpu()
    # display_labels_with_point(pred_[:1])
    pass


























