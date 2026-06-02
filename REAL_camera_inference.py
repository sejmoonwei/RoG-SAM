import torchvision.transforms.functional as TF
from utils import *
import matplotlib.pyplot as plt
from util.data.structure.img_real import Image
from util.data.structure.grasp_real import GraspMat, drawGrasp1
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import cv2
import matplotlib.patches as patches
from matplotlib.patches import Polygon, Rectangle
from utils import get_grasp
# from Grounding_sam import Grounding
from PIL import Image as I
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


def display_image_with_point(image, point=[0, 0], box=[0, 0, 0, 0]):
    fig, ax = plt.subplots()
    ax.imshow(image, alpha=0.7)

    # 创建并添加五角星
    star = create_star(point[0], point[1], 10, 5)
    ax.add_patch(star)

    # 添加边界框
    rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r',
                             facecolor='none')
    ax.add_patch(rect)

    prompt = 'bowlr'

    # 创建红色矩形区域
    text_box = patches.FancyBboxPatch((box[0] , box[1]-20), len(prompt)*10 + 5, 20, boxstyle="round,pad=0.3", linewidth=1, edgecolor='red',
                                      facecolor='red')
    ax.add_patch(text_box)

    # 在红色矩形区域内显示白色字体
    ax.text(box[0] , box[1] + 10 -20, prompt, fontsize=10, color='white', verticalalignment='center')

    plt.axis("off")
    plt.savefig('/data1/samgrasp/dataset/test/show_real/rgb.png', bbox_inches='tight', pad_inches=0)
    plt.close()  # 关闭图形以释放内存

def display_image_with_point1(image, point=[0, 0], box=[0, 0, 0, 0], dpi=111):
    # 根据图像的像素尺寸和目标 DPI 设置 figure 尺寸
    fig, ax = plt.subplots(figsize=(8,6), dpi=dpi)
    ax.imshow(image, alpha=1)

    # 创建并添加五角星
    star = create_star(point[0], point[1], 10, 5)
    # ax.add_patch(star)

    # 添加边界框
    rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
    # ax.add_patch(rect)

    # 创建红色矩形区域
    text_box = patches.FancyBboxPatch((box[0], box[1]-20), 60, 20, boxstyle="round,pad=0.3", linewidth=1, edgecolor='red', facecolor='red')
    # ax.add_patch(text_box)

    # 在红色矩形区域内显示白色字体
    # ax.text(box[0] + 5, box[1] + 10 -20, 'bowl', fontsize=10, color='white', verticalalignment='center')

    plt.axis("off")
    plt.savefig('/data1/samgrasp/dataset/test/show_real/rgb1.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
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
    plt.savefig('/data1/samgrasp/dataset/test/show_real/rgb_heatmap.png', bbox_inches='tight', pad_inches=0)
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
    plt.savefig('/data1/samgrasp/dataset/test/show_real/rgbgrasp.png', bbox_inches='tight', pad_inches=0)
    plt.close()  # 关闭图形以释放内存

def display_image_with_grasp_all(image, grasps):
    fig, ax = plt.subplots()
    ax.imshow(image)

    # show grasp
    long_edge_color = 'red'  # 绿色
    short_edge_color = 'green'  # 红色
    for grasp_obj in grasps:
        for pose in grasp_obj:
            for i in range(4):
                ax.plot(pose[i][1], pose[i][0], 'o', color='blue',markersize=3)  # 显示抓取点
                pt1 = (pose[i][1], pose[i][0])
                pt2 = (pose[(i + 1) % 4][1], pose[(i + 1) % 4][0])
                color = long_edge_color if i % 2 == 0 else short_edge_color
                ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=color)  # 显示抓取线条

    plt.axis("off")
    plt.savefig('/data1/samgrasp/dataset/test/show_real/rgbgrasp.png', bbox_inches='tight', pad_inches=0)
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


def show_heatmap(able_pred,i):
    able_pred_np = able_pred.squeeze().numpy()
    # 可视化为热度图
    cax = plt.imshow(able_pred_np, cmap='jet', interpolation='nearest',vmin=0.05,vmax=0.85) #0.7 for cornell
    # plt.colorbar()  # 添加颜色条
    # plt.title('Grasp Confidence Heatmap')
    plt.axis('off')
    # plt.show()
    plt.tight_layout()
    filename = f'/data1/samgrasp/dataset/test/show_real/heatmap{i}.png'
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    return cax

def save_heatmap(able_pred,i):
    npar = (able_pred.squeeze()*255).numpy().astype(np.uint8)
    image = I.fromarray(npar)
    filename = f'/data1/samgrasp/dataset/test/show_real/rawheatmap{i}.png'
    image.save(filename)



import json
def parse_detection_results(json_file_path):
    # 读取 JSON 文件
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # 创建一个列表来存储目标的字典信息
    detections = []

    # 遍历 JSON 数据中的每个目标
    for item in data:
        if item['label'] == 'stuff':  # 只处理 label 为 'stuff' 的目标
            detection = {
                'label': item['label'],
                'logit': item['logit'],
                'box': item['box']
            }
            detections.append(detection)

    return detections


transform_train = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
])

args = cfg.parse_args()
#-----------------------args--------------------------#

args.pretrain    = '/data1/ori/checkpoint/OCID45.pth'      #MixNew80.pth #OCIDsam902
args.image_size  = 512
args.mask_size   = 256
args.out_size    = 512
args.img_path = './png/pcd01014r.png'
args.json_path = './png/mask2.json'

GPUdevice = torch.device('cuda', args.gpu_device)

#------------------------------------------------------#
net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
print('Network build')
weights = torch.load(args.pretrain, map_location='cuda:0')['state_dict']
net.load_state_dict(weights,strict=True)



# raw image and raters images
image = Image(args.img_path)
# img归一化
image.nomalise()
img = cv2.cvtColor(image.img,cv2.COLOR_BGR2RGB)
img = img.transpose((2, 0, 1))  # (320, 320, 3) -> (3, 320, 320)
img = numpy_to_torch(img)  # 3,320,320 under1
min_val = img.min()
max_val = img.max()
img = (img - min_val) / (max_val - min_val)
img = transform_train(TF.to_pil_image(img))
img_in = img.unsqueeze(0).to(GPUdevice) #1,3,512,512   0-1

# loops for each object
#-------------get prompt--------------#
grasps = []
objs = parse_detection_results(args.json_path)
for i in range(len(objs)):
    boxi = objs[i]['box']
    boxi[0] *= 0.8
    boxi[2] *= 0.8
    boxi[1] = boxi[1] * 0.8 + 64
    boxi[3] = boxi[3] * 0.8 + 64
    px = int((boxi[0] + boxi[2])/2)
    py = int((boxi[1] + boxi[3])/2)
    pt = np.array([px, py])
    point_label = 1
    pt_coord = pt.copy()
    point_label = torch.tensor([point_label])
    pt = torch.tensor([pt])
    if point_label.clone().flatten()[0] != -1:
        # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
        point_coords = pt  # 390 339
        coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
        labels_torch = torch.as_tensor(point_label, dtype=torch.int, device=GPUdevice)
        if (len(point_label.shape) == 1):  # only one point prompt
            coords_torch, labels_torch = coords_torch[:, None, :], labels_torch[:, None]
        pt = (coords_torch, labels_torch)


    box_cup = boxi
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
        # angle_pred[angle_pred < 0.00000003] = 0
        # ang = torch.argmax(angle_pred, dim=0)
        # ang[ang>60] -= 60

        # ------visualize---------#

        grasp = get_grasp(able_pred.unsqueeze(0), angle_pred.unsqueeze(0), width_pred.unsqueeze(0))
        grasps.append(grasp)
        show_heatmap(combined_slices[0].cpu(),i)
        save_heatmap(combined_slices[0].cpu(),i)
        display_image_with_point(img_in.squeeze(0).permute(1,2,0).cpu().numpy(),pt_coord,box_cup)  # 320,320,3   255
        display_image_with_point1(img_in.squeeze(0).permute(1,2,0).cpu().numpy(),pt_coord,box_cup)  # 320,320,3   255
        display_image_with_grasp(img_in.squeeze(0).permute(1,2,0).cpu().numpy(),grasp)
        # display_image_with_hm(img_in.squeeze(0).permute(1,2,0).cpu().numpy(),pt_coord,box_cup,ang.cpu())
        # -------------------------#
    display_image_with_grasp_all(img_in.squeeze(0).permute(1,2,0).cpu().numpy(),grasps)

    pass