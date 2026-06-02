import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler

from utils import *

from .atlas import Atlas
from .brat import Brat
from .ddti import DDTI
from .isic import ISIC2016
from .kits import KITS
from .lidc import LIDC
from .lnq import LNQ
from .pendal import Pendal
from .refuge import REFUGE
from .segrap import SegRap
from .stare import STARE
from .toothfairy import ToothFairy
from .wbc import WBC
from.Cornell import Cornell
from .Jacquard import Jacquard
from .OCID import OCID
from .real import REAL
from.Graspnet import Graspnet

def get_dataloader(args):
    transform_train = transforms.Compose([
        transforms.Resize((args.image_size,args.image_size)),
        transforms.ToTensor(),
    ])

    transform_train_seg = transforms.Compose([
        transforms.Resize((args.out_size,args.out_size)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    transform_test_seg = transforms.Compose([
        transforms.Resize((args.out_size,args.out_size)),
        transforms.ToTensor(),
    ])
    
    if args.dataset == 'isic':
        '''isic data'''
        isic_train_dataset = ISIC2016(args, args.data_path, transform = transform_train, transform_msk= transform_train_seg, mode = 'Training')
        isic_test_dataset = ISIC2016(args, args.data_path, transform = transform_test, transform_msk= transform_test_seg, mode = 'Test')

        nice_train_loader = DataLoader(isic_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(isic_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'decathlon':
        nice_train_loader, nice_test_loader, transform_train, transform_val, train_list, val_list = get_decath_loader(args)


    elif args.dataset == 'REFUGE':
        '''REFUGE data'''
        refuge_train_dataset = REFUGE(args, args.data_path, transform = transform_train, transform_msk= transform_train_seg, mode = 'Training')
        refuge_test_dataset = REFUGE(args, args.data_path, transform = transform_test, transform_msk= transform_test_seg, mode = 'Test')

        nice_train_loader = DataLoader(refuge_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(refuge_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
        '''end'''


    # ----------------grasp----------------#

    elif args.dataset == 'Cornell':
        '''Cornell data'''
        cornell_train_dataset = Cornell(args, args.data_path, transform=transform_train,
                                         mode='Training')
        cornell_test_dataset = Cornell(args, args.data_path, transform=transform_test,
                                       mode='Test')
        nice_train_loader = DataLoader(cornell_train_dataset, batch_size=args.b, shuffle=True, num_workers=8,
                                       pin_memory=True)  # worker 8 to 1
        nice_test_loader = DataLoader(cornell_test_dataset, batch_size= 1, shuffle=False, num_workers=8,
                                      pin_memory=True)

        '''end'''
    elif args.dataset == 'Jacquard':
        '''Jacquard data'''
        Jacquard_dataset = Jacquard(args, args.data_path, transform=transform_train,
                                         mode='Training')

        # 定义训练集和测试集的大小
        train_size = int(0.8 * len(Jacquard_dataset))  # 80%训练集
        test_size = len(Jacquard_dataset) - train_size  # 20%测试集
        print('Train_size:{},Test_size:{}'.format(train_size,test_size))
        # 设置随机种子
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_dataset, test_dataset = random_split(Jacquard_dataset, [train_size, test_size])
        # 获取索引
        train_indices = train_dataset.indices
        test_indices = test_dataset.indices
        # 保存索引到文件
        np.save('train_indices.npy', train_indices)
        np.save('test_indices.npy', test_indices)

        nice_train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=True, num_workers=8,
                                       pin_memory=True)  # worker 8 to 1
        nice_test_loader = DataLoader(test_dataset, batch_size= 1 , shuffle=False, num_workers=8,
                                      pin_memory=True)

        '''end'''

    elif args.dataset == 'OCID':
        '''OCID data'''
        OCID_train_dataset = OCID(args,  transform = transform_train,split_name = 'data_split/training_0' , data_path = args.data_path)

        OCID_test_dataset = OCID(args,  transform = transform_test, split_name = 'data_split/validation_0' ,data_path = args.data_path)
        nice_train_loader = DataLoader(OCID_train_dataset, batch_size=args.b, shuffle=True, num_workers=8,
                                       pin_memory=True)  # worker 8 to 1
        nice_test_loader = DataLoader(OCID_test_dataset, batch_size=1, shuffle=False, num_workers=8,
                                      pin_memory=True)
        '''end'''


    elif args.dataset == 'Graspnet':
        '''Graspnet data'''
        Graspnet_train_dataset = Graspnet(args,transform_train, camera=args.camera ,data_path = '/data/myp/grasp_dataset', split='train')
        Graspnet_test_seen_dataset = Graspnet(args,transform_train, camera=args.camera ,data_path = '/data/myp/grasp_dataset', split='test_seen')
        Graspnet_test_similar_dataset = Graspnet(args,transform_train, camera=args.camera ,data_path = '/data/myp/grasp_dataset', split='test_similar')
        Graspnet_test_novel_dataset = Graspnet(args,transform_train, camera=args.camera ,data_path = '/data/myp/grasp_dataset', split='test_novel')


        nice_train_loader = DataLoader(Graspnet_train_dataset,  batch_size=args.b, shuffle=True, num_workers=8,
                                       pin_memory=True)  # worker 8 to 1
        nice_test_seen_loader = DataLoader(Graspnet_test_seen_dataset, batch_size=1, shuffle=False, num_workers=8,
                                      pin_memory=True)
        nice_test_similar_loader = DataLoader(Graspnet_test_similar_dataset, batch_size=1, shuffle=False, num_workers=8,
                                           pin_memory=True)
        nice_test_novel_loader = DataLoader(Graspnet_test_novel_dataset, batch_size=1, shuffle=False, num_workers=8,
                                           pin_memory=True)
        '''end'''
        return  nice_train_loader, nice_test_seen_loader, nice_test_similar_loader, nice_test_novel_loader


    elif args.dataset == 'Mixreal':
        '''Mixreal'''
        train_dataset1 = Cornell(args, '/data1/samgrasp/dataset/cornell_adapt', transform=transform_train,
                                         mode='Training')

        train_dataset2 = OCID(args,  transform = transform_train,split_name = 'data_split/training_0' , data_path = '/data1/samgrasp/dataset/OCID/OCID_grasp')

        train_dataset3 = REAL(args, transform_train, data_path = '/data1/samgrasp/dataset/real')

        train_dataset4 = Jacquard(args, '/data1/samgrasp/dataset/Jacquard_V2', transform=transform_train,
                                         mode='Training')

        test_dataset1 = Cornell(args, '/data1/samgrasp/dataset/cornell_adapt', transform=transform_test,
                               mode='Test')
        test_loader1 = DataLoader(test_dataset1, batch_size=1, shuffle=False, num_workers=8,
                                 pin_memory=True)
        test_dataset2 = OCID(args,  transform = transform_test, split_name = 'data_split/validation_0' ,data_path = '/data1/samgrasp/dataset/OCID/OCID_grasp')
        test_loader2 = DataLoader(test_dataset2, batch_size=1, shuffle=False, num_workers=8,
                                      pin_memory=True)



        return train_dataset1, train_dataset2, train_dataset3 , train_dataset4, test_loader1, test_loader2




    elif args.dataset == 'LIDC':
        '''LIDC data'''
        # dataset = LIDC(data_path = args.data_path)
        dataset = MyLIDC(args, data_path = args.data_path,transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.2 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'DDTI':
        '''DDTI data'''
        refuge_train_dataset = DDTI(args, args.data_path, transform = transform_train, transform_msk= transform_train_seg, mode = 'Training')
        refuge_test_dataset = DDTI(args, args.data_path, transform = transform_test, transform_msk= transform_test_seg, mode = 'Test')

        nice_train_loader = DataLoader(refuge_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(refuge_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'Brat':
        '''Brat data'''
        dataset = Brat(args, data_path = args.data_path,transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.3 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'STARE':
        '''STARE data'''
        # dataset = LIDC(data_path = args.data_path)
        dataset = STARE(args, data_path = args.data_path, transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.2 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'kits':
        '''kits data'''
        dataset = KITS(args, data_path = args.data_path,transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.3 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'WBC':
        '''WBC data'''
        dataset = WBC(args, data_path = args.data_path,transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.3 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'segrap':
        '''segrap data'''
        dataset = SegRap(args, data_path = args.data_path,transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.3 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'toothfairy':
        '''toothfairy data'''
        dataset = ToothFairy(args, data_path = args.data_path,transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.3 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'atlas':
        '''atlas data'''
        dataset = Atlas(args, data_path = args.data_path,transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.3 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'pendal':
        '''pendal data'''
        dataset = Pendal(args, data_path = args.data_path,transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.3 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'lnq':
        '''lnq data'''
        dataset = LNQ(args, data_path = args.data_path,transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.3 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    else:
        print("the dataset is not supported now!!!")
        
    return nice_train_loader, nice_test_loader