import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from .Cornell import Cornell
from .OCID import OCID


def get_dataloader(args):
    transform_train = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    if args.dataset == 'Cornell':
        train_dataset = Cornell(args, args.data_path, transform=transform_train, mode='Training')
        test_dataset = Cornell(args, args.data_path, transform=transform_test, mode='Test')
    elif args.dataset == 'OCID':
        train_dataset = OCID(
            args,
            transform=transform_train,
            split_name='data_split/training_0',
            data_path=args.data_path,
        )
        test_dataset = OCID(
            args,
            transform=transform_test,
            split_name='data_split/validation_0',
            data_path=args.data_path,
        )
    else:
        raise ValueError("Available dataloaders: 'Cornell' and 'OCID'.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.b,
        shuffle=True,
        num_workers=args.w,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.w,
        pin_memory=True,
    )
    return train_loader, test_loader
