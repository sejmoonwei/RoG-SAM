import torch

from cfg import parse_args


SUPPORTED_DATASETS = {'Cornell', 'OCID'}


def load_checkpoint(net, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    net.load_state_dict(state_dict, strict=False)


def main():
    args = parse_args()
    if args.dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"RoG-SAM accuracy supports only {sorted(SUPPORTED_DATASETS)}.")
    if not args.pretrain:
        raise ValueError("Pass a trained checkpoint with '-pretrain /path/to/checkpoint.pth'.")

    from dataset import get_dataloader
    from utils import get_network

    device = torch.device('cuda', args.gpu_device)
    _, test_loader = get_dataloader(args)

    net = get_network(
        args,
        args.net,
        use_gpu=args.gpu,
        gpu_device=device,
        distribution=args.distributed,
    )
    load_checkpoint(net, args.pretrain, device)

    from function import validate

    results = validate(net, device, test_loader, args)
    total = results['correct'] + results['failed']
    accuracy = results['correct'] / total if total else 0.0

    print('>>> graspable = {:.5f}'.format(results['graspable']))
    print('>>> acc: %d/%d = %f' % (results['correct'], total, accuracy))
    if results.get('Non'):
        print('>>> ignored_none: {}'.format(results['Non']))


if __name__ == '__main__':
    main()
