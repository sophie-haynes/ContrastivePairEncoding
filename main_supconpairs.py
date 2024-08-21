import torch
import argparse
import torch.multiprocessing as mp
from models.contrastive import PairDifferenceEncoder
from models.baseline import BaselineResNet50
import os
from utils.data import set_loader
from utils.loss import MultiPosConLoss
from utils.train import train_contrastive, train_baseline

#set os environment for multiprocessing
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29600'

def main(rank, world_size, args):
    if world_size > 1:
        # Multi-GPU setup
        device = torch.device(f"cuda:{rank}")
    else:
        # Single-GPU setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0  # Set rank to 0 for non-distributed single-GPU


    if args.model_type == 'contrastive':
        model = PairDifferenceEncoder(fine_tune=args.fine_tune)
        loss_fn = MultiPosConLoss(temperature=args.temperature)

        optimizer = torch.optim.Adam([
            {'params': model.backbone.parameters(), 'lr': args.backbone_lr},
            {'params': model.fc.parameters(), 'lr': args.fc_lr}
        ])

        train_loader, val_loader = set_loader(args, rank, world_size)
        train_contrastive(rank, world_size, args, model, loss_fn, train_loader, val_loader, optimizer, num_epochs=args.epochs, log_dir=args.log_dir, model_save_path=args.model_save_path)

    elif args.model_type == 'baseline':
        model = BaselineResNet50(fine_tune=args.fine_tune)
        criterion = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.fc_lr)

        train_loader, val_loader = set_loader(args, rank, world_size)
        train_baseline(rank, world_size, args, model, train_loader, val_loader, optimizer, criterion, num_epochs=args.epochs, log_dir=args.log_dir, model_save_path=args.model_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model")

    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--backbone_lr', type=float, default=1e-5, help='learning rate for the backbone')
    parser.add_argument('--fc_lr', type=float, default=1e-3, help='learning rate for the fully connected layers')
    parser.add_argument('--temperature', type=float, default=0.1, help='temperature for the contrastive loss')
    parser.add_argument('--fine_tune', action='store_true', help='whether to fine-tune the backbone')
    parser.add_argument('--log_dir', type=str, default='runs/experiment', help='directory for TensorBoard logs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for DataLoader')
    parser.add_argument('--size', type=int, default=224, help='size of the image crops')
    parser.add_argument('--processing', type=str, default='crop', choices=['crop', 'lung_seg', 'arch_seg'], help='type of processing to apply to the images')
    parser.add_argument('--dataset', type=str, default='cxr14', help='name of the dataset')
    parser.add_argument('--data_folder', type=str, required=True, help='path to the dataset root folder')
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU to use')
    parser.add_argument('--model_type', type=str, choices=['contrastive', 'baseline'], required=True, help='which model type to train')
    parser.add_argument('--model_save_path', type=str, default='model.pth', help='path to save the trained model')

    args = parser.parse_args()

    world_size = torch.cuda.device_count()  # Number of GPUs available
    if world_size > 1:
        mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
    else:
        main(rank=0, world_size=1, args=args)

