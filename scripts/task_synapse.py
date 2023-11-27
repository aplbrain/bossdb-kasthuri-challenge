#!/usr/bin/env python

import torch
from typing import Tuple, List
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import numpy as np
from torch.utils import data
import pathlib 
from kasthuri.models.unet import UNet
from kasthuri.trainer import Trainer
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms import RandomCrop
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json as json
from kasthuri.bossdbdataset import BossDBDataset
from datetime import datetime
import argparse
import os
import random

import sys
import ssl
from tqdm import tqdm 
import segmentation_models_pytorch as smp
os.system('git clone https://github.com/black0017/MedicalZooPytorch/ && cd MedicalZooPytorch && mv lib ../. && cd .. && rm -rf MedicalZooPytorch')
sys.path.append(os.getcwd()) 

from torchsummary import summary
from loading_model import load_model

#This was necessary to overcome an SSL cert error when downloading pretrained weights for SMP baselines- your milelage may vary here
ssl._create_default_https_context = ssl._create_unverified_context

class AugmentBoth:
    """A class to perform data augmentation on both image and mask.

    This class applies random rotations, random crops to a fixed size, 
    and random horizontal flips to both the input image and its corresponding mask.

    Attributes:
        p_flip (float): Probability of applying a horizontal flip. Default is 0.5.
        max_rotation (int): Maximum angle (in degrees) for random rotation. Default is 30.
    """

    def __init__(self, p_flip=0.5, max_rotation=30):
        """
        Args:
            p_flip (float, optional): Probability of applying a horizontal flip. Defaults to 0.5.
            max_rotation (int, optional): Maximum angle (in degrees) for random rotation. Defaults to 30.
        """
        self.p_flip = p_flip
        self.max_rotation = max_rotation

    def __call__(self, image, mask):
        """Apply random rotation, random crop, and random horizontal flip to the image and mask.

        Args:
            image: The input image to be augmented.
            mask: The corresponding mask to be augmented.

        Returns:
            tuple: Augmented image and mask.
        """
        # Random Rotation with a probability of 0.5
        if random.random() < 0.5:
            angle = random.uniform(-self.max_rotation, self.max_rotation)
            image = F.rotate(image, angle)
            mask = F.rotate(mask, angle)

        # Random Crop to 256x256
        i, j, h, w = RandomCrop.get_params(image, output_size=(256, 256))
        image = F.crop(image, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)

        # Random Horizontal Flip
        if random.random() < self.p_flip:
            image = F.hflip(image)
            mask = F.hflip(mask)

        return image, mask


def train_model(task_config,network_config,boss_config=None,gpu='cuda'):
    """Train a model based on the provided configurations.

    This function initializes datasets, dataloaders, model, criterion, optimizer, and scheduler.
    It then trains the model, saves it, and computes various metrics on the test set.

    Args:
        task_config (dict): Configuration dictionary for the task.
        network_config (dict): Configuration dictionary for the network.
        boss_config (dict, optional): Configuration dictionary for the BOSS database. Defaults to None.
        gpu (str, optional): GPU device to use for training. Defaults to 'cuda'.

    Returns:
        None: This function doesn't return anything but saves the trained model and metrics.
    """
    torch.manual_seed(network_config['seed'])
    
    np.random.seed(network_config['seed'])
    
    augmentation_transform = AugmentBoth()

    train_data = BossDBDataset(task_config, mode='train', transform=augmentation_transform)
    val_data = BossDBDataset(task_config, boss_config, "val")
    test_data = BossDBDataset(task_config, boss_config, "test")

    training_dataloader = data.DataLoader(dataset=train_data,
                                        batch_size=network_config['batch_size'],
                                        shuffle=True)
    validation_dataloader = data.DataLoader(dataset=val_data,
                                        batch_size=network_config['batch_size'],
                                        shuffle=True)
    test_dataloader = data.DataLoader(dataset=test_data,
                                        batch_size=network_config['batch_size'],
                                        shuffle=True)


    x, y = next(iter(training_dataloader))

    # Specify device
    device = torch.device('cuda') if torch.cuda.is_available() else  torch.device('cpu')

    model = load_model(network_config, device)

    # criterion
    criterion = torch.nn.CrossEntropyLoss()

    # optimizer
    if network_config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=network_config["learning_rate"])
    if network_config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=network_config["learning_rate"], betas=(network_config["beta1"],network_config["beta2"]))
    
    # Adjust the number of epochs
    epochs = 30

    # Introduce a learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    trainer = Trainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        training_DataLoader=training_dataloader,
        validation_DataLoader=validation_dataloader,
        lr_scheduler=scheduler,
        epochs=epochs,
        notebook=False
    )

    # start training
    training_losses, validation_losses, lr_rates = trainer.run_trainer()

    # save the model
    date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
    model_name =  network_config['outweightfilename'] + '_' + task_config['task_type'] + '_' + date + '.pt'
    os.makedirs(pathlib.Path.cwd() / network_config['outputdir'], exist_ok = True) 
    torch.save(model.state_dict(), pathlib.Path.cwd() / network_config['outputdir'] / model_name)

    #take loss curves
    plt.figure()
    plt.plot(training_losses,label='training')
    plt.plot(validation_losses,label='validation')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Learning Curve')
    plt.legend()
    model_name = model_name[:len(model_name)-3] + '_learning_curve.png'
    plt.savefig(pathlib.Path.cwd() / network_config['outputdir'] / model_name)

    def predict(img,
                model,
                device,
                ):
        model.eval()
        x = img.to(device)  # to torch, send to device
        with torch.no_grad():
            out = model(x)  # send through model/network

        out_argmax = torch.argmax(out, dim=1)  # perform softmax on outputs
        return out_argmax

    # Create output folder
    output_folder = f"./outputs/{task_config['task_type']}_test_outputs"
    os.makedirs(output_folder, exist_ok=True)

    # Initialize accumulators for metrics
    tp_tot = torch.empty(0, network_config['classes'])
    fp_tot = torch.empty(0, network_config['classes'])
    fn_tot = torch.empty(0, network_config['classes'])
    tn_tot = torch.empty(0, network_config['classes'])

    for idx, (x_batch, y_batch) in enumerate(test_dataloader):
        for i in range(x_batch.size(0)):
            x = x_batch[i].unsqueeze(0)
            y = y_batch[i].unsqueeze(0)
            output = predict(x, model, device)

            # Compute metrics for the current batch
            tp, fp, fn, tn = smp.metrics.get_stats(output, y.to(device), mode='multiclass', num_classes=network_config['classes'])
            tp_tot = torch.vstack((tp_tot, tp))
            fp_tot = torch.vstack((fp_tot, fp))
            fn_tot = torch.vstack((fn_tot, fn))
            tn_tot = torch.vstack((tn_tot, tn))
        
            # Visualization
            f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 15))
            ax1.imshow(x_batch[i].squeeze(), cmap='gray')
            ax1.title.set_text("Slice")
            ax1.axis('off')

            ax2.imshow(y_batch[i].squeeze())  
            ax2.title.set_text("Groundtruth Annotation")
            ax2.axis('off')

            ax3.imshow(output.cpu().squeeze()) 
            ax3.title.set_text("Model Prediction")
            ax3.axis('off')

            plt.savefig(os.path.join(output_folder, f"slice_{idx * x_batch.size(0) + i + 1}.png"))
            plt.close()


    # Inform the user where the images are saved
    print(f"Images from the test set have been saved to: {output_folder}")

    # then compute metrics with required reduction (see metric docs)
    model_name = model_name[:len(model_name)-19] + '_report.rpt'
    rh = open(pathlib.Path.cwd() / network_config['outputdir'] / model_name, 'w')
    
    #Accuracy Per Class 
    acc = (tp_tot.mean(dim=0)+tn_tot.mean(dim=0))/(fp_tot.mean(dim=0)+tn_tot.mean(dim=0)+fn_tot.mean(dim=0)+tp_tot.mean(dim=0))
    print('Accuracy per Class:')
    print(np.array(acc.cpu()))
    rh.write('Accuracy per Class:\n')
    rh.write(str(np.array(acc.cpu())))
    
    spec =  (tn_tot[:,1:].mean())/(fp_tot[:,1:].mean()+tn_tot[:,1:].mean())
    sens =  (tp_tot[:,1:].mean())/(fn_tot[:,1:].mean()+tp_tot[:,1:].mean())
    balacc = (spec + sens)/2
    print(f'Balanced accuracy (No background): {balacc}')
    rh.write(f'Balanced accuracy (No background): {balacc}\n')
    
    prec = tp_tot.mean(dim=0)/(fp_tot.mean(dim=0)+tp_tot.mean(dim=0))
    reca = tp_tot.mean(dim=0)/(fn_tot.mean(dim=0)+tp_tot.mean(dim=0))
    f1 = (2*reca*prec)/(reca+prec)
    print(f'F1-score: {np.array(f1.cpu())} Avg. F1-score: {f1.mean()}')
    rh.write(f'F1-score: {np.array(f1.cpu())} Avg. F1-score: {f1.mean()}\n')

    iou = (tp_tot.mean(0))/(fp_tot.mean(0)+fn_tot.mean(0)+tp_tot.mean(0))
    print(f'IoU: {np.array(iou.cpu())} Avg. IoU-score: {iou.mean()}')
    rh.write(f'IoU: {np.array(iou.cpu())} Avg. IoU-score: {iou.mean()}\n')

    rh.close()


if __name__ == '__main__':
    # usage python3 task2_2D_smp_main.py --task task2.json --network UNet_2D.json --boss boss_config.json
    parser = argparse.ArgumentParser(description='flags for training')
    parser.add_argument('--task', default="kasthuri/taskconfig/synapse_task.json",
                        help='task config json file')
    parser.add_argument('--network', default="UNet_2D.json",
                        help='network config json file name (inside kasthuri/networkconfig/)')
    parser.add_argument('--boss', 
                        help='boss config json file')
    parser.add_argument('--gpu', 
                        help='index of the gpu to use')
    args = parser.parse_args()
    
    if args.gpu:
        gpu = 'cuda:'+args.gpu
    else:
        gpu = 'cuda'

    # Prepend the path to the network config file name
    network_config_path = os.path.join("kasthuri/networkconfig/", args.network)

    task_config = json.load(open(args.task))
    network_config = json.load(open(network_config_path))
    if args.boss:
        boss_config = json.load(open(args.boss))
    else:
        boss_config = None
    print('begining training')
    train_model(task_config, network_config, boss_config, gpu)

