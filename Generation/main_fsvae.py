import os
import os.path
import numpy as np
import logging
import argparse


import torch
import torchvision

from torch.utils.tensorboard import SummaryWriter

import global_v as glv
from network_parser import parse
import load_dataset_snn
# from utils import aboutCudaDevices
from utils import AverageMeter
# from utils import CountMulAddSNN
import fsvae_models.fsvae as fsvae
from fsvae_models.snn_layers import LIFSpike
import metrics.inception_score as inception_score
import metrics.clean_fid as clean_fid
import metrics.autoencoder_fid as autoencoder_fid


max_accuracy = 0
min_loss = 1000






def write_weight_hist(net, index):
    for n, m in net.named_parameters():
        root, name = os.path.splitext(n)
        writer.add_histogram(root + '/' + name, m, index)

def train(network, trainloader, opti, epoch):
    n_steps = glv.network_config['n_steps']
    max_epoch = glv.network_config['epochs']
    
    loss_meter = AverageMeter()
    recons_meter = AverageMeter()
    dist_meter = AverageMeter()

    mean_q_z = 0
    mean_p_z = 0
    mean_sampled_z = 0

    network = network.train()
    
    for batch_idx, (real_img, labels) in enumerate(trainloader):   
        opti.zero_grad()
        real_img = real_img.to(init_device, non_blocking=True)
        labels = labels.to(init_device, non_blocking=True)
        # direct spike input
        spike_input = real_img.unsqueeze(-1).repeat(1, 1, 1, 1, n_steps) # (N,C,H,W,T)
        x_recon, q_z, p_z, sampled_z = network(spike_input, scheduled=network_config['scheduled']) # sampled_z(B,C,1,1,T)
        
        if network_config['loss_func'] == 'mmd':
            losses = network.loss_function_mmd(real_img, x_recon, q_z, p_z)
        elif network_config['loss_func'] == 'kld':
            losses = network.loss_function_kld(real_img, x_recon, q_z, p_z)
        else:
            raise ValueError('unrecognized loss function')
        
        losses['loss'].backward()
        
        opti.step()

        loss_meter.update(losses['loss'].detach().cpu().item())
        recons_meter.update(losses['Reconstruction_Loss'].detach().cpu().item())
        dist_meter.update(losses['Distance_Loss'].detach().cpu().item())

        mean_q_z = (q_z.mean(0).detach().cpu() + batch_idx * mean_q_z) / (batch_idx+1) # (C,k,T)
        mean_p_z = (p_z.mean(0).detach().cpu() + batch_idx * mean_p_z) / (batch_idx+1) # (C,k,T)
        mean_sampled_z = (sampled_z.mean(0).detach().cpu() + batch_idx * mean_sampled_z) / (batch_idx+1) # (C,T)

        print(f'Train[{epoch}/{max_epoch}] [{batch_idx}/{len(trainloader)}] Loss: {loss_meter.avg}, RECONS: {recons_meter.avg}, DISTANCE: {dist_meter.avg}')

        if batch_idx == len(trainloader)-1:
            os.makedirs(f'checkpoint/{name}/imgs/train/', exist_ok=True)
            torchvision.utils.save_image((real_img+1)/2, f'checkpoint/{name}/imgs/train/epoch{epoch}_input.png')
            torchvision.utils.save_image((x_recon+1)/2, f'checkpoint/{name}/imgs/train/epoch{epoch}_recons.png')
            writer.add_images('Train/input_img', (real_img+1)/2, epoch)
            writer.add_images('Train/recons_img', (x_recon+1)/2, epoch)

    logging.info(f"Train [{epoch}] Loss: {loss_meter.avg} ReconsLoss: {recons_meter.avg} DISTANCE: {dist_meter.avg}")
    writer.add_scalar('Train/loss', loss_meter.avg, epoch)
    writer.add_scalar('Train/recons_loss', recons_meter.avg, epoch)
    writer.add_scalar('Train/distance', dist_meter.avg, epoch)
    writer.add_scalar('Train/mean_q', mean_q_z.mean().item(), epoch)
    writer.add_scalar('Train/mean_p', mean_p_z.mean().item(), epoch)
    

    writer.add_image('Train/mean_sampled_z', mean_sampled_z.unsqueeze(0), epoch)
    mean_q_z = mean_q_z.permute(1,0,2) # (k,C,T)
    mean_p_z = mean_p_z.permute(1,0,2) # (k,C,T)
    writer.add_image(f'Train/mean_q_z', mean_q_z.mean(0).unsqueeze(0))
    writer.add_image(f'Train/mean_p_z', mean_p_z.mean(0).unsqueeze(0))

    return loss_meter.avg


def test(network, testloader, epoch):
    n_steps = glv.network_config['n_steps']
    max_epoch = glv.network_config['epochs']

    loss_meter = AverageMeter()
    recons_meter = AverageMeter()
    dist_meter = AverageMeter()

    mean_q_z = 0
    mean_p_z = 0
    mean_sampled_z = 0



    network = network.eval()
    with torch.no_grad():
        for batch_idx, (real_img, labels) in enumerate(testloader):   
            real_img = real_img.to(init_device, non_blocking=True)
            labels = labels.to(init_device, non_blocking=True)
            # direct spike input
            spike_input = real_img.unsqueeze(-1).repeat(1, 1, 1, 1, n_steps) # (N,C,H,W,T)

            x_recon, q_z, p_z, sampled_z = network(spike_input, scheduled=network_config['scheduled'])

            if network_config['loss_func'] == 'mmd':
                losses = network.loss_function_mmd(real_img, x_recon, q_z, p_z)
            elif network_config['loss_func'] == 'kld':
                losses = network.loss_function_kld(real_img, x_recon, q_z, p_z)
            else:
                raise ValueError('unrecognized loss function')

            mean_q_z = (q_z.mean(0).detach().cpu() + batch_idx * mean_q_z) / (batch_idx+1) # (C,k,T)
            mean_p_z = (p_z.mean(0).detach().cpu() + batch_idx * mean_p_z) / (batch_idx+1) # (C,k,T)
            mean_sampled_z = (sampled_z.mean(0).detach().cpu() + batch_idx * mean_sampled_z) / (batch_idx+1) # (C,T)
            
            loss_meter.update(losses['loss'].detach().cpu().item())
            recons_meter.update(losses['Reconstruction_Loss'].detach().cpu().item())
            dist_meter.update(losses['Distance_Loss'].detach().cpu().item())

            print(f'Test[{epoch}/{max_epoch}] [{batch_idx}/{len(testloader)}] Loss: {loss_meter.avg}, RECONS: {recons_meter.avg}, DISTANCE: {dist_meter.avg}')

            if batch_idx == len(testloader)-1:
                os.makedirs(f'checkpoint/{name}/imgs/test/', exist_ok=True)
                torchvision.utils.save_image((real_img+1)/2, f'checkpoint/{name}/imgs/test/epoch{epoch}_input.png')
                torchvision.utils.save_image((x_recon+1)/2, f'checkpoint/{name}/imgs/test/epoch{epoch}_recons.png')
                writer.add_images('Test/input_img', (real_img+1)/2, epoch)
                writer.add_images('Test/recons_img', (x_recon+1)/2, epoch)
                

    logging.info(f"Test [{epoch}] Loss: {loss_meter.avg} ReconsLoss: {recons_meter.avg} DISTANCE: {dist_meter.avg}")
    writer.add_scalar('Test/loss', loss_meter.avg, epoch)
    writer.add_scalar('Test/recons_loss', recons_meter.avg, epoch)
    writer.add_scalar('Test/distance', dist_meter.avg, epoch)
    writer.add_scalar('Test/mean_q', mean_q_z.mean().item(), epoch)
    writer.add_scalar('Test/mean_p', mean_p_z.mean().item(), epoch)


    writer.add_image('Test/mean_sampled_z', mean_sampled_z.unsqueeze(0), epoch)
    mean_q_z = mean_q_z.permute(1,0,2) # # (k,C,T)
    mean_p_z = mean_p_z.permute(1,0,2) # # (k,C,T)
    writer.add_image(f'Test/mean_q_z', mean_q_z.mean(0).unsqueeze(0))
    writer.add_image(f'Test/mean_p_z', mean_p_z.mean(0).unsqueeze(0))

    return loss_meter.avg

def sample(network, epoch, batch_size=128):
    network = network.eval()
    with torch.no_grad():
        sampled_x, sampled_z = network.sample(batch_size)
        writer.add_images('Sample/sample_img', (sampled_x+1)/2, epoch)
        writer.add_image('Sample/mean_sampled_z', sampled_z.mean(0).unsqueeze(0), epoch)
        os.makedirs(f'checkpoint/{name}/imgs/sample/', exist_ok=True)
        torchvision.utils.save_image((sampled_x+1)/2, f'checkpoint/{name}/imgs/sample/epoch{epoch}_sample.png')

def calc_inception_score(network, epoch, batch_size=256):
    network = network.eval()
    with torch.no_grad():
        if (epoch%5 == 0) or epoch==glv.network_config['epochs']-1:
            batch_times=10
        else:
            batch_times=4
        inception_mean, inception_std = inception_score.get_inception_score(network, device=init_device, batch_size=batch_size, batch_times=batch_times)
        logging.info(f"Train [{epoch}] inception_score_mean: {inception_mean} inception_score_std: {inception_std} ")
        writer.add_scalar('Sample/inception_score_mean', inception_mean, epoch)
        writer.add_scalar('Sample/inception_score_std', inception_std, epoch)

def calc_clean_fid(network, epoch):
    network = network.eval()
    with torch.no_grad():
        num_gen=5000
        fid_score = clean_fid.get_clean_fid_score(network, glv.network_config['dataset'], init_device, num_gen)
        writer.add_scalar('Sample/FID', fid_score, epoch)

def calc_autoencoder_frechet_distance(network, epoch):
    network = network.eval()
    if glv.network_config['dataset'] == "MNIST":
        dataset = 'mnist'
    elif glv.network_config['dataset'] == "FashionMNIST":
        dataset = 'fashion'
    elif glv.network_config['dataset'] == "CelebA":
        dataset = 'celeba'
    elif glv.network_config['dataset'] == "CIFAR10":
        dataset = 'cifar10'
    else:
        raise ValueError()

    with torch.no_grad():
        fid_score = autoencoder_fid.get_autoencoder_frechet_distance(network, dataset, init_device, 5000)
        logging.info(f"Train [{epoch}] autoencoder_frechet_distance: {fid_score} ")
        # writer.add_scalar('Sample/AutoencoderDist', fid_score, epoch)
        #


if __name__ == '__main__':
    path = r'D:\opencv-python\FullySpikingVAE-master\NetworkConfigs\CIFAR10.yaml'
    name = 'log_LIF_MPD'
    path2 = r'E:\snn实习\HH\log_HH_CIFAR10\checkpoint.pth'
    parser = argparse.ArgumentParser()
    # parser.add_argument('name', type=str,default='log_LIF_MPD')
    parser.add_argument('-config', action='store', dest='config', default=path,help='The path of config file')
    parser.add_argument('-checkpoint', action='store', dest='checkpoint', default=path2,help='The path of checkpoint, if use checkpoint')
    # parser.add_argument('-checkpoint', action='store', dest='checkpoint', help='The path of checkpoint, if use checkpoint')
    parser.add_argument('-device', type=int)

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        exit(0)

    if args.config is None:
        raise Exception('Unrecognized config file.')

    if args.device is None:
        init_device = torch.device("cuda:0")
    else:
        init_device = torch.device(f"cuda:{args.device}")
    
    os.makedirs(f'checkpoint/{name}', exist_ok=True)
    writer = SummaryWriter(log_dir=f'checkpoint/{name}/tb')
    logging.basicConfig(filename=f'checkpoint/{name}/{name}.log', level=logging.INFO)
    
    logging.info("start parsing settings")
    
    params = parse(args.config)
    network_config = params['Network']
    
    logging.info("finish parsing settings")
    logging.info(network_config)
    print(network_config)
        
    # Check whether a GPU is available

    
    glv.init(network_config, [args.device])

    dataset_name = glv.network_config['dataset']
    data_path = glv.network_config['data_path']
    
    logging.info("dataset loading...")
    if dataset_name == "MNIST":
        data_path = os.path.expanduser(data_path)
        train_loader, test_loader = load_dataset_snn.load_mnist(data_path)
    elif dataset_name == "FashionMNIST":
        data_path = os.path.expanduser(data_path)
        train_loader, test_loader = load_dataset_snn.load_fashionmnist(data_path)
        
    elif dataset_name == "CIFAR10":
        data_path = os.path.expanduser(data_path)
        train_loader, test_loader = load_dataset_snn.load_cifar10(data_path)
        
    elif dataset_name == "CelebA":
        data_path = os.path.expanduser(data_path)
        train_loader, test_loader = load_dataset_snn.load_celebA(data_path)
        
    else:
        raise Exception('Unrecognized dataset name.')
    logging.info("dataset loaded")

    if network_config['model'] == 'FSVAE_TAID':
        net = fsvae.FSVAE_TAID()
    elif network_config['model'] == 'FSVAELarge_TAID':
        net = fsvae.FSVAELarge_TAID()
    elif network_config['model'] == 'FSVAE':
        net = fsvae.FSVAE()
    elif network_config['model'] == 'FSVAELarge':
        net = fsvae.FSVAELarge()
    else:
        raise Exception('not defined model')

    net = net.to(init_device)
    
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint)    
    optimizer = torch.optim.AdamW(net.parameters(), 
                                lr=glv.network_config['lr'], 
                                betas=(0.9, 0.999), 
                                weight_decay=0.001)
    
    best_loss = 1e8
    for e in range(glv.network_config['epochs']):
        
        # write_weight_hist(net, e)
        # if network_config['scheduled']:
        #     net.update_p(e, glv.network_config['epochs'])
        #     logging.info("update p")
        # train_loss = train(net, train_loader, optimizer, e)
        # test_loss = test(net, test_loader, e)
        #
        # torch.save(net.state_dict(), f'checkpoint/{name}/checkpoint.pth')
        # if test_loss < best_loss:
        #     best_loss = test_loss
        #     torch.save(net.state_dict(), f'checkpoint/{name}/best.pth')
        from spikingjelly.clock_driven import functional
        # sample(net, e, batch_size=128)
        # functional.reset_net(net)
        calc_inception_score(net, e)
        functional.reset_net(net)
        calc_autoencoder_frechet_distance(net, e)
        functional.reset_net(net)
        calc_clean_fid(net, e)
        functional.reset_net(net)
        
    writer.close()

