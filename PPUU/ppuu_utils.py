import argparse
from array import array
import glob
import json
import math
import numpy
import os
import pdb
import re
import sys
from datetime import datetime
from os import path
import numpy as np
import copy

import matplotlib.pyplot as plt
import scipy
import sklearn.manifold as manifold
import torch
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from PIL import Image, ImageDraw
from sklearn import decomposition
from ferrygym import FerryGymEnv, cost, plot_helpers



"""
Source: https://github.com/Atcold/pytorch-PPUU/blob/master/utils.py

Lots of MASS specific functions added.
MASS functions are commented.
"""


def printnorm(x):
    print(x.norm())


def printgradnorm(self, grad_input, grad_output):
    print('Inside ' + self.__class__.__name__ + ' backward')
    print('Inside class:' + self.__class__.__name__)
    print('')
    print('grad_input: ', type(grad_input))
    print('grad_input[0]: ', type(grad_input[0]))
    print('grad_output: ', type(grad_output))
    print('grad_output[0]: ', type(grad_output[0]))
    print('')
    print('grad_input size:', grad_input[0].size())
    print('grad_output size:', grad_output[0].size())
    print('grad_input norm:', grad_input[0].norm())


def read_images(dirname, pytorch=True):
    imgs = []
    for f in glob.glob(dirname + '*.png'):
        im = scipy.misc.imread(f)
        if pytorch:
            im = torch.from_numpy(im)
        imgs.append(im)
    if pytorch:
        imgs = torch.stack(imgs).permute(0, 3, 1, 2).clone()
    return imgs


def lane_cost(images, car_size):
    SCALE = 0.25
    safe_factor = 1.5
    bsize, npred, nchannels, crop_h, crop_w = images.size()
    images = images.view(bsize * npred, nchannels, crop_h, crop_w)

    width, length = car_size[:, 0], car_size[:, 1]  # feet
    width = width * SCALE * (0.3048 * 24 / 3.7)  # pixels
    length = length * SCALE * (0.3048 * 24 / 3.7)  # pixels 

    # Create separable proximity mask
    width.fill_(24 * SCALE / 2)

    max_x = torch.ceil((crop_h - length) / 2)
    #    max_y = torch.ceil((crop_w - width) / 2)
    max_y = torch.ceil(torch.zeros(width.size()).fill_(crop_w) / 2)
    max_x = max_x.view(bsize, 1).expand(bsize, npred).contiguous().view(bsize * npred).cuda()
    max_y = max_y.view(bsize, 1).expand(bsize, npred).contiguous().view(bsize * npred).cuda()
    min_x = max_x
    min_y = torch.ceil(crop_w / 2 - width)  # assumes other._width / 2 = self._width / 2
    min_y = min_y.view(bsize, 1).expand(bsize, npred).contiguous().view(bsize * npred).cuda()
    x_filter = (1 - torch.abs(torch.linspace(-1, 1, crop_h))) * crop_h / 2

    x_filter = x_filter.unsqueeze(0).expand(bsize * npred, crop_h).cuda()
    x_filter = torch.min(x_filter, max_x.view(bsize * npred, 1).expand(x_filter.size()))
    x_filter = (x_filter == max_x.unsqueeze(1).expand(x_filter.size())).float()

    y_filter = (1 - torch.abs(torch.linspace(-1, 1, crop_w))) * crop_w / 2
    y_filter = y_filter.view(1, crop_w).expand(bsize * npred, crop_w).cuda()
    #    y_filter = torch.min(y_filter, max_y.view(bsize * npred, 1))
    y_filter = torch.max(y_filter, min_y.view(bsize * npred, 1))
    y_filter = (y_filter - min_y.view(bsize * npred, 1)) / (max_y.view(bsize * npred, 1) - min_y.view(bsize * npred, 1))
    x_filter = x_filter.cuda()
    y_filter = y_filter.cuda()
    proximity_mask = torch.bmm(x_filter.view(-1, crop_h, 1), y_filter.view(-1, 1, crop_w))
    proximity_mask = proximity_mask.view(bsize, npred, crop_h, crop_w)
    images = images.view(bsize, npred, nchannels, crop_h, crop_w)
    costs = torch.max((proximity_mask * images[:, :, 0].float()).view(bsize, npred, -1), 2)[0]
    return costs.view(bsize, npred), proximity_mask


def offroad_cost(images, proximity_mask):
    bsize, npred, nchannels, crop_h, crop_w = images.size()
    images = images.view(bsize, npred, nchannels, crop_h, crop_w)
    costs = torch.max((proximity_mask * images[:, :, 2].float()).view(bsize, npred, -1), 2)[0]
    return costs.view(bsize, npred)


def proximity_cost(images):
    """
    Compute the cost of getting close to other vessels
    """
    width = images.shape[3]
    bsize, npred,_ ,_ , width = images.shape

    proximity_mask = cost.get_proximity_mask(width).cuda()
    proximity_costs = torch.max((proximity_mask * images[:, :, 1].float()).view(bsize, npred, -1), 2)[0] 

    return proximity_costs, proximity_mask

def land_cost(images):
    """
    Compute the cost of getting close to restrictions of the water way.
    """
    width = images.shape[3]
    bsize, npred,_ ,_ , width = images.shape

    proximity_mask = cost.get_proximity_mask(width).cuda()
    costs = torch.max((proximity_mask * images[:, :, 0].float()).view(bsize, npred, -1), 2)[0] 

    return costs, proximity_mask


def target_cost(input_state, states, target_position, desired_speed, max_speed_influence, stats):
    """
    Compute the target cost of the states.
    Compare the direction of the agent and the optimal direction to the target.
    Add a penalty if the agent is not moving in desired_speed
    t = clamp(sin(delta_angle) + 1/desired_speed * speed  + 1), 0, 1)

    :param input_state: the last input state, normalized
    :param states: the predicted states of the, agent normalized
    :param target: the target position, unnormalized
    :param desired_speed: the desired speed of the agent, unnormalized
    :param max_speed_influence: the maximum influence of the speed on the cost
    :param stats: the stats of the dataset for unnormalization
    """
    # concat input state and states
    next_states = torch.cat((input_state.unsqueeze(1), states), dim=1)[:,:-1]
    # add next state to states
    states_extended = torch.cat((states, next_states), dim=2)
    states_extended = denormalize_states(states, stats)

    optimal_angles = torch.rad2deg(-torch.arctan2(target_position[1] - states_extended[:,:,1], states_extended[:,:,0] - target_position[0])) - 90
    # get difference between current and optimal angle
    angle_diff = torch.abs(optimal_angles - states_extended[:,:,3])
    angle_cost = torch.abs(torch.sin(torch.deg2rad(0.5*angle_diff)))

    # stationiary cost, add if speed is below hyperparameter linearly
    stationary_cost = torch.clamp(- 1/desired_speed * states_extended[:,:,2] + 1, 0, 1)

    # combine costs
    return torch.clamp(angle_cost + max_speed_influence * stationary_cost, 0, 1)

def target_cost_old(states, target, s_mean, s_std):
    """
    Compute the cost of the states with respect to the target
    Use normalized position of agent and target and return the distance
    """
    target_position = target.cuda()
    target_position -= s_mean[:2].cuda()
    target_position /= s_std[:2].cuda()
    # distance to normalized target as cost
    target_costs = torch.norm(states[:,:,:2] - target_position.expand(states.shape[0], states.shape[1],2), dim=2)
    return target_costs



def parse_car_path(path):
    splits = path.split('/')
    time_slot = splits[-2]
    car_id = int(re.findall('car(\d+).pkl', splits[-1])[0])
    data_files = {'trajectories-0400-0415': 0,
                  'trajectories-0500-0515': 1,
                  'trajectories-0515-0530': 2}
    time_slot = data_files[time_slot]
    return time_slot, car_id


def plot_mean_and_CI(mean, lb, ub, color_mean=None, color_shading=None):
    # plot the shaded range of the confidence intervals                                                                                                                                                   
    time_steps = [i + 3 for i in range(len(mean))]
    plt.fill_between(time_steps, ub, lb,
                     color=color_shading, alpha=0.2)
    # plot the mean on top                                                                                                                                                                                
    plt.plot(time_steps, mean, color_mean)


def mean_confidence_interval(data, confidence=0.95):
    n = data.shape[0]
    m, se = numpy.mean(data, 0), scipy.stats.sem(data, 0)
    h = numpy.std(data, 0)
    #    h = se * scipy.stats.t._ppf((1+confidence)/2., n-1)
    return m, m - h, m + h


# Logging function
def log(fname, s):
    if not os.path.isdir(os.path.dirname(fname)):
        os.system(f'mkdir -p {os.path.dirname(fname)}')
    f = open(fname, 'a')
    f.write(f'{str(datetime.now())}: {s}\n')
    f.close()


def combine(x, y, method):
    if method == 'add':
        return x + y
    elif method == 'mult':
        return x * y


def format_losses(loss_i, loss_s, loss_p=None, split='train'):
    log_string = ' '
    log_string += f'{split} loss ['
    log_string += f'i: {loss_i:.5f}, '
    log_string += f's: {loss_s:.5f}, '
    if loss_p is not None:
        log_string += f', p: {loss_p:.5f}'
    log_string += ']'
    return log_string




def save_movie(dirname, images, states, costs=None, actions=None, mu=None, std=None, pytorch=True, raw=False):
    images = images.data if hasattr(images, 'data') else images
    states = states.data if hasattr(states, 'data') else states
    if costs is not None:
        costs = costs.data if hasattr(costs, 'data') else costs
    if actions is not None:
        actions = actions.data if hasattr(actions, 'data') else actions

    os.system('mkdir -p ' + dirname)
    print(f'[saving movie to {dirname}]')
    if mu is not None:
        mu = mu.squeeze()
        std = std.squeeze()
    else:
        mu = actions
    if pytorch:
        images = images.permute(0, 2, 3, 1).cpu().numpy() * 255
    if raw:
        for t in range(images.shape[0]):
            img = images[t]
            img = numpy.uint8(img)
            Image.fromarray(img).save(path.join(dirname, f'im{t:05d}.png'))
        return
    for t in range(images.shape[0]):
        img = images[t]
        img = numpy.concatenate((img, numpy.zeros((24, 24, 3)).astype('float')), axis=0)
        img = numpy.uint8(img)
        pil = Image.fromarray(img).resize((img.shape[1] * 5, img.shape[0] * 5), Image.NEAREST)
        draw = ImageDraw.Draw(pil)

        text = ''
        if states is not None:
            text += f'x: [{states[t][0]:.2f}, {states[t][1]:.2f} \n'
            text += f'dx: {states[t][2]:.2f}, {states[t][3]:.2f}]\n'
        if costs is not None:
            text += f'c: [{costs[t][0]:.2f}, {costs[t][1]:.2f}]\n'
        if actions is not None:
            text += f'a: [{actions[t][0]:.2f}, {actions[t][1]:.2f}]\n'
            x = int(images[t].shape[1] * 5 / 2 - mu[t][1] * 30)
            y = int(images[t].shape[0] * 5 / 2 - mu[t][0] * 30)
            if std is not None:
                ex = max(3, int(std[t][1] * 100))
                ey = max(3, int(std[t][0] * 100))
            else:
                ex, ey = 3, 3
            bbox = (x - ex, y - ey, x + ex, y + ey)
            draw.ellipse(bbox, fill=(200, 200, 200))

        draw.text((10, 130 * 5 - 10), text, (255, 255, 255))
        pil.save(dirname + f'/im{t:05d}.png')


def grad_norm(net):
    total_norm = 0
    for p in net.parameters():
        if p.grad is None:
            pdb.set_trace()
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def read_config(file_path):
    """Read JSON config."""
    json_object = json.load(open(file_path, 'r'))
    return json_object


def log_pdf(z, mu, sigma):
    a = 0.5 * torch.sum(((z - mu) / sigma) ** 2, 1)
    b = torch.log(2 * math.pi * torch.prod(sigma, 1))
    loss = a.squeeze() + b.squeeze()
    return loss


def log_gaussian_distribution(y, mu, sigma):
    Z = 1.0 / ((2.0 * numpy.pi) ** (
            mu.size(2) / 2))  # normalization factor for Gaussians (!!can be numerically unstable)
    result = (y.unsqueeze(1).expand_as(mu) - mu) * torch.reciprocal(sigma)
    result = 0.5 * torch.sum(result * result, 2)
    result += torch.log(2 * math.pi * torch.prod(sigma, 2))
    #    result = torch.exp(result) / (1e-6 + torch.sqrt(torch.prod(sigma, 2)))
    #    result *= oneDivSqrtTwoPI
    return result


def gaussian_distribution(y, mu, sigma):
    oneDivSqrtTwoPI = 1.0 / ((2.0 * numpy.pi) ** (
            mu.size(2) / 2))  # normalization factor for Gaussians (!!can be numerically unstable)
    result = (y.unsqueeze(1).expand_as(mu) - mu) * torch.reciprocal(sigma)
    result = -0.5 * torch.sum(result * result, 2)
    result = torch.exp(result) / (1e-6 + torch.sqrt(torch.prod(sigma, 2)))
    result *= oneDivSqrtTwoPI
    return result


def hinge_loss(u, z):
    bsize = z.size(0)
    nz = z.size(1)
    uexp = u.view(bsize, 1, nz).expand(bsize, bsize, nz).contiguous()
    zexp = z.view(1, bsize, nz).expand(bsize, bsize, nz).contiguous()
    uexp = uexp.view(bsize * bsize, nz)
    zexp = zexp.view(bsize * bsize, nz)
    sim = torch.sum(uexp * zexp, 1).view(bsize, bsize)
    loss = sim - torch.diag(sim).view(-1, 1)
    loss = F.relu(loss)
    loss = torch.mean(loss)
    return loss


# second represents the prior
def kl_criterion(mu1, logvar1, mu2, logvar2):
    # KL( N(mu_1, sigma2_1) || N(mu_2, sigma2_2)) = 
    #   log( sqrt(
    # 
    bsize = mu1.size(0)
    sigma1 = logvar1.mul(0.5).exp()
    sigma2 = logvar2.mul(0.5).exp()
    kld = torch.log(sigma2 / sigma1) + (torch.exp(logvar1) + (mu1 - mu2) ** 2) / (2 * torch.exp(logvar2)) - 1 / 2
    return kld.sum() / bsize


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


# inputs are:
# pi: categorical distribution over mixture components
# mu: means of mixture components
# sigma: variances of mixture components (note, all mixture components are assumed to be diagonal)
# y: points to evaluate the negative-log-likelihood of, under the model determined by these parameters
def mdn_loss_fn(pi, sigma, mu, y, avg=True):
    minsigma = sigma.min().item()
    assert minsigma >= 0, f'{minsigma} < 0'
    c = mu.size(2)
    result = (y.unsqueeze(1).expand_as(mu) - mu) * torch.reciprocal(sigma)
    result = 0.5 * torch.sum(result * result, 2)
    result -= torch.log(pi)
    result += 0.5 * c * math.log(2 * math.pi)
    result += torch.sum(torch.log(sigma), 2)
    result = -result
    result = -log_sum_exp(result, dim=1)
    if avg:
        result = torch.mean(result)
    return result


# embed Z distribution as well as some special z's (ztop) using PCA and tSNE.
# Useful for visualizing predicted z vectors.
def embed(Z, ztop, ndim=3):
    bsize = ztop.shape[0]
    nsamples = ztop.shape[1]
    dim = ztop.shape[2]
    ztop = ztop.reshape(bsize * nsamples, dim)
    Z_all = numpy.concatenate((ztop, Z), axis=0)

    # PCA
    Z_all_pca = decomposition.PCA(n_components=ndim).fit_transform(Z_all)
    ztop_pca = Z_all_pca[0:bsize * nsamples].reshape(bsize, nsamples, ndim)
    Z_pca = Z_all_pca[bsize * nsamples:]
    ztop_only_pca = decomposition.PCA(n_components=3).fit_transform(ztop)

    # Spectral
    Z_all_laplacian = manifold.SpectralEmbedding(n_components=ndim).fit_transform(Z_all)
    ztop_laplacian = Z_all_laplacian[0:bsize * nsamples].reshape(bsize, nsamples, ndim)
    Z_laplacian = Z_all_laplacian[bsize * nsamples:]
    ztop_only_laplacian = manifold.SpectralEmbedding(n_components=3).fit_transform(ztop)

    # Isomap
    Z_all_isomap = manifold.Isomap(n_components=ndim).fit_transform(Z_all)
    ztop_isomap = Z_all_isomap[0:bsize * nsamples].reshape(bsize, nsamples, ndim)
    Z_isomap = Z_all_isomap[bsize * nsamples:]
    ztop_only_isomap = manifold.Isomap(n_components=3).fit_transform(ztop)

    # TSNE
    '''
    Z_all_tsne = TSNE(n_components=2).fit_transform(Z_all)
    ztop_tsne = Z_all_tsne[0:bsize*nsamples].reshape(bsize, nsamples, 2)
    Z_tsne = Z_all_tsne[bsize*nsamples:]
    '''
    #    Z_tsne, ztop_tsne = None, None
    return {'Z_pca': Z_pca, 'ztop_pca': ztop_pca,
            'Z_laplacian': Z_laplacian, 'ztop_laplacian': ztop_laplacian,
            'Z_isomap': Z_isomap, 'ztop_isomap': ztop_isomap,
            'ztop_only_pca': ztop_only_pca,
            'ztop_only_laplacian': ztop_only_laplacian,
            'ztop_only_isomap': ztop_only_isomap}


def parse_command_line(parser=None):
    if parser is None: parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    # data params
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-dataset', type=str, default='i80')
    parser.add_argument('-v', type=int, default=4)
    parser.add_argument('-model', type=str, default='fwd-cnn-vae-fp')
    parser.add_argument('-policy', type=str, default='policy-deterministic')
    parser.add_argument('-model_dir', type=str, default='/workspace/models/')
    parser.add_argument('-ncond', type=int, default=10)
    parser.add_argument('-npred', type=int, default=20)
    parser.add_argument('-layers', type=int, default=3)
    parser.add_argument('-batch_size', type=int, default=8)
    parser.add_argument('-nfeature', type=int, default=256)
    parser.add_argument('-n_hidden', type=int, default=256)
    parser.add_argument('-dropout', type=float, default=0.1, help='regular dropout')
    parser.add_argument('-lrt', type=float, default=0.0001, help='learning rate')
    parser.add_argument('-grad_clip', type=float, default=50.0)
    parser.add_argument('-epoch_size', type=int, default=500)
    parser.add_argument('-n_futures', type=int, default=10)
    parser.add_argument('-u_reg', type=float, default=0.05, help='coefficient of uncertainty regularization term')
    parser.add_argument('-u_hinge', type=float, default=0.5)
    parser.add_argument('-lambda_a', type=float, default=0.0, help='l2 regularization on actions')
    parser.add_argument('-lambda_l', type=float, default=0.2, help='coefficient of land cost')
    parser.add_argument('-lambda_p', type=float, default=0.2, help='coefficient of proximity cost')
    parser.add_argument('-lambda_t', type=float, default=1.0, help='coefficient of target cost')
    parser.add_argument('-target_location', type=array, default=[2244.34601982, 90.73506014], help='target location in environment coordinates')
    parser.add_argument('-desired_speed', type=float, default=2.0, help='desired speed in target cost m/s')
    parser.add_argument('-max_speed_influence', type=float, default=0.2, help='max speed influence in target cost')
    parser.add_argument('-lrt_z', type=float, default=0.0)
    parser.add_argument('-z_updates', type=int, default=0)
    parser.add_argument('-infer_z', action='store_true')
    parser.add_argument('-gamma', type=float, default=0.99)
    parser.add_argument('-learned_cost', action='store_true')
    m1 = 'model=fwd-cnn-vae-fp-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-dropout=0.1-nz=32-' + \
         'beta=1e-06-zdropout=0.5-gclip=5.0-warmstart=1-seed=1.step200000.model'
    m2 = 'model=fwd-cnn-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-dropout=0.1-gclip=5.0-' + \
         'warmstart=0-seed=1.step200000.model'
    m3 = 'model=fwd-cnn-vae-fp-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-dropout=0.1-nz=32-' + \
         'beta=1e-06-zdropout=0.5-gclip=5.0-warmstart=1-seed=1.step400000.model'
    m0 = '/workspace/models/model=fwd-cnn-layers=3-bsize=4-ncond=10-npred=10-lrt=0.0001-nfeature=256-dropout=0.0-gclip=5.0-warmstart=0-seed=1.step400000.model'
    vae1 = '/workspace/models/model=fwd-cnn-vae-fp-layers=3-bsize=16-ncond=30-npred=30-lrt=0.0001-nfeature=256-dropout=0.1-nz=32-beta=1e-06-zdropout=0.5-gclip=5.0-warmstart=1-seed=1.step80000.model'
    parser.add_argument('-mfile', type=str, default=vae1, help='dynamics model used to train the policy network')
    parser.add_argument('-value_model', type=str, default='')
    parser.add_argument('-load_model_file', type=str, default='')
    parser.add_argument('-combine', type=str, default='add')
    parser.add_argument('-debug', action='store_true')
    parser.add_argument('-save_movies', action='store_true', default=False)
    parser.add_argument('-l2reg', type=float, default=0.0)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-enable_tensorboard', action='store_true',
                    help='Enables tensorboard logging.')
    parser.add_argument('-tensorboard_dir', type=str, default='models/policy_networks',
                        help='path to the directory where to save tensorboard log. If passed empty path' \
                             ' no logs are saved.')

    opt = parser.parse_args()
    opt.n_inputs = 4
    opt.n_actions = 2
    opt.height = 100
    opt.width = 100
    opt.h_height = 12
    opt.h_width = 12
    opt.hidden_size = opt.nfeature * opt.h_height * opt.h_width
    return opt


def build_model_file_name(opt):
    if 'vae' in opt.mfile:
        opt.model_file += f'-model=vae'
    if 'zdropout=0.5' in opt.mfile:
        opt.model_file += '-zdropout=0.5'
    elif 'zdropout=0.0' in opt.mfile:
        opt.model_file += '-zdropout=0.0'
    if 'model=fwd-cnn-layers' in opt.mfile:
        opt.model_file += '-deterministic'
    opt.model_file += f'-nfeature={opt.nfeature}'
    opt.model_file += f'-bsize={opt.batch_size}'
    opt.model_file += f'-npred={opt.npred}'
    opt.model_file += f'-ureg={opt.u_reg}'
    opt.model_file += f'-lambdal={opt.lambda_l}'
    opt.model_file += f'-lambdaa={opt.lambda_a}'
    opt.model_file += f'-gamma={opt.gamma}'
    opt.model_file += f'-lrtz={opt.lrt_z}'
    opt.model_file += f'-updatez={opt.z_updates}'
    opt.model_file += f'-inferz={opt.infer_z}'
    opt.model_file += f'-learnedcost={opt.learned_cost}'
    opt.model_file += f'-seed={opt.seed}'
    if opt.value_model == '':
        opt.model_file += '-novalue'

    print(f'[will save as: {opt.model_file}]')
    
def create_tensorboard_writer(opt):
    tensorboard_enabled = opt.tensorboard_dir != '' and opt.enable_tensorboard
    if tensorboard_enabled:
        date_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')
        if hasattr(opt, 'model_file'):
            model_name = os.path.basename(opt.model_file)
        elif hasattr(opt, 'mfile'):
            model_name = os.path.basename(opt.policy_model) # eval_policy has mfile
        else:
            raise AttributeError("options doesn't contain neither model_file nor mfile field")
        script_name = os.path.splitext(sys.argv[0])[0]
        tensorboard_log_dir = os.path.join(opt.tensorboard_dir, f'tb_log_{script_name}_{model_name}_{date_str}')
        print('saving tensorboard logs to', tensorboard_log_dir)
        writer = SummaryWriter(log_dir=tensorboard_log_dir)
        return writer
    else:
        return None
        

def normalize_actions(actions, model_stats):
    '''
    normalite action to be in the range of the model

    :param action: action to be normalized
    :param model_stats: model statistics
    '''
    actions -= model_stats['a_mean'].view(1, 2).expand(actions.size())
    actions /= model_stats['a_std'].view(1, 2).expand(actions.size())
    return actions

def denormalize_actions(actions, model_stats):
    '''
    denormalite action to be in the range of the model

    :param action: action to be denormalized
    :param model_stats: model statistics
    '''
    actions = torch.clone(actions)
    if actions.dim() == 2:
        actions = actions.unsqueeze(0)
    if actions.dim() == 1:
        actions = actions.unsqueeze(0).unsqueeze(0)
    actions *= model_stats['a_std'].view(1,1,2).expand(actions.size()).cuda()
    actions += model_stats['a_mean'].view(1,1,2).expand(actions.size()).cuda()
    return actions

def denormalize_states(states, model_stats):
    '''
    denominalise state

    :param state: state to denormalise
    :param model_stats: model stats
    '''
    # detach states
    states = torch.clone(states)
    if states.dim() == 2:
        states = states.unsqueeze(0)
    if states.dim() == 1:
        states = states.unsqueeze(0).unsqueeze(0)
    states *= model_stats['s_std'].view(1, 1, 4).expand(states.size()).cuda()
    states += model_stats['s_mean'].view(1, 1, 4).expand(states.size()).cuda()
    return states

def normalize_inputs(images, states, stats, device='cuda'):
    images = images.clone().float().div_(255.0)
    states -= stats['s_mean'].view(1, 4).expand(states.size())
    states /= stats['s_std'].view(1, 4).expand(states.size())
    if images.dim() == 4:  # if processing single vehicle
        images = images.to(device).unsqueeze(0)
        states = states.to(device).unsqueeze(0)
    return images, states

def test_action(model, inputs, action, timesteps = 40, sample_z = False, normalizeAction = True):
    if normalizeAction:
        action = normalize_actions(action, model.stats).cuda()
    batch_size = inputs[0].size(0)
    # reshape action to be batch_size x 1 x 2
    if action.shape[0] != batch_size:
        actions = action.expand(batch_size, 1, 2)
    else:
        actions = action
    # sample z
    if sample_z:
        z = model.sample_z(batch_size).cuda()
    else:
        z = torch.zeros(batch_size, 32).cuda()
    # add batch dimension
    input_images = inputs[0].cuda()
    input_states = inputs[1].cuda()
    pred_images = []
    pred_states = []

    for i in range(timesteps):
        pred_image, pred_state = model.forward_single_step(input_images, input_states, actions, z)

        # Auto regress: enqueue output as new element of the input
        input_images = torch.cat((input_images[:, 1:], pred_image), 1)
        input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)

        # predict next states
        pred_images.append(pred_image)
        pred_states.append(pred_state)

    pred_images_action = torch.cat(pred_images, 1)
    pred_states_action = torch.stack(pred_states, 1)
    return [pred_images_action, pred_states_action]

def plot_denormalized_states(stats, inputs, targets, preds, batch_no):
    """
    Plot denormalited states

    Used for evaluation
    
    """
    input_states = denormalize_states(inputs[1][batch_no], stats )[0].cpu().detach().numpy()
    print(input_states[-1])
    target_states = denormalize_states(targets[1][batch_no],stats )[0].cpu().detach().numpy()
    if preds is not None:
        pred_states = denormalize_states(preds[1][batch_no], stats )[0].cpu().detach().numpy()
        plot_helpers.plot_states([(input_states, 'input states', 'g'), (pred_states, 'predicted states', 'r'), (target_states, 'target states', 'm')], margin=400)
    else:
        plot_helpers.plot_states([(input_states, 'input states', 'g'), (target_states, 'target states', 'm')], margin=400)
    

def save_batch(inputs, actions, targets, preds, stats, batch_no, save_path):
    """
    Save batch of images and states to file
    """
    batch = {
        'inputs': inputs,
        'actions': actions,
        'targets': targets,
        'preds': preds,
        'stats': stats,
        'batch_no': batch_no
    }
    torch.save(batch, save_path)

def load_batch(path, device='cuda:0'):
    """
    Load batch of images and states from file
    """
    batch = torch.load(path, map_location=device)
    inputs = batch['inputs']
    actions = batch['actions']
    targets = batch['targets']
    batch_no = batch['batch_no']
    preds = batch['preds']
    stats = batch['stats']
    return inputs, actions, targets, preds, stats, batch_no

def integrate_state(state, action, t=1):
    """
    Integrate state using action
    """
    x, y, speed, direction = state
    speed = speed + action[0] * t
    direction = direction + action[1] * t
    x2 = x + speed * np.sin(np.radians(direction)) * t
    y2 = y - speed * np.cos(np.radians(direction)) * t
    return [x2, y2, speed, direction]

def calculate_state_mses(prediction, target, stats):
    """
    Calculate the mean squared error of the states
    Will denormalize first.
    ----
    :param prediction: predicted states tensor normalized
    :param target: target states tensor normalized
    """

    batch_size = prediction[1].shape[0]
    # denormalize states
    prediction_states = denormalize_states(prediction[1], stats)
    target_states = denormalize_states(target[1], stats)
    # calculate MSE
    mse = torch.mean((prediction_states[:batch_size] - target_states[:batch_size])**2)
    mse_position = torch.mean((prediction_states[:batch_size, :2] - target_states[:batch_size, :2])**2)
    mse_velocity = torch.mean((prediction_states[:batch_size, 2] - target_states[:batch_size, 2])**2)
    mse_direction = torch.mean((prediction_states[:batch_size, 3] - target_states[:batch_size, 3])**2)
    return mse, mse_position, mse_velocity, mse_direction

def calculate_image_mse(prediction, target, stats):
    """
    Calculate the mean squared error of the images
    ----
    :param prediction: predicted images tensor normalized
    :param target: target images tensor normalized
    """
    batch_size = prediction[0].shape[0]
    # calculate MSE
    mse = torch.mean((prediction[0] - target[0][:batch_size])**2)
    return mse
        

def get_hypeparameters(filename):
    """
    Get models hyperparameters from filename
    """
    # get batch_size
    batch_size = int(re.search('bsize=(\d+)', filename).group(1))
    # get npred
    npred = int(re.search('npred=(\d+)', filename).group(1))
    # get ncond
    ncond = int(re.search('ncond=(\d+)', filename).group(1))
    # get steps
    steps = int(re.search('step(\d+)', filename).group(1))
    return batch_size, npred, ncond, steps

def fit_to_model(t, m):
    """
    Size inputs to match the models input size
    ----
    t: tensor, trajectory: inputs, actions, targets, synthetic_actions, synthetic_target_states
    m: model, dict containing model_path
    """
    batch_size, npred, ncond, steps = get_hypeparameters(m['model_path'])
    t = list(copy.deepcopy(t))
    # fit to batch_size
    t[0][0] = t[0][0][:batch_size]
    t[0][1] = t[0][1][:batch_size]
    t[1] = t[1][:batch_size]
    t[2][0] = t[2][0][:batch_size]
    t[2][1] = t[2][1][:batch_size]
    if (len(t) > 3):
        t[3] = t[3][:batch_size]
        t[4] = t[4][:batch_size]
    return tuple(t)

def apply_random_actions(trajectory, dataloader):
    """
    Apply random actions to trajectories
    ----
    :param trajectories: list of trajectory = input, action, target
    :param stats: dict of stats
    """
    npred = trajectory[2][1].shape[1]
    # get starting state
    starting_states =  trajectory[0][1][:,-1]

    zero_action = [0, 0]
    target_states = []
    actions = []
    for b in range(starting_states.shape[0]):
        batch_states = [starting_states[b]]
        batch_actions = []
        action = dataloader.sample_syntetic_action(starting_states[b].cpu().detach().numpy())
        # sample how long to continue action
        nsteps = np.random.randint(1, npred)
        # integrate action for nsteps
        for i in range(nsteps):
            next_state = integrate_state(batch_states[-1].cpu().detach().numpy(), action)
            batch_states.append(torch.tensor(next_state).to(starting_states.device))
            batch_actions.append(torch.tensor(action).to(starting_states.device))
        for i in range(npred - nsteps):
            next_state = integrate_state(batch_states[-1].cpu().detach().numpy(), zero_action)
            batch_states.append(torch.tensor(next_state).to(starting_states.device))
            batch_actions.append(torch.tensor(zero_action).to(starting_states.device))
        batch_actions = torch.stack(batch_actions)
        batch_states = torch.stack(batch_states[1:])
        target_states.append(batch_states)
        actions.append(batch_actions)
    target_states = torch.stack(target_states)
    actions = torch.stack(actions)
    return actions, target_states


