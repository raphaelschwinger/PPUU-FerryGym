import math
# add /workspace to path
import sys, os
sys.path.append('/workspace')
from collections import OrderedDict

import numpy
import os
import ipdb
import random
import torch
import torch.optim as optim
from os import path

import planning
import utils
from dataloader import DataLoader

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#################################################
# Train a policy / controller
#################################################

opt = utils.parse_command_line()

# Create file_name
opt.model_file = path.join(opt.model_dir, 'policy_networks', 'MPUR-' + opt.policy)
utils.build_model_file_name(opt)

os.system('mkdir -p ' + path.join(opt.model_dir, 'policy_networks'))

random.seed(opt.seed)
numpy.random.seed(opt.seed)
torch.manual_seed(opt.seed)

# Define default device
opt.device = torch.device('cuda' if torch.cuda.is_available() and not opt.no_cuda else 'cpu')
if torch.cuda.is_available() and opt.no_cuda:
    print('WARNING: You have a CUDA device, so you should probably run without -no_cuda')

# load the model
print('[loading data]')
dataloader = DataLoader(opt, data_dir='/workspace/data/rev-moenk2/training/', dataframe='df_2022-04-01 12:00:01.pkl')

model_dir = '/workspace/models/policy_networks/3-11-MPUR-vae-mix'
model_path = '/workspace/results/26-10-vae-mix/model=fwd-cnn-vae-fp-layers=3-bsize=16-ncond=10-npred=20-lrt=0.0001-nfeature=256-dropout=0.1-nz=32-beta=1e-06-zdropout=0.5-gclip=5.0-warmstart=1-seed=1.step56000.model'
print('Loading model from ' + model_path)
if path.exists(model_path):
    model = torch.load(model_path)
elif path.exists(opt.mfile):
    model = torch.load(opt.mfile)
else:
    raise runtime_error(f'couldn\'t find file {opt.mfile}')

if not hasattr(model.encoder, 'n_channels'):
    model.encoder.n_channels = 3

if type(model) is dict: model = model['model']
model.opt.lambda_l = opt.lambda_l  # used by planning.py/compute_uncertainty_batch
model.opt.lambda_t = opt.lambda_t  # used by planning.py/compute_uncertainty_batch
model.opt.lambda_p = opt.lambda_p

model.opt.target_location = torch.Tensor(opt.target_location).cuda()  # used by planning.py/compute_uncertainty_batch
model.opt.desired_speed = opt.desired_speed # used by planning.py/compute_uncertainty_batch
model.opt.max_speed_influence = opt.max_speed_influence # used by planning.py/compute_uncertainty_batch

if opt.value_model != '':
    value_function = torch.load(path.join(opt.model_dir, 'value_functions', opt.value_model)).to(opt.device)
    model.value_function = value_function

# Create policy
model.create_policy_net(opt)
optimizer = optim.Adam(model.policy_net.parameters(), opt.lrt)  # POLICY optimiser ONLY!

# add dropout
model.opt.dropout = opt.dropout

# Load normalisation stats

model.stats = dataloader.get_stats()  # used by planning.py/compute_uncertainty_batch
if 'ten' in opt.mfile:
    p_z_file = opt.model_dir + opt.mfile + '.pz'
    p_z = torch.load(p_z_file)
    model.p_z = p_z

# Send to GPU if possible
model.to(opt.device)
model.policy_net.stats_d = {}
# for k, v in stats.items():
#     if isinstance(v, torch.Tensor):
#         model.policy_net.stats_d[k] = v.to(opt.device)

if opt.learned_cost:
    print('[loading cost regressor]')
    model.cost = torch.load(path.join(opt.model_dir, opt.mfile + '.cost.model'))['model']

# set dropout

def set_dropout(model, drop_rate=0):
    for name, child in model.named_children():
        if isinstance(child, torch.nn.Dropout):
            child.p = drop_rate
        set_dropout(child, drop_rate=drop_rate)

set_dropout(model, drop_rate=opt.dropout)

model.train()
model.opt.u_hinge = opt.u_hinge
planning.estimate_uncertainty_stats(model, dataloader, n_batches=50, npred=opt.npred)
model.eval()


def start(what, nbatches, npred):
    train = True if what is 'train' else False
    model.train()
    model.policy_net.train()
    n_updates, grad_norm = 0, 0
    total_losses = dict(
        proximity=0,
        uncertainty=0,
        land=0,
        target=0,
        action=0,
        policy=0,
    )
    for j in range(nbatches):
        inputs, actions, targets = dataloader.get_batch_fm(what, npred)
        pred, actions = planning.train_policy_net_mpur(
            model, inputs, targets, n_models=10, lrt_z=opt.lrt_z,
            n_updates_z=opt.z_updates, infer_z=opt.infer_z
        )
        pred['policy'] = opt.u_reg * pred['uncertainty'] + \
                         opt.lambda_l * pred['land'] + \
                         opt.lambda_a * pred['action'] + \
                         opt.lambda_p * pred['proximity'] + \
                         opt.lambda_t * pred['target']

        if not math.isnan(pred['policy'].item()):
            if train:
                optimizer.zero_grad()
                pred['policy'].backward()  # back-propagation through time!
                grad_norm += utils.grad_norm(model.policy_net).item()
                torch.nn.utils.clip_grad_norm_(model.policy_net.parameters(), opt.grad_clip)
                optimizer.step()
            for loss in total_losses: total_losses[loss] += pred[loss].item()
            n_updates += 1
        else:
            print('warning, NaN')  # Oh no... Something got quite fucked up!
            ipdb.set_trace()

        if j == 0 and opt.save_movies and train:
            # save videos of normal and adversarial scenarios
            for b in range(opt.batch_size):
                state_img = pred['state_img'][b]
                state_vct = pred['state_vct'][b]
                utils.save_movie(opt.model_file + f'.mov/sampled/mov{b}', state_img, state_vct, None, actions[b])

        del inputs, actions, targets, pred

    for loss in total_losses: total_losses[loss] /= n_updates
    if train: print(f'[avg grad norm: {grad_norm / n_updates:.4f}]')
    return total_losses


print('[training]')
utils.log(opt.model_file + '.log', f'[job name: {opt.model_file}]')
n_iter = 0
losses = OrderedDict(
    p='proximity',
    l='land',
    t='target',
    u='uncertainty',
    a='action',
    π='policy',
)

writer = utils.create_tensorboard_writer(opt)

for i in range(500):
    train_losses = start('train', opt.epoch_size, opt.npred)
    with torch.no_grad():  # Torch, please please please, do not track computations :)
        valid_losses = start('valid', opt.epoch_size // 2, opt.npred)

    if writer is not None:
        for key in train_losses:
            writer.add_scalar(f'Loss/train_{key}', train_losses[key], i)
        for key in valid_losses:
            writer.add_scalar(f'Loss/valid_{key}', valid_losses[key], i)

    n_iter += opt.epoch_size
    model.to('cpu')
    torch.save(dict(
        model=model,
        optimizer=optimizer.state_dict(),
        opt=opt,
        n_iter=n_iter,
    ), opt.model_file + '.model')
    if (n_iter / opt.epoch_size) % 10 == 0:
        torch.save(dict(
            model=model,
            optimizer=optimizer.state_dict(),
            opt=opt,
            n_iter=n_iter,
        ), opt.model_file + f'step{n_iter}.model')

    model.to(opt.device)

    log_string = f'step {n_iter} | '
    log_string += 'train: [' + ', '.join(f'{k}: {train_losses[v]:.4f}' for k, v in losses.items()) + '] | '
    log_string += 'valid: [' + ', '.join(f'{k}: {valid_losses[v]:.4f}' for k, v in losses.items()) + ']'
    print(log_string)
    utils.log(opt.model_file + '.log', log_string)

if writer is not None:
    writer.close()
