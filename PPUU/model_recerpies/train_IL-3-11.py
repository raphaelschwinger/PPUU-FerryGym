import torch, numpy, argparse, pdb, os, time, math, random, re
# add /workspace to path
import sys, os
sys.path.append('/workspace')
import utils
from dataloader import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import models, planning
import importlib

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#################################################
# Train an action-conditional forward model
#################################################

parser = argparse.ArgumentParser()
# data params
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-dataset', type=str, default='i80')
parser.add_argument('-v', type=int, default=4)
parser.add_argument('-model', type=str, default='fwd-cnn')
parser.add_argument('-policy', type=str, default='policy-deterministic')
parser.add_argument('-model_dir', type=str, default='models/')
parser.add_argument('-ncond', type=int, default=10)
parser.add_argument('-npred', type=int, default=10)
parser.add_argument('-batch_size', type=int, default=4)
parser.add_argument('-layers', type=int, default=3)
parser.add_argument('-nfeature', type=int, default=256)
parser.add_argument('-n_hidden', type=int, default=256)
parser.add_argument('-beta', type=float, default=0.0, help='weight coefficient of prior loss')
parser.add_argument('-p_dropout', type=float, default=0.0, help='set z=0 with this probability')
parser.add_argument('-dropout', type=float, default=0.0, help='regular dropout')
parser.add_argument('-nz', type=int, default=2)
parser.add_argument('-n_mixture', type=int, default=10)
parser.add_argument('-context_dim', type=int, default=2)
parser.add_argument('-actions_subsample', type=int, default=4)
parser.add_argument('-lrt', type=float, default=0.0001)
parser.add_argument('-grad_clip', type=float, default=1.0)
parser.add_argument('-epoch_size', type=int, default=500)
parser.add_argument('-curriculum_length', type=int, default=16)
parser.add_argument('-zeroact', type=int, default=0)
parser.add_argument('-warmstart', type=int, default=0)
parser.add_argument('-targetprop', type=int, default=0)
parser.add_argument('-loss_c', type=int, default=0)
parser.add_argument('-lambda_c', type=float, default=0.0)
parser.add_argument('-lambda_h', type=float, default=0.0)
parser.add_argument('-lambda_lane', type=float, default=0.1)
parser.add_argument('-lrt_traj', type=float, default=0.5)
parser.add_argument('-niter_traj', type=int, default=10)
parser.add_argument('-gamma', type=float, default=1.0)
#parser.add_argument('-mfile', type=str, default='model=fwd-cnn-vae-fp-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-dropout=0.1-nz=32-beta=1e-06-zdropout=0.5-gclip=5.0-warmstart=1-seed=1.step200000.model')
parser.add_argument('-mfile', type=str, default='model=fwd-cnn-layers=3-bsize=4-ncond=10-npred=10-lrt=0.0001-nfeature=256-dropout=0.0-gclip=5.0-warmstart=0-seed=1.step400000.model')
parser.add_argument('-load_model_file', type=str, default='')
parser.add_argument('-combine', type=str, default='add')
parser.add_argument('-debug', action='store_true')
parser.add_argument('-test_only', type=int, default=0)
parser.add_argument('-enable_tensorboard', action='store_true',
                    help='Enables tensorboard logging.')
parser.add_argument('-tensorboard_dir', type=str, default='models/policy_networks/il',
                    help='path to the directory where to save tensorboard log. If passed empty path' \
                         ' no logs are saved.')
opt = parser.parse_args()

opt.n_inputs = 4
opt.n_actions = 2
opt.height = 100
opt.width = 100
opt.h_height = 12
opt.h_width = 12
opt.hidden_size = opt.nfeature*opt.h_height*opt.h_width


os.system('mkdir -p ' + opt.model_dir + '/policy_networks/')

random.seed(opt.seed)
numpy.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)



opt.model_file = f'{opt.model_dir}/policy_networks/IL-3-11/'

opt.model_file += f'mbil-IL'


print(f'[will save as: {opt.model_file}]')


model = torch.load('/workspace/results/20-10-fm-big-df/model=fwd-cnn-layers=3-bsize=16-ncond=10-npred=20-lrt=0.0001-nfeature=256-dropout=0.1-action-fix-gclip=5.0-warmstart=0-seed=1.step44000.model')
if type(model) is dict: model = model['model']
model.create_policy_net(opt)
model.opt.actions_subsample = opt.actions_subsample
optimizer = optim.Adam(model.policy_net.parameters(), opt.lrt)
n_iter = 0


if opt.actions_subsample == -1:
    opt.context_dim = 0

model.intype('gpu')
model.cuda()


print('[loading data]')
dataloader = DataLoader(opt, data_dir='/workspace/data/rev-moenk2/training/', dataframe='df_2022-04-01 12:00:01-straight-to-target.pkl' )

model.stats = dataloader.get_stats()


# training and testing functions. We will compute several losses:
# loss_i: images
# loss_s: states
# loss_c: costs
# loss_p: prior (optional)

def compute_loss(action, target_action):
    loss_a = F.mse_loss(action, target_action)
    return loss_a

def train(nbatches, npred):
    gamma_mask = torch.Tensor([opt.gamma**t for t in range(npred)]).view(1, -1).cuda()
    model.eval()
    model.policy_net.train()
    total_loss_a, total_loss_policy,  n_updates = 0, 0, 0
    for i in range(nbatches):
        optimizer.zero_grad()
        inputs, actions, targets = dataloader.get_batch_fm('train', npred)
        action, _, _, _ = model.policy_net(inputs[0], inputs[1])
        loss_a = compute_loss(action, actions[:,0,:])
#        
        loss_policy = loss_a
        if not math.isnan(loss_policy.item()):
            loss_policy.backward()
            torch.nn.utils.clip_grad_norm(model.policy_net.parameters(), opt.grad_clip)
            optimizer.step()
            total_loss_policy += loss_policy.item()
            n_updates += 1
        else:
            print('warning, NaN')

        del inputs, actions, targets

    total_loss_policy /= n_updates
    return total_loss_a, total_loss_policy


def test(nbatches, npred):
    gamma_mask = torch.Tensor([opt.gamma**t for t in range(npred)]).view(1, -1).cuda()
    model.eval()
    total_loss_a, total_loss_policy, n_updates = 0, 0, 0
    for i in range(nbatches):
        inputs, actions, targets = dataloader.get_batch_fm('valid', npred)
        action, _, _, _ = model.policy_net(inputs[0], inputs[1])
        loss_a = compute_loss(action,actions[:,0,:] )
#        
        loss_policy = loss_a
        if not math.isnan(loss_policy.item()):
            loss_policy.backward()
            torch.nn.utils.clip_grad_norm(model.policy_net.parameters(), opt.grad_clip)
            optimizer.step()
            total_loss_policy += loss_policy.item()
            n_updates += 1
        else:
            print('warning, NaN')

        del inputs, actions, targets

        total_loss_policy /= n_updates
    return total_loss_a, total_loss_policy


# set by hand to fit on 12gb GPU
def get_batch_size(npred):
    if npred <= 15:
        return 4
    elif npred <= 50:
        return 32
    elif npred <= 100:
        return 16
    elif npred <= 200:
        return 8
    elif npred <= 400:
        return 4
    elif npred <= 800:
        return 2
    else:
        return 1


if opt.test_only == 1:
    print('[testing]')
    valid_losses = test(10, 200)
else:

    writer = utils.create_tensorboard_writer(opt)

    print('[training]')
    utils.log(opt.model_file + '.log', f'[job name: {opt.model_file}]')
    npred = opt.npred if opt.npred != -1 else 16

    for i in range(500):
        bsize = get_batch_size(npred)
        dataloader.opt.batch_size = bsize
        train_losses = train(opt.epoch_size, npred)
        valid_losses = test(int(opt.epoch_size / 2), npred)
        n_iter += opt.epoch_size
        model.intype('cpu')
        torch.save({'model': model,
                    'optimizer': optimizer.state_dict(),
                    'opt': opt,
                    'n_iter': n_iter},
                   opt.model_file + '.model')
        model.intype('gpu')

        if writer is not None:
            writer.add_scalar('Loss/train_state_img', train_losses[1], i)

            
            writer.add_scalar('Loss/validation_policy', valid_losses[1], i)
            

        log_string = f'step {n_iter} |  bsize {bsize} | esize {opt.epoch_size} | '
        log_string += utils.format_losses(train_losses[0], train_losses[1], split='train')
        log_string += utils.format_losses(valid_losses[0], valid_losses[1], split='valid')
        print(log_string)
        utils.log(opt.model_file + '.log', log_string)
        

    if writer is not None:
        writer.close()


