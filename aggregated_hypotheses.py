# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import random

# from common.arguments import parse_args
from common.arguments_edited import parse_args  # H:10, K:20
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno
import math

from einops import rearrange, repeat
from copy import deepcopy

from common.camera import *
import collections

from common.diffusionpose import *

from common.loss import *
from common.generators import ChunkedGenerator_Seq, UnchunkedGenerator_Seq

from common.utils import *
from common.logging import Logger
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime, timedelta
import time

# cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

args = parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if args.evaluate != '':
    description = "Evaluate!"
elif args.evaluate == '':

    description = "Train!"

# initial setting
TIMESTAMP = "{0:%Y%m%dT%H-%M-%S/}".format(datetime.now())
# tensorboard
if not args.nolog:
    writer = SummaryWriter(args.log + '_' + TIMESTAMP)
    writer.add_text('description', description)
    writer.add_text('command', 'python ' + ' '.join(sys.argv))
    # logging setting
    logfile = os.path.join(args.log + '_' + TIMESTAMP, 'logging.log')
    sys.stdout = Logger(logfile)
print(description)
print('python ' + ' '.join(sys.argv))
print("CUDA Device Count: ", torch.cuda.device_count())
print(args)

manualSeed = 1
random.seed(manualSeed)
torch.manual_seed(manualSeed)
np.random.seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)

# if not assign checkpoint path, Save checkpoint file into log folder
if args.checkpoint_d3dp == '':
    args.checkpoint_d3dp = args.log + '_' + TIMESTAMP
try:
    # Create checkpoint directory if it does not exist
    os.makedirs(args.checkpoint_d3dp)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint_d3dp)

# dataset loading
print('Loading dataset...')
dataset_path = 'data/data_3d_' + args.dataset + '.npz'
if args.dataset == 'h36m':
    from common.h36m_dataset import Human36mDataset

    dataset = Human36mDataset(dataset_path)
elif args.dataset.startswith('humaneva'):
    from common.humaneva_dataset import HumanEvaDataset

    dataset = HumanEvaDataset(dataset_path)
elif args.dataset.startswith('custom'):
    from common.custom_dataset import CustomDataset

    dataset = CustomDataset('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz')
else:
    raise KeyError('Invalid dataset')

print('Preparing data...')
for subject in dataset.subjects():
    for action in dataset[subject].keys():
        anim = dataset[subject][action]

        if 'positions' in anim:
            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                pos_3d[:, 1:] -= pos_3d[:, :1]  # Remove global offset, but keep trajectory in first position
                positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d

print('Loading 2D detections...')
keypoints = np.load('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz', allow_pickle=True)
keypoints_metadata = keypoints['metadata'].item()
keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
keypoints = keypoints['positions_2d'].item()

###################
for subject in dataset.subjects():
    assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
    for action in dataset[subject].keys():
        assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(
            action, subject)
        if 'positions_3d' not in dataset[subject][action]:
            continue

        for cam_idx in range(len(keypoints[subject][action])):

            # We check for >= instead of == because some videos in H3.6M contain extra frames
            mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
            assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

            if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                # Shorten sequence
                keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

        assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])

for subject in keypoints.keys():
    for action in keypoints[subject]:
        for cam_idx, kps in enumerate(keypoints[subject][action]):
            # Normalize camera frame
            cam = dataset.cameras()[subject][cam_idx]
            kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
            keypoints[subject][action][cam_idx] = kps

subjects_train = args.subjects_train.split(',')
subjects_semi = [] if not args.subjects_unlabeled else args.subjects_unlabeled.split(',')
if not args.render:
    subjects_test = args.subjects_test.split(',')
else:
    subjects_test = [args.viz_subject]


def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):  # Iterate across cameras
                out_poses_2d.append(poses_2d[i])

            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)):  # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None

    stride = args.downsample
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i]) // stride * subset) * stride)
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start + n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start + n_frames:stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]

    return out_camera_params, out_poses_3d, out_poses_2d


action_filter = None if args.actions == '*' else args.actions.split(',')
if action_filter is not None:
    print('Selected actions:', action_filter)

cameras_valid, poses_valid, poses_valid_2d = fetch(subjects_test, action_filter)

# set receptive_field as number assigned
receptive_field = args.number_of_frames
print('INFO: Receptive field: {} frames'.format(receptive_field))
if not args.nolog:
    writer.add_text(args.log + '_' + TIMESTAMP + '/Receptive field', str(receptive_field))
pad = (receptive_field - 1) // 2  # Padding on each side
min_loss = args.min_loss
width = cam['res_w']
height = cam['res_h']
num_joints = keypoints_metadata['num_joints']

model_pos_train = D3DP(args, joints_left, joints_right, is_train=True)
model_pos_test_temp = D3DP(args, joints_left, joints_right, is_train=False)
model_pos = D3DP(args, joints_left, joints_right, is_train=False, num_proposals=args.num_proposals,
                 sampling_timesteps=args.sampling_timesteps)

causal_shift = 0
model_params = 0
for parameter in model_pos.parameters():
    model_params += parameter.numel()
print('INFO: Trainable parameter count:', model_params / 1000000, 'Million')
if not args.nolog:
    writer.add_text(args.log + '_' + TIMESTAMP + '/Trainable parameter count', str(model_params / 1000000) + ' Million')

# make model parallel
if torch.cuda.is_available():
    model_pos = nn.DataParallel(model_pos)
    model_pos = model_pos.cuda()
    model_pos_train = nn.DataParallel(model_pos_train)
    model_pos_train = model_pos_train.cuda()
    model_pos_test_temp = nn.DataParallel(model_pos_test_temp)
    model_pos_test_temp = model_pos_test_temp.cuda()

if args.resume or args.evaluate:
    chk_filename = os.path.join(args.checkpoint_d3dp, args.resume if args.resume else args.evaluate)
    # chk_filename = args.resume or args.evaluate
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    print('This model was trained for {} epochs'.format(checkpoint['epoch']))
    model_pos_train.load_state_dict(checkpoint['model_pos'], strict=False)
    model_pos.load_state_dict(checkpoint['model_pos'], strict=False)

test_generator = UnchunkedGenerator_Seq(cameras_valid, poses_valid, poses_valid_2d,
                                        pad=pad, causal_shift=causal_shift, augment=False,
                                        kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                        joints_right=joints_right)
print('INFO: Testing on {} frames'.format(test_generator.num_frames()))
if not args.nolog:
    writer.add_text(args.log + '_' + TIMESTAMP + '/Testing Frames', str(test_generator.num_frames()))


def eval_data_prepare(receptive_field, inputs_2d, inputs_3d):
    assert inputs_2d.shape[:-1] == inputs_3d.shape[:-1], "2d and 3d inputs shape must be same! " + str(
        inputs_2d.shape) + str(inputs_3d.shape)
    inputs_2d_p = torch.squeeze(inputs_2d)
    inputs_3d_p = torch.squeeze(inputs_3d)

    if inputs_2d_p.shape[0] / receptive_field > inputs_2d_p.shape[0] // receptive_field:
        out_num = inputs_2d_p.shape[0] // receptive_field + 1
    elif inputs_2d_p.shape[0] / receptive_field == inputs_2d_p.shape[0] // receptive_field:
        out_num = inputs_2d_p.shape[0] // receptive_field

    eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
    eval_input_3d = torch.empty(out_num, receptive_field, inputs_3d_p.shape[1], inputs_3d_p.shape[2])

    for i in range(out_num - 1):
        eval_input_2d[i, :, :, :] = inputs_2d_p[i * receptive_field:i * receptive_field + receptive_field, :, :]
        eval_input_3d[i, :, :, :] = inputs_3d_p[i * receptive_field:i * receptive_field + receptive_field, :, :]
    if inputs_2d_p.shape[0] < receptive_field:
        from torch.nn import functional as F
        pad_right = receptive_field - inputs_2d_p.shape[0]
        inputs_2d_p = rearrange(inputs_2d_p, 'b f c -> f c b')
        inputs_2d_p = F.pad(inputs_2d_p, (0, pad_right), mode='replicate')
        # inputs_2d_p = np.pad(inputs_2d_p, ((0, receptive_field-inputs_2d_p.shape[0]), (0, 0), (0, 0)), 'edge')
        inputs_2d_p = rearrange(inputs_2d_p, 'f c b -> b f c')
    if inputs_3d_p.shape[0] < receptive_field:
        pad_right = receptive_field - inputs_3d_p.shape[0]
        inputs_3d_p = rearrange(inputs_3d_p, 'b f c -> f c b')
        inputs_3d_p = F.pad(inputs_3d_p, (0, pad_right), mode='replicate')
        inputs_3d_p = rearrange(inputs_3d_p, 'f c b -> b f c')
    eval_input_2d[-1, :, :, :] = inputs_2d_p[-receptive_field:, :, :]
    eval_input_3d[-1, :, :, :] = inputs_3d_p[-receptive_field:, :, :]

    return eval_input_2d, eval_input_3d


def aggregation_train_data():
    # load checkpoint d3dp
    checkpoint_d3dp = torch.load(os.path.join(args.checkpoint_d3dp, args.evaluate),
                                 map_location=lambda storage, loc: storage)
    model_pos.load_state_dict(checkpoint_d3dp['model_pos'], strict=False)
    model_pos.eval()
    cameras_train, poses_train, poses_train_2d = fetch(subjects_train, action_filter, subset=args.subset)

    # get training data
    train_generator = ChunkedGenerator_Seq(args.batch_size // args.stride, cameras_train, poses_train, poses_train_2d,
                                           args.number_of_frames,
                                           pad=pad, causal_shift=causal_shift, shuffle=True,
                                           augment=args.data_augmentation,
                                           kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                           joints_right=joints_right)

    model_pos_train.train()
    N = 0
    iteration = 0

    num_batches = train_generator.batch_num()

    return_predictions = False

    epoch_loss_3d_pos = torch.zeros(args.sampling_timesteps).cuda()
    epoch_loss_3d_pos_h = torch.zeros(args.sampling_timesteps).cuda()
    epoch_loss_3d_pos_mean = torch.zeros(args.sampling_timesteps).cuda()
    epoch_loss_3d_pos_select = torch.zeros(args.sampling_timesteps).cuda()

    for cameras_train, batch_3d, batch_2d in train_generator.next_epoch():

        if iteration % 1000 == 0:
            print("%d/%d" % (iteration, num_batches))

        if cameras_train is not None:
            cameras_train = torch.from_numpy(cameras_train.astype('float32'))
        inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
        inputs_2d = torch.from_numpy(batch_2d.astype('float32'))

        if torch.cuda.is_available():
            inputs_3d = inputs_3d.cuda()
            inputs_2d = inputs_2d.cuda()
            if cameras_train is not None:
                cameras_train = cameras_train.cuda()
        inputs_traj = inputs_3d[:, :, :1].clone()
        inputs_3d[:, :, 0] = 0

        # Predict 3D poses
        predicted_3d_pos = model_pos(inputs_2d, inputs_3d)

        # predicted_3d_pos = torch.stack(predicted_3d_pos, dim=1)

        predicted_3d_pos[:, :, :, :, 0] = 0

        if return_predictions:
            return predicted_3d_pos.squeeze().cpu().numpy()

        # 2D reprojection
        b_sz, t_sz, h_sz, f_sz, j_sz, c_sz = predicted_3d_pos.shape
        # inputs_traj.unsqueeze(0).unsqueeze(2).repeat(5, 1, 5, 1, 17, 1)
        inputs_traj_single_all = inputs_traj.unsqueeze(1).unsqueeze(1).repeat(1, t_sz, h_sz, 1, 1, 1)
        predicted_3d_pos_abs_single = predicted_3d_pos + inputs_traj_single_all  # add trajectory
        predicted_3d_pos_abs_single = predicted_3d_pos_abs_single.reshape(b_sz * t_sz * h_sz * f_sz, j_sz, c_sz)
        cam_single_all = cameras_train.repeat(t_sz * h_sz * f_sz, 1)
        reproject_2d = project_to_2d(predicted_3d_pos_abs_single, cam_single_all)
        reproject_2d = reproject_2d.reshape(b_sz, t_sz, h_sz, f_sz, j_sz, 2)

        # Compute errors
        error_j_best = mpjpe_diffusion_all_min(predicted_3d_pos, inputs_3d)  # J-Best
        error_p_best = mpjpe_diffusion(predicted_3d_pos, inputs_3d)  # P-Best
        error_p_agg = mpjpe_diffusion_all_min(predicted_3d_pos, inputs_3d, mean_pos=True)  # P-Agg
        error_j_agg = mpjpe_diffusion_reproj(predicted_3d_pos, inputs_3d, reproject_2d, inputs_2d)  # J-Agg

        epoch_loss_3d_pos += inputs_3d.shape[0] * inputs_3d.shape[1] * error_j_best.clone()
        epoch_loss_3d_pos_h += inputs_3d.shape[0] * inputs_3d.shape[1] * error_p_best.clone()
        epoch_loss_3d_pos_mean += inputs_3d.shape[0] * inputs_3d.shape[1] * error_p_agg.clone()
        epoch_loss_3d_pos_select += inputs_3d.shape[0] * inputs_3d.shape[1] * error_j_agg.clone()

        N += inputs_3d.shape[0] * inputs_3d.shape[1]

        iteration += 1
        # if iteration == 2:
        #     break

    e1 = (epoch_loss_3d_pos / N) * 1000  # J-best
    e1_h = (epoch_loss_3d_pos_h / N) * 1000  # P-Best
    e1_mean = (epoch_loss_3d_pos_mean / N) * 1000  # p-agg
    e1_select = (epoch_loss_3d_pos_select / N) * 1000  # j-agg

    print("====================== 10 Hypotheses Aggregation Train set======================")
    log_path = os.path.join('H10_train_hypotheses_aggregation.txt')
    f = open(log_path, mode='a')
    f.write("======================10 Hypotheses Aggregation Train set======================\n")
    for t in range(e1.shape[0]):
        print('step %d : Protocol #1 Error (MPJPE) J_Best:' % t, e1[t].item(), 'mm')
        f.write('step %d : Protocol #1 Error (MPJPE) J_Best: %f mm\n' % (t, e1[t].item()))
        print('step %d : Protocol #1 Error (MPJPE) P_Best:' % t, e1_h[t].item(), 'mm')
        f.write('step %d : Protocol #1 Error (MPJPE) P_Best: %f mm\n' % (t, e1_h[t].item()))
        print('step %d : Protocol #1 Error (MPJPE) P_Agg:' % t, e1_mean[t].item(), 'mm')
        f.write('step %d : Protocol #1 Error (MPJPE) P_Agg: %f mm\n' % (t, e1_mean[t].item()))
        print('step %d : Protocol #1 Error (MPJPE) J_Agg:' % t, e1_select[t].item(), 'mm')
        f.write('step %d : Protocol #1 Error (MPJPE) J_Agg: %f mm\n' % (t, e1_select[t].item()))

    f.write('----------\n')

    f.close()

    # print("Done!!!")


def aggregation_test_data():
    with torch.no_grad():
        # load checkpoint d3dp
        checkpoint_d3dp = torch.load(os.path.join(args.checkpoint_d3dp, args.evaluate), map_location=lambda storage, loc: storage)
        model_pos.load_state_dict(checkpoint_d3dp['model_pos'])
        model_pos.eval()
        N = 0
        iteration = 0

        return_predictions = False

        epoch_loss_3d_pos = torch.zeros(args.sampling_timesteps).cuda()
        epoch_loss_3d_pos_h = torch.zeros(args.sampling_timesteps).cuda()
        epoch_loss_3d_pos_mean = torch.zeros(args.sampling_timesteps).cuda()
        epoch_loss_3d_pos_select = torch.zeros(args.sampling_timesteps).cuda()

        if not args.no_eval:
            # Evaluate on test set
            for cam, batch, batch_2d in test_generator.next_epoch():
                inputs_3d = torch.from_numpy(batch.astype('float32'))
                inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                cam = torch.from_numpy(cam.astype('float32'))

                ##### apply test-time-augmentation (following Videopose3d)
                inputs_2d_flip = inputs_2d.clone()
                inputs_2d_flip[:, :, :, 0] *= -1
                inputs_2d_flip[:, :, kps_left + kps_right, :] = inputs_2d_flip[:, :, kps_right + kps_left, :]

                ##### convert size
                inputs_3d_p = inputs_3d
                inputs_2d, inputs_3d = eval_data_prepare(receptive_field, inputs_2d, inputs_3d_p)
                inputs_2d_flip, _ = eval_data_prepare(receptive_field, inputs_2d_flip, inputs_3d_p)

                if torch.cuda.is_available():
                    inputs_3d = inputs_3d.cuda()
                    inputs_2d = inputs_2d.cuda()
                    inputs_2d_flip = inputs_2d_flip.cuda()
                    cam = cam.cuda()

                inputs_traj = inputs_3d[:, :, :1].clone()
                inputs_3d[:, :, 0] = 0

                bs = args.batch_size
                total_batch = (inputs_3d.shape[0] + bs - 1) // bs

                for batch_cnt in range(total_batch):

                    if (batch_cnt + 1) * bs > inputs_3d.shape[0]:
                        inputs_2d_single = inputs_2d[batch_cnt * bs:]
                        inputs_2d_flip_single = inputs_2d_flip[batch_cnt * bs:]
                        inputs_3d_single = inputs_3d[batch_cnt * bs:]
                        inputs_traj_single = inputs_traj[batch_cnt * bs:]
                    else:
                        inputs_2d_single = inputs_2d[batch_cnt * bs:(batch_cnt + 1) * bs]
                        inputs_2d_flip_single = inputs_2d_flip[batch_cnt * bs:(batch_cnt + 1) * bs]
                        inputs_3d_single = inputs_3d[batch_cnt * bs:(batch_cnt + 1) * bs]
                        inputs_traj_single = inputs_traj[batch_cnt * bs:(batch_cnt + 1) * bs]

                # predicted_3d_pos = model_pos(inputs_2d, inputs_3d,
                #                                        input_2d_flip=inputs_2d_flip)  # b, t, h, f, j, c

                predicted_3d_pos = model_pos(inputs_2d_single, inputs_3d_single,
                                                     input_2d_flip=inputs_2d_flip_single)  # b, t, h, f, j, c

                # predicted_3d_pos = torch.stack(predicted_3d_pos, dim=1) # use when augment = false

                predicted_3d_pos[:, :, :, :, 0] = 0

                if return_predictions:
                    return predicted_3d_pos.squeeze().cpu().numpy()

                # 2D reprojection
                b_sz, t_sz, h_sz, f_sz, j_sz, c_sz = predicted_3d_pos.shape
                # inputs_traj.unsqueeze(0).unsqueeze(2).repeat(5, 1, 5, 1, 17, 1)
                inputs_traj_single_all = inputs_traj_single.unsqueeze(1).unsqueeze(1).repeat(1, t_sz, h_sz, 1, 1, 1)
                predicted_3d_pos_abs_single = predicted_3d_pos + inputs_traj_single_all  # add trajectory
                predicted_3d_pos_abs_single = predicted_3d_pos_abs_single.reshape(b_sz * t_sz * h_sz * f_sz, j_sz, c_sz)
                cam_single_all = cam.repeat(b_sz*t_sz*h_sz*f_sz, 1)
                reproject_2d = project_to_2d(predicted_3d_pos_abs_single, cam_single_all)
                reproject_2d = reproject_2d.reshape(b_sz, t_sz, h_sz, f_sz, j_sz, 2)

                # Compute errors
                error_j_best = mpjpe_diffusion_all_min(predicted_3d_pos, inputs_3d_single)  # J-Best
                error_p_best = mpjpe_diffusion(predicted_3d_pos, inputs_3d_single)  # P-Best
                error_p_agg = mpjpe_diffusion_all_min(predicted_3d_pos, inputs_3d_single, mean_pos=True)  # P-Agg
                error_j_agg = mpjpe_diffusion_reproj(predicted_3d_pos, inputs_3d_single, reproject_2d, inputs_2d_single)  # J-Agg

                epoch_loss_3d_pos += inputs_3d_single.shape[0] * inputs_3d_single.shape[1] * error_j_best.clone()
                epoch_loss_3d_pos_h += inputs_3d_single.shape[0] * inputs_3d_single.shape[1] * error_p_best.clone()
                epoch_loss_3d_pos_mean += inputs_3d_single.shape[0] * inputs_3d_single.shape[1] * error_p_agg.clone()
                epoch_loss_3d_pos_select += inputs_3d_single.shape[0] * inputs_3d_single.shape[1] * error_j_agg.clone()

                N += inputs_3d_single.shape[0] * inputs_3d_single.shape[1]

                iteration += 1
                # if iteration == 2:
                #     break

            e1 = (epoch_loss_3d_pos / N) * 1000  # J-best
            e1_h = (epoch_loss_3d_pos_h / N) * 1000  # P-Best
            e1_mean = (epoch_loss_3d_pos_mean / N) * 1000  # p-agg
            e1_select = (epoch_loss_3d_pos_select / N) * 1000  # j-agg

            print("======================10 Hypotheses Aggregation Test set======================")
            log_path = os.path.join('H10K20_test_hypotheses_aggregation_augment.txt')
            f = open(log_path, mode='a')
            f.write("======================10 Hypotheses Aggregation Test set======================\n")
            for t in range(e1.shape[0]):
                print('step %d : Protocol #1 Error (MPJPE) J_Best:' % t, e1[t].item(), 'mm')
                f.write('step %d : Protocol #1 Error (MPJPE) J_Best: %f mm\n' % (t, e1[t].item()))
                print('step %d : Protocol #1 Error (MPJPE) P_Best:' % t, e1_h[t].item(), 'mm')
                f.write('step %d : Protocol #1 Error (MPJPE) P_Best: %f mm\n' % (t, e1_h[t].item()))
                print('step %d : Protocol #1 Error (MPJPE) P_Agg:' % t, e1_mean[t].item(), 'mm')
                f.write('step %d : Protocol #1 Error (MPJPE) P_Agg: %f mm\n' % (t, e1_mean[t].item()))
                print('step %d : Protocol #1 Error (MPJPE) J_Agg:' % t, e1_select[t].item(), 'mm')
                f.write('step %d : Protocol #1 Error (MPJPE) J_Agg: %f mm\n' % (t, e1_select[t].item()))

            f.write('----------\n')

            f.close()

            # print("Done!!!")


if __name__ == "__main__":

    # # Generate hypotheses for training set
    # print("\n=== Aggregation hypotheses for training set ===\n")
    # start_train = time.time()
    # aggregation_train_data()
    # Training_time = (time.time() - start_train) / 60
    # print("Time spend on generate training:", Training_time)

    # Generate hypotheses for testing set
    print("\n=== H10 K20 Aggregation hypotheses for testing set ===\n")
    start_test = time.time()
    aggregation_test_data()
    Testing_time = (time.time() - start_test) / 60
    print("Time spend on generate testing:", Testing_time)
