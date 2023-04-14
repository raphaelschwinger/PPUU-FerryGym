import utils
import numpy as np
import torch

# test batch_size, npred and ncond
def check_dimensions(inputs, actions, targets, batch_size, npred, ncond):
    if inputs[0].shape[0] != batch_size or inputs[1].shape[0] != batch_size or targets[0].shape[0] != batch_size or targets[1].shape[0] != batch_size or actions.shape[0] != batch_size:
        print('batch_size is incorrect')
        print('batch_size: ', batch_size)
        return False
    if inputs[0].shape[1] != ncond or inputs[1].shape[1] != ncond or targets[0].shape[1] != npred or targets[1].shape[1] != npred:
        print('npred and ncond or incorrect')
        print('npred: ', npred)
        print('ncond: ', ncond)
        return False
    return True

# get rows from state
def get_rows_from_states(dataloader, states, df):
    # denormalize states
    states = utils.denormalize_states(states, dataloader.get_stats()).detach().cpu().numpy()
    N = 0.0005
    rows = np.empty((states.shape[0], states.shape[1]), dtype=object)
    for i in range(states.shape[0]):
        multiple_offset = 0
        mmsi = ''
        for j in range(states.shape[1]):
            # get state
            state = states[i][j]
            # get rows from state
            row = df.loc[(df['x'] > state[0]-N) & (df['x'] < state[0]+N) & (df['y'] > state[1]-N) & (df['y'] < state[1]+N) & (df['speed_calculated'] > state[2]-N) & (df['speed_calculated'] < state[2]+N) & (df['direction'] > state[3]-N) & (df['direction'] < state[3]+N)]
            if len(row) == 1:
                rows[i][j] = row.iloc[0]
                multiple_offset = 0
            elif len(row) > 1 or len(row) == 0:
                return None
    return rows

def check_mmsi(rows):
    for i in range(rows.shape[0]):
        for j in range(rows.shape[1]):
            mmsi = rows[i][j]['mmsi']
            for k in range(rows.shape[1]):
                if rows[i][k].shape[0] > 0:
                    if rows[i][k]['mmsi'] != mmsi:
                        print('mmsi is not the same')
                        return False
    return True

    # check if all ships in batch are 1s apart
def check_time(rows):
    for i in range(rows.shape[0]):
        for j in range(rows.shape[1] - 1):
            if rows[i][j]['datetime'].timestamp() != rows[i][j+1]['datetime'].timestamp() - 1:
                print('indicies: ', i, j)
                print('time difference is not 1s: ', rows[i][j]['datetime'], rows[i][j+1]['datetime'])
                print('df row: ', rows[i][j].name)
                print('next row: ', rows[i][j+1].name)
                return False
    return True

# check if new speed = speed + acceleration*t
def check_speed_integration(rows, N=0.15):
    for i in range(rows.shape[0]):
        for j in range(rows.shape[1] - 1):
            target_speed = rows[i][j+1]['speed_calculated']
            speed = rows[i][j]['speed_calculated']
            speed_ = speed + rows[i][j]['acceleration']
            if target_speed < (speed_ - N) or target_speed > (speed_ + N):
                print('indicies: ', i, j)
                print('speed integration is not correct')
                print('current speed: ',speed)
                print('acceleration: ', rows[i][j]['acceleration'])
                print('target speed: ', target_speed)
                print('calculated speed: ', speed_)
                print('df row: ', rows[i][j].name)
                print('next row: ', rows[i][j+1].name)
                return False
    return True

def direction_integration(direction, direction_change):
    direction = direction + direction_change
    direction = direction % 360
    return direction

def check_direction_integration(rows, N=0.15):
    for i in range(rows.shape[0]):
        for j in range(rows.shape[1] - 1):
            target_direction = rows[i][j+1]['direction']
            direction = rows[i][j]['direction']
            direction_change = rows[i][j]['direction_change']
            direction_ = direction_integration(direction, direction_change)
            if target_direction < direction_integration(direction_, -N) or target_direction > direction_integration(direction_,N):
                print('indicies: ', i, j)
                print('direction integration is not correct')
                print('direction: ', direction)
                print('target direction: ', target_direction)
                print('calculated direction: ', direction_)
                print('direction_change: ', direction_change)
                print('df row: ', rows[i][j].name)
                return False
    return True

def check_position_integration(rows, N=0.01):
    for i in range(rows.shape[0]):
        for j in range(rows.shape[1] - 1):
            speed = rows[i][j]['speed_calculated']
            direction = rows[i][j]['direction']
            x = rows[i][j]['x']
            y = rows[i][j]['y']
            x2 = rows[i][j+1]['x']
            y2 = rows[i][j+1]['y']
            acceleration = rows[i][j]['acceleration']
            direction_change = rows[i][j]['direction_change']
            x2_, y2_, _, _ = utils.integrate_state([x, y, speed, direction], [acceleration, direction_change])
            # check if new position is correct
            if x2 > x2_ + N or x2 < x2_ - N or y2 > y2_ + N or y2 < y2_ - N:
                print('indicies: ', i, j)
                print('position integration is not correct')
                print(rows[i][j].name,': ', x, y)
                print(rows[i][j+1].name,': ', x2, y2)
                print('calculated: ', x2_, y2_)
                print('speed: ', speed)
                print('direction: ', direction)
                return False
    return True

# check if state, action integration is correct
def check_state_action_integration(dataloader, rows, actions, ncond, N=0.01):
    # denormalize
    actions = utils.denormalize_actions(actions, dataloader.get_stats())
    for i in range(rows.shape[0]):
        for j in range(ncond -1, rows.shape[1]-1):
            speed = rows[i][j]['speed_calculated']
            direction = rows[i][j]['direction']
            x = rows[i][j]['x']
            y = rows[i][j]['y']
            x2 = rows[i][j+1]['x']
            y2 = rows[i][j+1]['y']
            action = actions[i][j-ncond+1]
            x2_, y2_, _, _ = utils.integrate_state([x, y, speed, direction], action)
            if x2 > x2_ + N or x2 < x2_ - N or y2 > y2_ + N or y2 < y2_ - N:
                print('indicies: ', i, j)
                print('position integration is not correct')
                print(rows[i][j].name,': ', x, y)
                print(rows[i][j+1].name,': ', x2, y2)
                print('calculated: ', x2_, y2_)
                print('speed: ', speed)
                print('direction: ', direction)
                print('action: ', i, j-ncond,  action)
                return False
    return True

def check_state_action_target_integration(dataloader, inputs, actions, targets, ncond, N=0.01):
    input_states = utils.denormalize_states(inputs[1], dataloader.get_stats())
    target_states = utils.denormalize_states(targets[1], dataloader.get_stats())
    actions = utils.denormalize_actions(actions, dataloader.get_stats()).cpu().numpy()
    # concat input_states and target_states
    states = torch.cat((input_states, target_states), dim=1).cpu().numpy()
    for i in range(states.shape[0]):
        for j in range(ncond -1, states.shape[1]-1):
            speed = states[i][j][2]
            direction = states[i][j][3]
            x = states[i][j][0]
            y = states[i][j][1]
            x2 = states[i][j+1][0]
            y2 = states[i][j+1][1]
            action = actions[i][j-ncond+1]
            x2_, y2_, _, _ = utils.integrate_state([x, y, speed, direction], action)
            if x2 > x2_ + N or x2 < x2_ - N or y2 > y2_ + N or y2 < y2_ - N:
                print('indicies: ', i, j)
                print('position integration is not correct')
                print(rows[i][j].name,': ', x, y)
                print(rows[i][j+1].name,': ', x2, y2)
                print('calculated: ', x2_, y2_)
                print('speed: ', speed)
                print('direction: ', direction)
                print('action: ', i, j-ncond,  action)
                return False
    return True