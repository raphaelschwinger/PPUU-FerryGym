import copy
import datetime
import re
import matplotlib.pyplot as plt
import numpy as np
import gym
import sys
import matplotlib.animation as animation
import torch

sys.path.append('/workspace')

from ferrygym import FerryGymEnv

def load_log_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        regex = re.compile(r".*step\s(\d+)\s\|\s+train\sloss\s\[i:\s(\w+\.*\w*),\ss:\s(\w+\.*\w*),\s,\sp:\s(\w+\.*\w*)]\svalid\sloss\s\[i:\s(\w+\.*\w*),\ss:\s(\w+\.*\w*),\s,\sp:\s(\w+\.*\w*)\]", re.IGNORECASE)
        steps = []
        train_losses_i = []
        train_losses_s = []
        validation_losses_i = []
        validation_losses_s = []
        for line in lines:
            match = regex.match(line)
            if match:
                steps.append(int(match.group(1)))
                train_losses_i.append(float(match.group(2)))
                train_losses_s.append(float(match.group(3)))
                validation_losses_i.append(float(match.group(5)))
                validation_losses_s.append(float(match.group(6)))
        data = dict(
            steps=steps,
            train_losses_i=train_losses_i,
            train_losses_s=train_losses_s,
            validation_losses_i=validation_losses_i,
            validation_losses_s=validation_losses_s,
        )
    return data

def load_MPUR_log_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        regex = re.compile(r".*step\s(\d+)\s\|\s+train:\s\[p:\s((\w+\.*\w*)),\sl:\s((\w+\.*\w*)),\st:\s((\w+\.*\w*)),\su:\s((\w+\.*\w*)),\sa:\s((\w+\.*\w*)),\s.:\s((\w+\.*\w*))]\s\|\svalid:\s\[p:\s((\w+\.*\w*)),\sl:\s((\w+\.*\w*)),\st:\s((\w+\.*\w*)),\su:\s((\w+\.*\w*)),\sa:\s((\w+\.*\w*)),\s.:\s((\w+\.*\w*))\]", re.IGNORECASE)
        steps = []
        train_losses_p = []
        train_losses_l = []
        train_losses_t = []
        train_losses_u = []
        train_losses_a = []
        train_losses_pi = []
        validation_losses_p = []
        validation_losses_l = []
        validation_losses_t = []
        validation_losses_u = []
        validation_losses_a = []
        validation_losses_pi = []
        for line in lines:
            match = regex.match(line)
            if match:
                steps.append(int(match.group(1)))
                train_losses_p.append(float(match.group(2)))
                train_losses_l.append(float(match.group(3)))
                train_losses_t.append(float(match.group(4)))
                train_losses_u.append(float(match.group(5)))
                train_losses_a.append(float(match.group(6)))
                train_losses_pi.append(float(match.group(7)))
                validation_losses_p.append(float(match.group(8)))
                validation_losses_l.append(float(match.group(9)))
                validation_losses_t.append(float(match.group(10)))
                validation_losses_u.append(float(match.group(11)))
                validation_losses_a.append(float(match.group(12)))
                validation_losses_pi.append(float(match.group(13)))

        data = dict(
            steps=steps,
            train_losses_p=train_losses_p,
            train_losses_l=train_losses_l,
            train_losses_t=train_losses_t,
            train_losses_u=train_losses_u,
            train_losses_a=train_losses_a,
            train_losses_pi=train_losses_pi,
            validation_losses_p=validation_losses_p,
            validation_losses_l=validation_losses_l,
            validation_losses_t=validation_losses_t,
            validation_losses_u=validation_losses_u,
            validation_losses_a=validation_losses_a,
            validation_losses_pi=validation_losses_pi,
        )
    return data

# plot losses, i losses with scale on the left, s losses with scale on the right
def plot_MPUR_losses(data):
    fig, ax1 = plt.subplots()
    ax1.plot(data['steps'], data['train_losses_p'], label='train p')
    ax1.plot(data['steps'], data['train_losses_l'], label='train l')
    ax1.plot(data['steps'], data['train_losses_u'], label='train u')
    ax1.plot(data['steps'], data['validation_losses_p'], label='validation p')
    ax1.plot(data['steps'], data['validation_losses_l'], label='validation l')
    ax1.plot(data['steps'], data['validation_losses_u'], label='validation u')
    ax1.set_xlabel('step')
    ax1.set_ylabel('loss')

    ax2 = ax1.twinx()
    ax2.plot(data['steps'], data['train_losses_t'], label='train t')
    ax2.plot(data['steps'], data['train_losses_a'], label='train a')
    ax2.plot(data['steps'], data['train_losses_pi'], label='train pi')
    ax2.plot(data['steps'], data['validation_losses_t'], label='validation t')
    ax2.plot(data['steps'], data['validation_losses_a'], label='validation a')
    ax2.plot(data['steps'], data['validation_losses_pi'], label='validation pi')
    plt.legend()

# plot losses, i losses with scale on the left, s losses with scale on the right
def plot_losses(data):
    fig, ax1 = plt.subplots()
    ax1.plot(data['steps'], data['train_losses_i'], label='train i')
    ax1.plot(data['steps'], data['validation_losses_i'], label='validation i')
    ax1.set_xlabel('step')
    ax1.set_ylabel('loss i')

    ax2 = ax1.twinx()
    ax2.plot(data['steps'], data['train_losses_s'], label='train s')
    ax2.plot(data['steps'], data['validation_losses_s'], label='validation s')
    plt.legend()


def animate_images(images, num_frames, saveVideo = False):
    """animate a set of images

    Parameters
    ----------
    images : list
        list of images as tensor possible on gpu
    num_frames : int
        number of frames to animate
    """
    images = images.detach().permute(0, 2, 3, 1).cpu()
    images = torch.rot90(images, 1, [1, 2])
    fig, ax = plt.subplots()
    ims = []
    for i in range(num_frames):
       image = images[i] 
       im = ax.imshow(image, animated=True)
       if i == 0:
           ax.imshow(images[0])  # show an initial one first
       ims.append([im])   
    
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                   repeat_delay=1000)
    plt.gca().invert_yaxis()
    if saveVideo:
        ani.save(saveVideo)  # type: ignore
    plt.show() 
    # To save the animation, use e.g.
    #
    #
    # or
    #
    # writer = animation.FFMpegWriter(
    #     fps=15, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save("movie.mp4", writer=writer)

def plot_image(image):
    """
    plot a single image
    """

    image = torch.rot90(image.permute(1,2,0), 1, [0, 1]).detach().cpu().numpy()
    plt.imshow(image)
    plt.gca().invert_yaxis()

def load_environment(render_observation=True):
    kwargs = dict(
        generate_training_data=False,
        data_directory='/workspace/data/rev-moenk/training/',
        df_filename='2022-04-10-13->14.pkl',
        render_observation=render_observation,
    )

    gym.register(
        id="FerryGym-v0",
        entry_point="FerryGymEnv:FerryGymEnv",
        kwargs=kwargs,
    )


    """
    hardcoded policy
    this policy turns right a couple of timesteps to avoid hitting ground
    and then goes straight to the target
    """

    env = gym.make('FerryGym-v0')
    return env

class NQueue:
    def __init__(self, n):
        self.n = n
        self.queue = []

    def push(self, item):
        self.queue.append(item)
        if len(self.queue) > self.n:
            self.queue.pop(0)

    # make sure to always return N values, if there are less than N values in the queue, fill up with the first value
    def get(self):
        if len(self.queue) < self.n:
            return [self.queue[0]] * (self.n - len(self.queue)) + self.queue
        else:
            return self.queue

def run_policy(env, policy,stack_obs=0, max_steps=500, nr_episodes=1, returnObservations=False, resetOptions=None):
    """
    Run policy in environment

    Parameters
    ----------
    env : gym environment
    policy : policy
    stack_obs : int number of input states (ncond)
    max_steps : int maximum number of steps per episode
    nr_episodes : int number of episodes to run
    returnObservations : bool if true, return full observations
    resetOptions : dict options to reset environment

    Returns
    -------
    List of episodes containing:
        [0]: episode number
        [1]: rewards for each step
        [2]: agent positions for each step
        [3]: done_info for each step
        [4]: agent speeds for each step
        [5]: agent directions for each step
        [6]: agent actions for each step
        [7]: agent observations for each step
    """

    totals = []
    episodes = []
    observationQueue = NQueue(stack_obs)
    for episode in range(nr_episodes):
        agent_positions = []
        agent_speeds = []
        agent_directions = []
        observations = []
        actions = []
        episode_rewards = 0
        if resetOptions is None:
            obs = env.reset()
        else:
            obs = env.reset(options=resetOptions)
        agent_positions.append(copy.copy(obs['agent_position']))
        agent_speeds.append(copy.copy(obs['agent_speed']))
        agent_directions.append(copy.copy(obs['agent_direction']))
        actions.append(copy.copy([0,0]))
        done_info = None
        if stack_obs > 0:
                observationQueue.push(obs)
                obs = observationQueue.get()
        for step in range(max_steps):
            action = policy(obs, step)
            obs, reward, done, info = env.step(action)
            agent_speeds.append(copy.copy(obs['agent_speed']))
            agent_directions.append(copy.copy(obs['agent_direction']))
            actions.append(copy.copy(action))
            if returnObservations:
                observations.append(copy.deepcopy(obs))
            if stack_obs > 0:
                observationQueue.push(obs)
                obs = observationQueue.get()
                agent_positions.append(copy.copy(obs[-1]['agent_position']))
            else:
                agent_positions.append(copy.copy(obs['agent_position']))
            episode_rewards += reward
            if done:
                break
            # env.render()
        done_info = info
        totals.append(episode_rewards)
        print('Episode: {}, Total reward: {}'.format(episode, episode_rewards))
        # append episode nr, total reward and agent positions
        episodes.append((episode, episode_rewards, agent_positions, done_info, agent_speeds, agent_directions, actions, observations))
    return episodes


def load_nn_policy(model_path, stats):
    """
    Load a policy from a model file

    Parameters
    ----------
    model_path : str path to model file
    stats : dict statistics of the training data (use dataloader.get_stats())
    """
    model = torch.load(model_path)['model']
    model.cuda()
    # load stats
    model.policy_net.stats = stats
    # move s_mean s_std, a_std and a_mean to cuda
    model.policy_net.stats['s_mean'] = model.policy_net.stats['s_mean'].cuda()
    model.policy_net.stats['s_std'] = model.policy_net.stats['s_std'].cuda()
    model.policy_net.stats['a_mean'] = model.policy_net.stats['a_mean'].cuda()
    model.policy_net.stats['a_std'] = model.policy_net.stats['a_std'].cuda()

    # build state from observation
    def policy(obs, step):
        # build input images from list of observations each with a dict named 'neighborhood'
        images = torch.tensor([obs['neighborhood'] for obs in obs]).to(device='cuda', non_blocking=True)
        # permute to match dataset
        # print(images.shape)
        # images = images.permute(0, 2, 1, 3)
        # mirror images to match images in training set
        images = torch.flip(images, (1,))
        # print(images.shape)
        # 
        # build input state
        # todo fix speed (2D to 1D)
        states = torch.tensor([[obs['agent_position'][0],
         obs['agent_position'][1],
         obs['agent_speed'],
         obs['agent_direction'],
        ] for obs in obs]).to(device='cuda', non_blocking=True)
        # cast to float32
        states = states.float()
        pred_action = model.policy_net(images,states, context=None, sample=True,
                normalize_inputs=True, normalize_outputs=True)
        actions_array = pred_action[0].cpu().numpy()
        acceloration = actions_array[0][0]
        steering = actions_array[0][1]
        return acceloration, steering 
    return policy

def get_rates(policy_results):
    """
    Get the rates of the policies

    returns success rate, collision rate, dataset_rate and timeout_rate
    """
    successes = 0
    collisions = 0
    land = 0
    dataset = 0
    timeout = 0
    for result in policy_results:
        if result[3]['status'] == 'success':
            successes += 1
        if result[3]['status'] == 'collision':
            collisions += 1
        if result[3]['status'] == 'land':
            land += 1
        if result[3]['status'] == 'dataset ended':
            dataset += 1
        if result[3]['status'] == 'running':
            timeout += 1

    success_rate = successes / len(policy_results)
    collision_rate = collisions / len(policy_results)
    land_rate = land / len(policy_results)
    dataset_rate = dataset / len(policy_results)
    timeout_rate = timeout / len(policy_results)
    return success_rate, collision_rate, land_rate, dataset_rate, timeout_rate

def get_average_path_length(policy_results):
    """
    Get the average path length of the policies
    """
    path_lengths = []
    for result in policy_results:
        points = np.array(result[2])
        path_lengths.append(np.sum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))))
    return np.mean(path_lengths)

def get_mean_distance_to_other_ships(policy_results):
    """
    Get the mean distance to other ships
    """
    return np.mean([result[3]['distance'] for result in policy_results])
