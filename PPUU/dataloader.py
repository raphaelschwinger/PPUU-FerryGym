import sys
import numpy, random, pdb, math, pickle, glob, time, os, re
import torch
import pandas as pd
import numpy as np
# add current directory to path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import ppuu_utils


"""
Inspired by https://github.com/Atcold/pytorch-PPUU/blob/master/dataloader.py

The DataLoader has the following responsibilities:
- Load the data from the data directory
- Split into train, validation, and test sets
- Provide statistics for normalization              get_stats()
- Provide a function to get a batch of data         get_batch_fm()
- Provide a function to get a batch synthetic data  get_syntetic_batch_fm()
"""


class DataLoader:
    def __init__(self, opt, data_dir='/workspace/data/rev-moenk2/training/', dataframe='df_2022-04-01 12:00:01.pkl',image_dir='/workspace/data/rev-moenk2/training/images/', single_shard=False, seedRandom=False):
        if opt.debug:
            single_shard = True
        self.opt = opt
        self.random = random.Random()
        if not seedRandom:
            self.random.seed(12345)  # use this so that the same batches will always be picked

        self.df =  pd.read_pickle(data_dir + dataframe)
        self.image_dir = image_dir
        self.images = []
        self.actions = []
        self.costs = []
        self.states = []
        self.ids = []
        zeroNeighbourhoodPath =  os.path.join(os.path.dirname(os.path.abspath(__file__)), "empty_neighbourhood.npy")
        self.zero_neighbourhood = torch.tensor(np.load(zeroNeighbourhoodPath)).permute(2,0,1)

        if opt.loadImagesInMemory:
            self.df = self.load_images_in_memory(self.image_dir) 


        # preprocess data for ncond
        N = self.opt.ncond
        self.df.sort_values(by=['mmsi', 'datetime'], ignore_index=True, inplace=True)
        # add new column named 'index_column'
        self.df['index_column'] = self.df.index
        self.df = self.df.groupby('mmsi').apply(lambda x: x.assign(group_index = range(len(x))))

        # add group_count column that states the number of groups
        self.df['group_count'] = self.df.groupby('mmsi').ngroup()

        # add new column counting the rows where the group_index is greater than N
        self.df['train_index'] = self.df.apply(lambda row: row.index_column - N*(1+row.group_count) if row.group_index > N else None, axis=1)


        self.n_episodes = int(self.df['train_index'].max())
        print(f'Number of episodes: {self.n_episodes}')
        splits_path = data_dir + '/splits.pth'
        if os.path.exists(splits_path):
            print(f'[loading data splits: {splits_path}]')
            self.splits = torch.load(splits_path)
            self.train_indx = self.splits.get('train_indx')
            self.valid_indx = self.splits.get('valid_indx')
            self.test_indx = self.splits.get('test_indx')
        else:
            print('[generating data splits]')
            rgn = numpy.random.RandomState(0)
            perm = rgn.permutation(self.n_episodes)
            n_train = int(math.floor(self.n_episodes * 0.8))
            n_valid = int(math.floor(self.n_episodes * 0.1))
            self.train_indx = perm[0 : n_train]
            self.valid_indx = perm[n_train : n_train + n_valid]
            self.test_indx = perm[n_train + n_valid :]
            torch.save(dict(
                train_indx=self.train_indx,
                valid_indx=self.valid_indx,
                test_indx=self.test_indx,
            ), splits_path)

        # compute stats
        self.ac_mean = self.df['acceleration'].mean()
        self.ac_std = self.df['acceleration'].std()
        self.dc_mean = self.df['direction_change'].mean()
        self.dc_std = self.df['direction_change'].std()
        self.x_mean = self.df['x'].mean()
        self.x_std = self.df['x'].std()
        self.y_mean = self.df['y'].mean()
        self.y_std = self.df['y'].std()
        self.d_mean = self.df['direction'].mean()
        self.d_std = self.df['direction'].std()
        self.sc_mean = self.df['speed_calculated'].mean()
        self.sc_std = self.df['speed_calculated'].std()

        self.a_mean = torch.tensor([self.ac_mean, self.dc_mean])
        self.a_std = torch.tensor([self.ac_std, self.dc_std])
        self.s_mean = torch.tensor([self.x_mean, self.y_mean, self.sc_mean, self.d_mean])
        self.s_std = torch.tensor([self.x_std, self.y_std, self.sc_std, self.d_std])
        torch.save({
            'ac_mean': self.ac_mean,
            'ac_std': self.ac_std,
            'dc_mean': self.dc_mean,
            'dc_std': self.dc_std,
            'x_mean': self.x_mean,
            'x_std': self.x_std,
            'y_mean': self.y_mean,
            'y_std': self.y_std,
            'd_mean': self.d_mean,
            'd_std': self.d_std,
            'sc_mean': self.sc_mean,
            'sc_std': self.sc_std,
            'a_mean': self.a_mean,
            'a_std': self.a_std,
            's_mean': self.s_mean,
            's_std': self.s_std,
        }, data_dir + '/stats.pth')

        self.x_min = self.df['x'].min()
        self.x_max = self.df['x'].max()
        self.y_min = self.df['y'].min()
        self.y_max = self.df['y'].max()
        self.sc_min = self.df['speed_calculated'].min()
        self.sc_max = self.df['speed_calculated'].max()
        self.ac_min = self.df['acceleration'].min()
        self.ac_max = self.df['acceleration'].max()
        self.dc_min = self.df['direction_change'].min()
        self.dc_max = self.df['direction_change'].max()

    def get_stats(self):
        return {
            'a_mean': self.a_mean,
            'a_std': self.a_std,
            's_mean': self.s_mean,
            's_std': self.s_std,
        }

    def load_images_in_memory(self, image_dir):
        """load images in memory

        Parameters
        ----------
        df : pd.DataFrame
            the dataframe containing the image paths
        image_dir : str
            the directory containing the images
        """
        print('[loading images in memory]')
        self.df['image'] = self.df.apply(lambda row: numpy.load(self.image_dir +row['filename']), axis=1)



    
    def get_batch_fm(self, split, npred=-1, cuda=True, onlyMoving=False, onlyTurning=False, onlyOtherShips=False):
        """get a batch of data for training a model

        Parameters
        ----------
        split : str {'train', 'valid', 'test'}
            the split to use
        npred : int
            the number of predictions to make
        cuda : bool
            whether to return cuda tensors
        onlyMoving : bool
            whether to only return ships which are moving (speed_calculated > 1)
        onlyTurning : bool
            whether to only return which currently are turning (direction_change > 5 degrees)
        """

        # Choose the correct device
        device = torch.device('cuda') if cuda else torch.device('cpu')

        if split == 'train':
            indx = self.train_indx
        elif split == 'valid':
            indx = self.valid_indx
        elif split == 'test':
            indx = self.test_indx

        if npred == -1:
            npred = self.opt.npred

        images, states, actions, target_states, target_images = [], [], [], [], []
        nb = 0
        T = self.opt.ncond + npred
        while nb < self.opt.batch_size:
            s = self.random.choice(indx)

            # skip iteration of s == 0
            if s == 0:
                continue
            
            # get row with train_index == s
            row = self.df.loc[self.df['train_index'] == s]

            # check if row exists and if it has index_column
            if row.empty or row['index_column'].empty:
                continue


            # get ship at position s in dataset
            index = row.index_column.values[0]
            # check if index + npred is in dataset
            if index + npred >= len(self.df):
                continue

            # check if ship is moving
            if onlyMoving and self.df.loc[index, 'speed_calculated'] <= 1:
                continue
                
            # check if ship is turning
            if onlyTurning and self.df.loc[index, 'direction_change'] <= 5:
                continue

            if onlyOtherShips and not self.otherShipsInImage(index):
                continue

            # test if goup_index is group_index + npred / we have target data
            if self.df.loc[index + npred].group_index == self.df.loc[index].group_index + npred:

                # data holders for batch
                batch_images = []
                batch_states = []
                batch_actions = []
                batch_target_states = []
                batch_target_images = []
                
                for index in self.df.loc[(self.df['index_column'] > index-self.opt.ncond) & (self.df['index_column'] <= index)].index_column.values:
                # states.append(df.loc[index]['state'])
                # get action at indexs
                    
                    # get state
                    x = self.df.loc[index]['x']
                    y = self.df.loc[index]['y']
                    speed = self.df.loc[index]['speed_calculated']
                    direction =self.df.loc[index]['direction']
                    state = torch.tensor([x, y, speed, direction], dtype=torch.float)
                    batch_states.append(state)
                    # load image
                    if self.opt.loadImagesInMemory:
                        image = self.df.loc[index]['image']
                    else:
                        image = numpy.load(self.image_dir +self.df.loc[index]['filename'])
                    # save as torch tensor
                    image = torch.from_numpy(image).type(torch.uint8)
                    image = image.permute(2, 0, 1)
                    batch_images.append(image)
                
                # add target data
                for index in self.df.loc[(self.df['index_column'] > index) & (self.df['index_column'] <= index+npred)].index_column.values:
                    # get actions to reach next state
                    acceleration = self.df.loc[index-1]['acceleration']
                    directionChange = self.df.loc[index-1]['direction_change']
                    batch_actions.append(torch.tensor([acceleration, directionChange], dtype=torch.float))
                    # get state
                    x = self.df.loc[index]['x']
                    y = self.df.loc[index]['y']
                    speed = self.df.loc[index]['speed_calculated']
                    direction = self.df.loc[index]['direction']
                    state = torch.tensor([x, y, speed, direction], dtype=torch.float)
                    batch_target_states.append(state)
                     # load image
                    if self.opt.loadImagesInMemory:
                        image = self.df.loc[index]['image']
                    else:
                        image = numpy.load(self.image_dir +self.df.loc[index]['filename'])
                    # save as torch tensor
                    image = torch.from_numpy(image).type(torch.uint8)
                    image = image.permute(2, 0, 1)
                    batch_target_images.append(image)

                # convert to torch tensors
                batch_images = torch.stack(batch_images)
                batch_states = torch.stack(batch_states)
                batch_actions = torch.stack(batch_actions)
                batch_target_states = torch.stack(batch_target_states)
                batch_target_images = torch.stack(batch_target_images)

                
                # append batch arrays to data holders
                images.append(batch_images)
                states.append(batch_states)
                actions.append(batch_actions)
                target_states.append(batch_target_states)
                target_images.append(batch_target_images)


                nb += 1

        # Pile up stuff
        images  = torch.stack(images).to(device)
        states  = torch.stack(states).to(device)
        actions = torch.stack(actions).to(device)
        target_states = torch.stack(target_states).to(device)
        target_images = torch.stack(target_images).to(device)
        # ego_cars = torch.stack(ego_cars)

        # Normalise actions, state_vectors, state_images
        actions = self.normalise_action(actions)
        states = self.normalise_state_vector(states)
        images = self.normalise_state_image(images)
        target_states = self.normalise_state_vector(target_states)
        target_images = self.normalise_state_image(target_images)


        t0 = self.opt.ncond
        t1 = T
        input_images  = images.float().contiguous()
        input_states  = states [:,   :t0].float().contiguous()
        target_images = target_images.float().contiguous()
        target_states = target_states.float().contiguous()
        # target_costs  = costs  [:, t0:t1].float().contiguous()
        t0 -= 1; t1 -= 1
        actions       = actions.float().contiguous()
       
        return [input_images, input_states], actions, [target_images, target_states]

    def get_syntetic_batch_fm(self, cuda=True):
        """
        Get a syntetic batch of MASS training data
        - sample a random starting state
        - sample a random action
        - ncond input states integrated with zero action (going in straight line with random speed and direction)
        - npred target states integrated with random action for random amount of steps
        - fill up till npred target states
        - use zero_image for all images
        """
        
         # Choose the correct device
        device = torch.device('cuda') if cuda else torch.device('cpu')
        images, states, actions, target_states, target_images = [], [], [], [], []
        nb = 0
        while nb < self.opt.batch_size:
            # sample random synthetic starting state
            start_x = np.random.uniform(self.x_min, self.x_max)
            start_y = np.random.uniform(self.y_min, self.y_max)
            start_vx = np.random.uniform(-10, 10)
            start_vy = np.random.uniform(-10, 10)
            start_state = torch.tensor([start_x, start_y, start_vx, start_vy], dtype=torch.float)
            zero_action = [0,0]
            # integrate 0 action for ncond steps
            
            batch_states = []
            batch_actions = []
            batch_states.append(start_state)
            for i in range(self.opt.ncond-1):
                next_state = ppuu_utils.integrate_state(batch_states[-1], zero_action)
                batch_states.append(torch.tensor(next_state))
            # sample random action
            action = self.sample_syntetic_action(batch_states[-1])
            # sample how long to continue action
            nsteps = np.random.randint(1, self.opt.npred)
            # integrate action for nsteps
            for i in range(nsteps):
                next_state = ppuu_utils.integrate_state(batch_states[-1], action)
                batch_states.append(torch.tensor(next_state))
                batch_actions.append(torch.tensor(action))
            for i in range(self.opt.npred - nsteps):
                next_state = ppuu_utils.integrate_state(batch_states[-1], zero_action)
                batch_states.append(torch.tensor(next_state))
                batch_actions.append(torch.tensor(zero_action))

            # convert to tensors
            batch_target_states = batch_states[self.opt.ncond:]
            batch_states = batch_states[:self.opt.ncond]
            batch_states = torch.stack(batch_states)
            batch_actions = torch.stack(batch_actions)
            batch_images = self.zero_neighbourhood.expand(self.opt.ncond, 3, self.opt.width, self.opt.height)
            batch_target_states = torch.stack(batch_target_states)
            batch_target_images = self.zero_neighbourhood.expand(self.opt.npred, 3, self.opt.width, self.opt.height)
            # append to data holders
            images.append(batch_images)
            states.append(batch_states)
            actions.append(batch_actions)
            target_states.append(batch_target_states)
            target_images.append(batch_target_images)

            nb +=1
    

        images = torch.stack(images).to(device)
        states = torch.stack(states).to(device)
        actions = torch.stack(actions).to(device)
        target_states = torch.stack(target_states).to(device)
        target_images = torch.stack(target_images).to(device)

        # normalize states, actions, target_states, images
        actions = self.normalise_action(actions)
        states = self.normalise_state_vector(states)
        images = self.normalise_state_image(images)
        target_states = self.normalise_state_vector(target_states)
        target_images = self.normalise_state_image(target_images)

        input_images  = images.float().contiguous()
        input_states  = states [:,   :self.opt.ncond].float().contiguous()
        target_images = target_images.float().contiguous()
        target_states = target_states.float().contiguous()
        actions       = actions.float().contiguous()

        return [input_images, input_states], actions, [target_images, target_states]


    def sample_syntetic_action(self, state):
        x, y, s, d = state
        if s > 0:
            a = np.random.uniform((1/15)*s -1, -(5/15) * s + 5)
        else:
            a = np.random.uniform(0, 4)
        dc = np.random.uniform(2*s -30, -2*s +30)
        return torch.Tensor([a,dc])

    def otherShipsInImage(self, index):
        '''
        Checks if there are any other ships in the image.
        '''
        image = numpy.load(self.image_dir +self.df.loc[index]['filename'])
        # check if sum of all pixel in green channel is higher than 10
        if np.sum(image[:,:,1]) > 40:
            return True
        else:
            return False



    @staticmethod
    def normalise_state_image(images):
        return images.float().div_(255.0)

    def normalise_state_vector(self, states):
        shape = (1, 1, 4) if states.dim() == 3 else (1, 4)  # dim = 3: state sequence, dim = 2: single state
        states -= self.s_mean.view(*shape).expand(states.size()).to(states.device)
        states /= (1e-8 + self.s_std.view(*shape).expand(states.size())).to(states.device)
        return states

    def normalise_action(self, actions):
        actions -= self.a_mean.view(1, 1, 2).expand(actions.size()).to(actions.device)
        actions /= (1e-8 + self.a_std.view(1, 1, 2).expand(actions.size())).to(actions.device)
        return actions


if __name__ == '__main__':
    # Create some dummy options
    class DataSettings:
        debug = False
        batch_size = 4
        npred = 20
        ncond = 20
    # Instantiate data set object
    d = DataLoader(DataSettings)
    # Retrieve first training batch
    x = d.get_batch_fm('train', cuda=False)
    print(x)