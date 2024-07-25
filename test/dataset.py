import os
import time
import numpy as np
import torch
from torch.utils.data import Dataset, SubsetRandomSampler


class FeaturesDataset(Dataset):
    def __init__(self, data_folder, scene_info=None, subjects=None):
        super(FeaturesDataset, self).__init__()
        if scene_info is None:   # scene_info = [] or ['depth'] or ['mask'] or ['depth', 'mask']
            scene_info = ['depth', 'mask']
        if subjects is None:     # subjects = [] or [1] or [1,2] or [1,2,3] or [1,2,3,4] or [1,2,3,4,5] ......
            subjects = [1,2,3,4]

        self.action_folder = os.path.join(data_folder, 'motion')
        self.action_files = sorted([file for file in os.listdir(self.action_folder) if int(file.split('.')[0].split('_')[-1]) in subjects])
        self.action_data = None
        self.label_data = None
        self.scene_label_data = None
        self.frames_per_chunk = 64
        self.labels = ['climb', 'climb_wall', 'crouch', 'hand_wave', 'hit', 'hit_window', 'run', 'sit', 'stand', 'walk']
        self.action_data, self.label_data, self.scene_label_data = self.load_action(self.action_files)
        self.scene_info = scene_info


        if self.scene_info:
            self.depth_folder = os.path.join(data_folder, 'depth')
            self.depth_files = sorted([file for file in os.listdir(self.depth_folder) if int(file.split('.')[0].split('_')[-1]) in subjects])
            self.depth_shapes = {'climb_1.npy': (9216, 1, 112, 112), 'hand_wave_1.npy': (8896, 1, 112, 112),
                                 'crouch_3.npy': (8832, 1, 112, 112), 'climb_3.npy': (9216, 1, 112, 112),
                                 'sit_4.npy': (8832, 1, 112, 112), 'climb_2.npy': (5888, 1, 112, 112),
                                 'hit_1.npy': (9280, 1, 112, 112), 'hit_window_4.npy': (8704, 1, 112, 112),
                                 'stand_3.npy': (8640, 1, 112, 112), 'stand_4.npy': (8064, 1, 112, 112),
                                 'run_1.npy': (8448, 1, 112, 112), 'hit_3.npy': (9344, 1, 112, 112),
                                 'walk_2.npy': (2240, 1, 112, 112), 'crouch_1.npy': (8768, 1, 112, 112),
                                 'hand_wave_3.npy': (8960, 1, 112, 112), 'crouch_2.npy': (5120, 1, 112, 112),
                                 'hit_window_2.npy': (2752, 1, 112, 112), 'climb_wall_3.npy': (8960, 1, 112, 112),
                                 'climb_wall_1.npy': (8640, 1, 112, 112), 'run_2.npy': (3328, 1, 112, 112),
                                 'walk_4.npy': (8768, 1, 112, 112), 'hand_wave_2.npy': (1984, 1, 112, 112),
                                 'sit_1.npy': (2304, 1, 112, 112), 'hit_2.npy': (5312, 1, 112, 112),
                                 'hit_window_3.npy': (8640, 1, 112, 112), 'crouch_4.npy': (8064, 1, 112, 112),
                                 'run_3.npy': (8960, 1, 112, 112), 'sit_2.npy': (2304, 1, 112, 112),
                                 'hit_4.npy': (8896, 1, 112, 112), 'sit_3.npy': (8768, 1, 112, 112),
                                 'hit_window_1.npy': (6592, 1, 112, 112), 'hand_wave_4.npy': (4288, 1, 112, 112),
                                 'stand_2.npy': (3456, 1, 112, 112), 'climb_4.npy': (8832, 1, 112, 112),
                                 'climb_wall_4.npy': (8640, 1, 112, 112), 'walk_3.npy': (8960, 1, 112, 112),
                                 'walk_1.npy': (8192, 1, 112, 112), 'stand_1.npy': (8896, 1, 112, 112),
                                 'run_4.npy': (8640, 1, 112, 112), 'climb_wall_2.npy': (3392, 1, 112, 112),
                                 'hit_5.npy': (9088, 1, 112, 112), 'climb_5.npy': (9216, 1, 112, 112),
                                 'climb_wall_5.npy': (9600, 1, 112, 112), 'hit_window_5.npy': (9024, 1, 112, 112),
                                 'hit_6.npy': (9600, 1, 112, 112), 'climb_6.npy': (9088, 1, 112, 112), 
                                 'climb_wall_6.npy': (9024, 1, 112, 112), 'hit_window_6.npy': (9024, 1, 112, 112),
                                 'hand_wave_5.npy': (9088, 1, 112, 112), 'crouch_5.npy': (9216, 1, 112, 112),
                                 'run_5.npy': (9024, 1, 112, 112), 'stand_5.npy': (9024, 1, 112, 112),
                                 'walk_5.npy': (9152, 1, 112, 112), 'sit_5.npy': (9216, 1, 112, 112),
                                 'hand_wave_6.npy': (9728, 1, 112, 112), 'crouch_6.npy': (9536, 1, 112, 112),
                                 'stand_6.npy': (9152, 1, 112, 112), 'sit_6.npy': (9088, 1, 112, 112),
                                 'run_6.npy': (9216, 1, 112, 112), 'walk_6.npy': (9344, 1, 112, 112),
                                 'climb_7.npy': (2624, 1, 112, 112), 'climb_wall_7.npy': (3072, 1, 112, 112),
                                 'hand_wave_7.npy': (2688, 1, 112, 112), 'crouch_7.npy': (2880, 1, 112, 112),
                                 'hit_7.npy': (3328, 1, 112, 112), 'run_7.npy': (4704, 1, 112, 112),
                                 'stand_7.npy': (2432, 1, 112, 112), 'sit_7.npy': (3072, 1, 112, 112),
                                 'walk_7.npy': (3712, 1, 112, 112), 'hit_window_7.npy': (3584, 1, 112, 112)
                                 }
            assert len(self.action_files) == len(self.depth_files)

            if 'depth' in self.scene_info:
                self.depth_data = None
                self.depth_frame_indices = self.calculate_frame_indices(
                    [os.path.join(self.depth_folder, file_name) for file_name in self.depth_files])
                print("Use depth information!")

            if 'mask' in self.scene_info:
                self.mask_folder = os.path.join(data_folder, 'mask')
                self.mask_files = sorted([file for file in os.listdir(self.mask_folder) if int(file.split('.')[0].split('_')[-1]) in subjects])
                assert len(self.action_files) == len(self.mask_files)
                self.mask_data = None
                self.mask_frame_indices = self.calculate_frame_indices(
                    [os.path.join(self.depth_folder, file_name) for file_name in self.depth_files])
                print("Use mask information!")
        else:
            print("No use of scene information!")

    def __len__(self):
        return self.action_data.shape[0]

    def __getitem__(self, idx):
        # Get action and label data from the given index
        action = self.action_data[idx]
        label = self.label_data[idx]
        scene_label = self.scene_label_data[idx]
        action = torch.from_numpy(action)
        label = torch.tensor(label)
        scene_label = torch.tensor(scene_label)
        depth = torch.tensor([])
        mask = torch.tensor([])

        # Load depth and mask data
        if 'depth' in self.scene_info:
            depth = self.load_video(idx, video_type='depth')
            depth = torch.from_numpy(depth)
        if 'mask' in self.scene_info:
            mask = self.load_video(idx, video_type='mask')
            mask = torch.from_numpy(mask)

        return action, depth, mask, label, scene_label


    def load_action(self, file_name_list):
        # Load action data from the given file path
        actions = np.empty((0, 1, 512)).astype(np.float32)
        labels = []
        scene_labels = []
        for file_name in file_name_list:
            action = np.load(self.action_folder + '/' + file_name)
            actions = np.concatenate((actions, action), axis=0)
            # Extract label from the file name
            label, scene_label = self.extract_label(file_name)
            for i in range(action.shape[0]):
                labels.append(label)
                scene_labels.append(scene_label)
        labels = np.array(labels)
        scene_labels = np.array(scene_labels)
        print("All action features have been loaded!")
        print("Action features shape:", actions.shape)

        return actions, labels, scene_labels


    def load_video(self, idx=0, video_type='depth'):
        if video_type == 'depth':
            file_name, start_frame_idx = self.depth_frame_indices[idx]
            file_path = os.path.join(self.depth_folder, file_name)
        elif video_type == 'mask':
            file_name, start_frame_idx = self.mask_frame_indices[idx]
            file_path = os.path.join(self.mask_folder, file_name)
        else:
            raise ValueError("Type must be 'mask' or 'depth'")
        # Load a video from the given file path
        video_data = np.load(file_path, mmap_mode='r+')

        # Initialize an empty list to store processed frames
        frames_per_chunk = self.frames_per_chunk

        # Read and process each frame
        frames = video_data[start_frame_idx:start_frame_idx + frames_per_chunk]

        # If the number of processed frames reaches the specified chunk size, create a chunk
        if frames.shape[0] == frames_per_chunk:
            chunk = np.transpose(frames, (1, 0, 2, 3))
        else:
            chunk = None

        return chunk

    def extract_label(self, file_path):
        # Extract label from the file path
        # Assuming the label is the part before the last underscore (_) in the action file name
        raw_label = os.path.basename(file_path).rsplit('_', 1)[0]
        label_index = self.labels.index(raw_label)
        scene_label = (1 if label_index in [0,1,4,5] else 0)

        return label_index, scene_label

    def calculate_frame_indices(self, file_name_list):
        # Calculate the starting frame index for each chunk
        frame_indices = {}
        current_frame_index = 0
        frames_per_chunk = self.frames_per_chunk

        for file_name in file_name_list:
            frame_count = int(self.depth_shapes[file_name.split('/')[-1]][0])

            for i in range(0, frame_count, frames_per_chunk):
                frame_indices[current_frame_index // frames_per_chunk] = (file_name.split('/')[-1], i)
                current_frame_index += frames_per_chunk

        return frame_indices


# Example usage
if __name__ == "__main__":
    time0 = time.time()
    data_folder = os.path.join(os.path.dirname(__file__), '..', 'Data')
    dataset = FeaturesDataset(data_folder, scene_info=['depth'], subjects=[5,6,7])
    batch_size = 16
    test_split_ratio = 0.2
    val_split_ratio = 0.16
    shuffle_dataset = True
    random_seed = 42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    test_split = int(np.floor(test_split_ratio * dataset_size))
    val_split = int(np.floor(val_split_ratio * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices, test_indices = indices[test_split + val_split:], indices[:val_split], indices[
                                                                                                      val_split:val_split + test_split]
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)


    def custom_collate(batch):
        action = torch.stack([item[0] for item in batch])
        depth = torch.stack([item[1] for item in batch])
        mask = torch.stack([item[2] for item in batch])
        label = torch.stack([item[3] for item in batch])

        return action, depth, mask, label

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    print("Time taken to load dataset set: ", time.time() - time0)

    def test():
        timexx = time.time()
        for batch_idx, (action_tensor, depth_tensor, mask_tensor, target) in enumerate(train_loader):
            # print("Batch index: ", idx)
            print("Batch index: ", batch_idx)
            print("Action shape: ", action_tensor.shape)
            print("Depth shape: ", depth_tensor.shape)
            print("Mask shape: ", mask_tensor.shape)
            print("Target shape: ", target.shape)
            print("Time taken to load 3 batch: ", time.time() - timexx)
            if batch_idx == 2:
                break

    test()

