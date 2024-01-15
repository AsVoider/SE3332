from sklearn.model_selection import train_test_split
import os
import cv2 as cv
import torch
from torch.utils.data import DataLoader as dtl, random_split as rs


class VideoDataset(torch.utils.data.Dataset):
    '''
    Custom Dataset for loading videos and their class labels
    '''

    def __init__(self, data_dir, num_classes=10, num_frames=20, transform=None, target_transform=None):
        super().__init__()

        self.data_dir = data_dir

        self.transform = transform
        self.target_transform = target_transform
        self.num_classes = num_classes
        self.num_frames = num_frames

        self.video_filename_list = []
        self.classesIdx_list = []

        self.class_dict = {class_label: idx for idx, class_label in enumerate(
            sorted(os.listdir(self.data_dir)))}
        print(self.class_dict)

        for class_label, class_idx in self.class_dict.items():
            class_dir = os.path.join(self.data_dir, class_label)
            for video_filename in sorted(os.listdir(class_dir)):
                self.video_filename_list.append(
                    os.path.join(class_label, video_filename))
                self.classesIdx_list.append(class_idx)

    def __len__(self):
        return len(self.video_filename_list)

    def read_video(self, video_path):
        frames = []
        cap = cv.VideoCapture(video_path)
        count_frames = 0
        while True:
            ret, frame = cap.read()
            if ret:
                if self.transform:
                    transformed = self.transform(image=frame)
                    frame = transformed['image']

                frames.append(frame)
                count_frames += 1
            else:
                break

        stride = count_frames // self.num_frames
        new_frames = []
        count = 0
        for i in range(0, count_frames, stride):
            if count >= self.num_frames:
                break
            new_frames.append(frames[i])
            count += 1

        cap.release()

        return torch.stack(new_frames, dim=0)

    def __getitem__(self, idx):
        classIdx = self.classesIdx_list[idx]
        video_filename = self.video_filename_list[idx]
        video_path = os.path.join(self.data_dir, video_filename)
        frames = self.read_video(video_path)
        return frames, classIdx


def transform_data(num_class, batch_sz, num_frame, num_workers, data_dir, transform_fun):
    full_dataset = VideoDataset(data_dir=data_dir, num_frames=num_frame, num_classes=num_class, transform=transform_fun)
    train_val, test = train_test_split(full_dataset, test_size=0.2, random_state=42)
    train_, val_ = train_test_split(train_val, test_size=0.2, random_state=42)
    train_loader = dtl(train_, batch_size=batch_sz, shuffle=True, num_workers=num_workers)
    val_loader = dtl(val_, batch_size=batch_sz, shuffle=True, num_workers=num_workers)
    test_loader = dtl(test, batch_size=batch_sz, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def read_radio(data_path, transform, num_frames):
    frames = []
    cap = cv.VideoCapture(data_path)
    count_frames = 0
    while True:
        ret, frame = cap.read()
        if ret:
            if transform:
                transformed = transform(image=frame)
                frame = transformed['image']

            frames.append(frame)
            count_frames += 1
        else:
            break

    stride = count_frames // num_frames
    new_frames = []
    count = 0
    for i in range(0, count_frames, stride):
        if count >= num_frames:
            break
        new_frames.append(frames[i])
        count += 1

    cap.release()

    return torch.stack(new_frames, dim=0)


def split_dataloader(train_data, validation_split=0.2):
    train_ratio = 1 - validation_split
    train_size = int(train_ratio * len(train_data.dataset))
    val_size = len(train_data.dataset) - train_size

    train_dataset, val_dataset = rs(train_data, [train_size, val_size])
    batch_size = train_data.batch_size
    num_workers = train_data.num_workers

    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                             drop_last=True)
    val_data = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers,
                                           drop_last=True)

    return train_data, val_data

# if __name__ == '__main__':
#
#     logger.info('Loading dataset')
#     full_dataset = VideoDataset(data_dir="data", num_frames=num_frames, num_classes=num_classes, transform=transform)
#
#     train_dataset, test_dataset = train_test_split(full_dataset, test_size=0.2, random_state=42)
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
#                                                num_workers=num_workers)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
#                                               num_workers=num_workers)
#     logger.info('Dataset loaded')
#
#     for batch_idx, (data, target) in enumerate(train_loader):
#         print(data.shape)
#         print(target.shape)
#         break

# Hint: refer to https://www.kaggle.com/code/nguyenmanhcuongg/pytorch-video-classification-with-conv2d-lstm to implement your model and other functions
