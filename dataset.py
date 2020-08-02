from torch.utils.data import Dataset
import numpy as np 
import torch
from PIL import Image
import os
import pickle
import matplotlib.pyplot as plt
from inception import *

from torchvision import transforms

tranfunc = transforms.Compose([transforms.Resize((224, 224))])
tranfunc2 = transforms.Compose([transforms.Resize((112, 112))])

# Max length: 208

# image vector length: 2048

def pad_tensor(vec, pad, dim):
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)

def pad_array(A, length):
    arr = np.zeros(length)
    arr[:len(A)] = A
    return arr

device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
class AccidentDataset(Dataset):
    def __init__(self, load=False):
        super(AccidentDataset, self).__init__()
        
        self.video_names = []

        if(load):
            label = pickle.load(open('./datasets/A3D_labels.pkl','rb'))
            cnnmodel = inception_v3(True).to(device)
            videos = os.listdir('./frames')
            for video in videos:
                images = os.path.join('./frames', video)
                images = os.path.join(images, 'images')
                images = os.listdir(images)
                
                print(video)
                if(len(images)==0):
                    continue

                self.video_names.append(video)

                S_array = []

                for image in images:
                    img = tranfunc(Image.open(os.path.join('./frames', video, 'images', image)))
                    img = np.transpose(img, (2, 0, 1))
                    img = np.expand_dims(img, 0)
                    img = torch.tensor(img).type(torch.FloatTensor).to(device)
                    img_vec = cnnmodel(img)
                    img_vec = img_vec[0].detach().cpu().numpy()
                    S_array.append(img_vec)

                print(np.array(S_array).shape)
                curr_seq = np.array(S_array)
                curr_label = pad_array(label[video]['target'], 210)
                print(curr_label)
                print(curr_label.shape)
            
                file_handler = open('./pickle/accident_train_{}.pkl'.format(video), 'wb+')
                pickle.dump((curr_seq, curr_label), file_handler)
                file_handler.close()

            file_handler = open('./pickle/video_names.pkl', 'wb+')
            pickle.dump(self.video_names, file_handler)
            file_handler.close()

        else:
            file_handler = open('./pickle/video_names.pkl', 'rb+')
            self.video_names = pickle.load(file_handler)
            file_handler.close()

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        file_handler = open('./pickle/accident_train_{}.pkl'.format(self.video_names[idx]), 'rb+')
        curr_seq, curr_label = pickle.load(file_handler)
        file_handler.close()
        inputs, targets = torch.tensor(curr_seq).type(torch.FloatTensor), torch.tensor(curr_label).type(torch.FloatTensor)
        inputs = pad_tensor(inputs, 210, 0)
        return inputs, targets

class AccidentDatasetC3D(Dataset):
    def __init__(self, load=False):
        super(AccidentDatasetC3D, self).__init__()
        
        self.video_names = []

        if(load):
            label = pickle.load(open('./datasets/A3D_labels.pkl','rb'))
            videos = os.listdir('./frames')
            for video in videos:
                images = os.path.join('./frames', video)
                images = os.path.join(images, 'images')
                images = os.listdir(images)
                
                print(video)
                if(len(images)==0):
                    continue

                self.video_names.append(video)

                S_array = []

                for image in images:
                    img = tranfunc2(Image.open(os.path.join('./frames', video, 'images', image)))
                    S_array.append(np.array(img))

                print(np.array(S_array).shape)
                curr_seq = np.array(S_array)
                # curr_seq = np.transpose(np.array(S_array), (3, 0, 1, 2))  # transpose((3, 0, 1, 2))
                curr_label = pad_array(label[video]['target'], 210)
                print(curr_label)
                print(curr_label.shape)
            
                file_handler = open('./pickle_img/accident_train_{}.pkl'.format(video), 'wb+')
                pickle.dump((curr_seq, curr_label), file_handler)
                file_handler.close()

            file_handler = open('./pickle_img/video_names.pkl', 'wb+')
            pickle.dump(self.video_names, file_handler)
            file_handler.close()

        else:
            file_handler = open('./pickle_img/video_names.pkl', 'rb+')
            self.video_names = pickle.load(file_handler)
            file_handler.close()

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        file_handler = open('./pickle_img/accident_train_{}.pkl'.format(self.video_names[idx]), 'rb+')
        curr_seq, curr_label = pickle.load(file_handler)
        file_handler.close()
        inputs, targets = torch.tensor(curr_seq).type(torch.FloatTensor), torch.tensor(curr_label).type(torch.FloatTensor)
        inputs = pad_tensor(inputs, 210, 0).permute(3, 0, 1, 2)
        return inputs, targets
        
if __name__ == '__main__':
    dataset = AccidentDatasetC3D(True)
