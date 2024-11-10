import cv2
import os
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import xml.etree.ElementTree as ET
import time
import torch

# convert to RGB
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 
    

# normalization
def preprocess(image):
    image /= 255.0
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    return image

def rand(a=0, b=1):
        return np.random.rand()*(b-a) + a

def augmentation(images, boxes,h, w, hue=.1, sat=0.7, val=0.4):
    # images [5, w, h, 3], bbox [:,4]
    #------------------------------------------#
    #   翻转图像
    #------------------------------------------#
    filp = rand()<.5
    if filp:
        for i in range(len(images)):
            images[i] = Image.fromarray(images[i].astype('uint8')).convert('RGB').transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        for i in range(len(boxes)):
            boxes[i][[0,2]] = w - boxes[i][[2,0]]

    images      = np.array(images, np.uint8)
    #---------------------------------#
    #   对图像进行色域变换
    #   计算色域变换的参数
    #---------------------------------#
    r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
    #---------------------------------#
    #   将图像转到HSV上
    #---------------------------------#
    for i in range(len(images)):
        hue, sat, val   = cv2.split(cv2.cvtColor(images[i], cv2.COLOR_RGB2HSV))
        dtype           = images[i].dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        images[i] = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        images[i] = cv2.cvtColor(images[i], cv2.COLOR_HSV2RGB)

    return np.array(images,dtype=np.float32), np.array(boxes,dtype=np.float32)



import glob
class seqDataset(Dataset):
    def __init__(self, dataset_path, image_size, num_frame=5, type='train'):
        super(seqDataset, self).__init__()
        self.image_size = image_size
        self.num_frame = num_frame
        if type == 'train':
            self.aug = True
        else:
            self.aug = False

        self.img_idx = []
        self.anno_idx = []

        with open(dataset_path) as f: 
            data_lines = f.readlines()
            self.length = len(data_lines)
            for line in data_lines:
                line = line.strip('\n').split()
                self.img_idx.append(line[0])
                self.anno_idx.append(np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]]))
        
        self.radius = 2
        self.suffix = '.bmp'
    
    def get_data(self, index):
        image_data = []
        h, w = self.image_size, self.image_size
        # Thanks for your attention! After the paper accept, we will open the details soon.

        image_data = np.array(image_data) 
        label_data = np.array(label_data, dtype=np.float32) 
        if self.aug is True:
            pass
            # image_data, label_data[:, :4] = augmentation(image_data, label_data[:, :4], h, w)
        return image_data, label_data

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        images, box = self.get_data(index)
        images = np.transpose(preprocess(images), (3, 0, 1, 2))  # t, h, w, c --> c, t, h, w 
        if len(box) != 0:
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + (box[:, 2:4] / 2)
        return images, box

                    
def dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    return images, bboxes

                
            
    
if __name__ == "__main__":
    train_dataset = seqDataset("/home/coco_train.txt", 256, 5, 'train')
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2, collate_fn=dataset_collate)
    t = time.time()
    for index, batch in enumerate(train_dataloader):
        images, targets = batch[0], batch[1]
        print(index)
    print(time.time()-t)