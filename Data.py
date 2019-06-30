from PIL import Image
from tqdm import tqdm
from pathlib import Path
import torch

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as tf



class CCPD5000:
    def __init__(self, img_dir):  # use CCPD5000(path) initiate
        self.img_dir = Path(img_dir)

        self.img_paths_all = self.img_dir.glob("*.jpg")
        self.img_paths_list = list(self.img_paths_all)
        self.img_paths = sorted(self.img_paths_list)  # sorted image path

    #    def __init__(self, img_dir):
    #        self.img_dir = Path(img_dir)
    #        self.img_paths = self.img_dir.glob('*.jpg')
    #        self.img_paths = sorted(list(self.img_paths))

    def __len__(self):  # return size
        return len(self.img_paths)

#    def __len__(self):
#        return len(self.img_paths)

    def __getitem__(self, index):  # call with CCPD5000((int)index) return [img , [8, ](tensor)ground_truth]
        target_img_path = self.img_paths[index]
        target_img = Image.open(target_img_path)
        (W, H) = target_img.size

        # get [8, ] ground truth
        img_name = target_img_path.name
        spilt_component = img_name.split("-")
        information = spilt_component[3]
        information_split1 = "&".join(information.split("_"))
        information_split2 = information_split1.split("&")

        ground_truth = list()
        for i in range(len(information_split2)):
            ground_truth.append(float(information_split2[i]))

        for idx in range(len(ground_truth)):
            if(idx%2==0):
                ground_truth[idx] = ground_truth[idx]/W
            else:
                ground_truth[idx] = ground_truth[idx]/H

        ground_truth = torch.tensor(ground_truth)

        # adjust image UNCHANGED
        target_img = target_img.convert("RGB")
        target_img = target_img.resize((192, 320))
        target_img = tf.to_tensor(target_img)

        return (target_img, ground_truth)



#    def __getitem__(self, idx):
#        img_path = self.img_paths[idx]

# load image
#        img = Image.open(img_path)
#        W, H = img.size
#        img = img.convert('RGB')
#        img = img.resize((192, 320))
#        img = tf.to_tensor(img)

# parse annotation
#        name = img_path.name
#        token = name.split('-')[3]
#        token = token.replace('&', '_')
#        kpt = [float(val) for val in token.split('_')]
#        kpt = torch.tensor(kpt)  # [8,]
#        kpt = kpt.view(4, 2)  # [4, 2]
#        kpt = kpt / torch.FloatTensor([W, H])
#        kpt = kpt.view(-1)  # [8,]

#        return img, kpt


train_set = CCPD5000('./ccpd5000/train')
print(len(train_set))

img, kpt = train_set[90]
print(img.size())
print(kpt.size())

print(kpt)

# image display for colab (disable for pycharm)

#img = tf.to_pil_image(img)
#vis = draw_kpts(img, kpt, c='orange')
#vis = draw_plate(vis, kpt)
#vis.save('./check.jpg')

#from IPython import display
#display.Image('./check.jpg')
