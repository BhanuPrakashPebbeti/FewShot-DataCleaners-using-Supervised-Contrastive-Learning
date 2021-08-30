import os
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import os
from PIL import Image
import pickle
from torch.utils.data import DataLoader, Dataset
from random import shuffle
import albumentations as A
import torch
import pandas as pd
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import STL10
import torchvision.transforms.functional as tvf
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, SequentialSampler
import random
from PIL import Image
from torch.multiprocessing import cpu_count
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from argparse import Namespace

# img_dirs = []
# labels_temp = []
# files_cervix = os.listdir("/workstation/home/bijoy/data_from_b170007ec/Datasets/Cervix Cancer/train/train")
# files_cervix = sorted(files_cervix)
# for i in files_cervix[1:]:
#     path = "/workstation/home/bijoy/data_from_b170007ec/Datasets/Cervix Cancer/train/train/"+ str(i)
#     sub = os.listdir(path)
#     sub = sorted(sub)
#     for j in sub[1:]:
#         sub_path = path + "/" + str(j)
#         img_dirs.append(sub_path)
#         labels_temp.append(0)
        
# print("Length of Cervical images:",len(img_dirs))

# files_noise = os.listdir("/workstation/home/bijoy/data_from_b170007ec/Programs/Bhanu/SCLEARNING/Imagenet/train")
# files_noise = sorted(files_noise)
# for k in files_noise[:20000]:
#     path = "/workstation/home/bijoy/data_from_b170007ec/Programs/Bhanu/SCLEARNING/Imagenet/train/" + str(k)
#     img = Image.open(path)
#     t2 = transforms.ToTensor()(img)
#     if t2.shape[0] == 3:
#         img_dirs.append(path)
#         labels_temp.append(1)  
        
# blood_noise = os.listdir("/workstation/home/bijoy/data_from_b170007ec/Programs/Bhanu/SCLEARNING/Blood_images/train")
# blood_noise = sorted(blood_noise)
# for l in blood_noise[1:]:
#     path = "/workstation/home/bijoy/data_from_b170007ec/Programs/Bhanu/SCLEARNING/Blood_images/train/" + str(l)
#     img = Image.open(path)
#     t2 = transforms.ToTensor()(img)
#     if t2.shape[0] == 3:
#         img_dirs.append(path)
#         labels_temp.append(1)   
        
# eye_noise = os.listdir("/workstation/home/bijoy/data_from_b170007ec/Programs/Bhanu/SCLEARNING/eye_dataset")
# eye_noise = sorted(eye_noise)
# for m in eye_noise[1000:]:
#     path = "/workstation/home/bijoy/data_from_b170007ec/Programs/Bhanu/SCLEARNING/eye_dataset/" + str(m)
#     img = Image.open(path)
#     t2 = transforms.ToTensor()(img)
#     if t2.shape[0] == 3:
#         img_dirs.append(path)
#         labels_temp.append(1)  
        
# skin_noise = os.listdir("/workstation/home/bijoy/data_from_b170007ec/Programs/Bhanu/SCLEARNING/skin_cancer")
# skin_noise = sorted(skin_noise)
# for n in skin_noise[1000:]:
#     path = "/workstation/home/bijoy/data_from_b170007ec/Programs/Bhanu/SCLEARNING/skin_cancer/" + str(n)
#     img = Image.open(path)
#     t2 = transforms.ToTensor()(img)
#     if t2.shape[0] == 3:
#         img_dirs.append(path)
#         labels_temp.append(1)  

# surgery_noise = os.listdir("/workstation/home/bijoy/data_from_b170007ec/Programs/Bhanu/SCLEARNING/Surgery_frames")
# surgery_noise = sorted(surgery_noise)
# for o in surgery_noise[2000:]:
#     path = "/workstation/home/bijoy/data_from_b170007ec/Programs/Bhanu/SCLEARNING/Surgery_frames/" + str(o)
#     img = Image.open(path)
#     t2 = transforms.ToTensor()(img)
#     if t2.shape[0] == 3:
#         img_dirs.append(path)
#         labels_temp.append(1)        
  
train_df = pd.read_csv("/workstation/home/bijoy/data_from_b170007ec/Programs/Bhanu/SCLEARNING/TESTING_ON_5DATASETS/training_data.csv")
img_dirs = list(train_df["Path"])
labels_temp = list(train_df["Noise"])

data_temp = img_dirs
print("Length of total data :", len(data_temp))

temp = list(zip(data_temp, labels_temp))
random.shuffle(temp)
data, labels = zip(*temp)

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
       
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
     
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

    

def random_rotate(image):
    if random.random() > 0.5:
        return tvf.rotate(image, angle=random.choice((0, 90, 180, 270)))
    return image

class ResizedRotation():
    def __init__(self, angle, output_size=(512,512)):
        self.angle = angle
        self.output_size = output_size
        
    def angle_to_rad(self, ang): return np.pi * ang / 180.0
        
    def __call__(self, image):
        w, h = image.size
        new_h = int(np.abs(w * np.sin(self.angle_to_rad(90 - self.angle))) + np.abs(h * np.sin(self.angle_to_rad(self.angle))))
        new_w = int(np.abs(h * np.sin(self.angle_to_rad(90 - self.angle))) + np.abs(w * np.sin(self.angle_to_rad(self.angle))))
        img = tvf.resize(image, (new_w, new_h))
        img = tvf.rotate(img, self.angle)
        img = tvf.center_crop(img, self.output_size)
        return img

class WrapWithRandomParams():
    def __init__(self, constructor, ranges):
        self.constructor = constructor
        self.ranges = ranges
    
    def __call__(self, image):
        randoms = [float(np.random.uniform(low, high)) for _, (low, high) in zip(range(len(self.ranges)), self.ranges)]
        return self.constructor(*randoms)(image)


class PretrainingDatasetWrapper(Dataset):
    def __init__(self, ds: Dataset, l: labels, target_size=(512,512), debug=False):
        super().__init__()
        self.ds = ds
        self.labels = l
        self.debug = debug
        self.target_size = target_size
        if debug:
            print("DATASET IN DEBUG MODE")
        
        # I will be using network pre-trained on ImageNet first, which uses this normalization.
        # Remove this, if you're training from scratch or apply different transformations accordingly
        self.preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        random_resized_rotation = WrapWithRandomParams(lambda angle: ResizedRotation(angle, target_size), [(0.0, 360.0)])
        self.randomize = transforms.Compose([
            transforms.RandomResizedCrop(target_size, scale=(340/512, 340/512), ratio=(1.0, 1.0)),
            transforms.RandomChoice([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Lambda(random_rotate)
            ]),
            transforms.RandomApply([
                random_resized_rotation
            ], p=0.6),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.5, contrast=0.6, saturation=0.6, hue=0.3)
            ], p=0.8)
        ])
    
    def __len__(self): return len(self.ds)
    
    def __getitem_internal__(self, idx, preprocess=True):
        this_image_raw_path = self.ds[idx]
        label = self.labels[idx]
        this_image_raw = Image.open(this_image_raw_path)
        if self.debug:
            random.seed(idx)
            t1 = self.randomize(this_image_raw)
            random.seed(idx + 1)
            t2 = self.randomize(this_image_raw)
        else:
            t1 = self.randomize(this_image_raw)
            t2 = self.randomize(this_image_raw)
        
        if preprocess:
            t1 = self.preprocess(t1)
            t2 = self.preprocess(t2)
        else:
            t1 = transforms.ToTensor()(t1)
            t2 = transforms.ToTensor()(t2)
            
        return (t1, t2), torch.tensor(label)

    def __getitem__(self, idx):
        return self.__getitem_internal__(idx, True)
    
    def raw(self, idx):
        return self.__getitem_internal__(idx, False)



from efficientnet_pytorch import EfficientNet
class ImageEmbedding(nn.Module):       
    class Identity(nn.Module):
        def __init__(self): super().__init__()

        def forward(self, x):
            return x
    
        
    def __init__(self, embedding_size=1024):
        super().__init__()
        
        base_model = EfficientNet.from_pretrained("efficientnet-b2")
        internal_embedding_size = base_model._fc.in_features
        base_model._fc = ImageEmbedding.Identity()
        
        self.embedding = base_model
        
        self.projection = nn.Sequential(
            nn.Linear(in_features=internal_embedding_size, out_features=embedding_size),
            nn.ReLU(),
            nn.Linear(in_features=embedding_size, out_features=embedding_size)
        )

    def calculate_embedding(self, image):
        return self.embedding(image)

    def forward(self, X):
        image = X
        embedding = self.calculate_embedding(image)
        projection = self.projection(embedding)
        return embedding, projection


class ImageEmbeddingModule(pl.LightningModule):
    def __init__(self, hparams):
        hparams = Namespace(**hparams) if isinstance(hparams, dict) else hparams
        super().__init__()
        #self.hparams = hparams
        self.model = ImageEmbedding()
        self.loss = SupConLoss()
    
    def total_steps(self):
        return len(self.train_dataloader()) // hparams.epochs
    
    def train_dataloader(self):
        return DataLoader(PretrainingDatasetWrapper(data,labels,
                                             debug=getattr(hparams, "debug", False)),
                          batch_size=hparams.batch_size, 
                          num_workers=4,#cpu_count(),
                          sampler=SubsetRandomSampler(list(range(hparams.train_size))),
                         drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(PretrainingDatasetWrapper(data,labels,
                                            debug=getattr(hparams, "debug", False)),
                          batch_size=hparams.batch_size, 
                          shuffle=False,
                          num_workers=4,#cpu_count(),
                          sampler=SequentialSampler(list(range(hparams.train_size + 1, hparams.train_size + hparams.validation_size))),
                         drop_last=True)
    
    def forward(self, X):
        return self.model(X)
    
    def step(self, batch, step_name = "train"):
        (X, Y), labels = batch
        embX, projectionX = self.forward(X)
        embY, projectionY = self.forward(Y)
        z_i = F.normalize(projectionX , dim=1)
        z_j = F.normalize(projectionY, dim=1)
        projX = torch.reshape(z_i,(z_i.shape[0],1,z_i.shape[1]))
        projY = torch.reshape(z_j,(z_j.shape[0],1,z_j.shape[1]))
        features = torch.cat([projX, projY], dim=1)
        loss = self.loss(features=features,labels=labels)
        loss_key = f"{step_name}_loss"
        tensorboard_logs = {loss_key: loss}
        self.log("loss" if step_name == "train" else loss_key, loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #return { ("loss" if step_name == "train" else loss_key): loss, 'log': tensorboard_logs,
                        #"progress_bar": {loss_key: loss}}
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")
    
    def validation_end(self, outputs):
        if len(outputs) == 0:
            return {"val_loss": torch.tensor(0)}
        else:
            loss = torch.stack([x["val_loss"] for x in outputs]).mean()
            return {"val_loss": loss, "log": {"val_loss": loss}}

    def configure_optimizers(self):
        optimizer = RMSprop(self.model.parameters(), lr=hparams.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=15, verbose=True)
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler,
           'monitor': 'val_loss'
       }

hparams = Namespace(
    lr=0.0001,
    epochs=500,
    batch_size=22,
    train_size=33396,
    validation_size=3708
)

checkpoint_callback = ModelCheckpoint(
    dirpath="/workstation/home/bijoy/data_from_b170007ec/Programs/Bhanu/SCLEARNING/Checkpoints", 
    filename="model", 
    monitor='val_loss',
    verbose=True, 
    save_top_k=1,
    mode='min'
)
early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.0, patience=18, verbose=1, mode='min')
module = ImageEmbeddingModule(hparams)
trainer = pl.Trainer(gpus = [1],
                     accumulate_grad_batches=1518,
                     max_epochs=hparams.epochs,
                     replace_sampler_ddp = False,
                    callbacks=[checkpoint_callback,early_stop_callback],
                    progress_bar_refresh_rate=20)

trainer.fit(module)

