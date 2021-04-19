from pycocotools.coco import COCO 
from PIL import Image 
import numpy as np 
from torch.utils import data 
import torch 
from pathlib import Path 
import os 

import random 
from torchvision import transforms

import albumentations as A
class COCODataset(data.Dataset):
    def __init__(self, 
        img_folder=None,
        annot_file=None,
        transform=None,
        max_annot = -1
    ):
        super(COCODataset, self).__init__() 

        assert img_folder is not None, "Missing image folder path!"
        assert annot_file is not None, "Missing annotations json file, should be a coco format json file!"

        self.img_folder = Path(img_folder)
        self.coco = COCO(annot_file)

        print("Num classes", len(self.coco.getCatIds()))
        self.imgIds = sorted(self.coco.getImgIds()) 

        self.max_annot = max_annot

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.transform = self._augmentation

        if not os.path.isfile("palette.png"):
            print("Downloading palette...")
            os.system("gdown --id 1DT0b0WeGxiLQVcUai4hR3KIWwyh56rKu -O palette.png")
        self.palette = Image.open("palette.png").getpalette()
        #print(self.palette, Image.open("00000.png").size)

    def _augmentation(self, img, mask):
        train_transform = [
            A.PadIfNeeded(min_height=480, min_width=480, always_apply=True,border_mode = 0),
            A.Resize(480, 480),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=20, p = 1, border_mode = 0),
            A.RandomCrop(height=384, width=384, always_apply=True),
        ]
        transform = A.Compose(train_transform)
        result = transform(image = np.array(img), mask = np.array(mask))
        img, mask = result['image'], result['mask']
        return img, mask 

    def __len__(self): 
        return len(self.imgIds)
        
    def __getitem__(self, index):
        imgId = self.coco.loadImgs(self.imgIds[index])[0]

        img_path = self.img_folder / imgId['file_name']
        img = self.preprocess(Image.open(img_path).convert('RGB'))


        #print("Image shape", img.shape)
        # Get annotations of this image
        annIds = self.coco.getAnnIds(imgId['id']) 
        
        # Choose randomly x annot in annIds
        annIds = random.sample(annIds, k = min(self.max_annot, len(annIds)))

        anns = self.coco.loadAnns(annIds) 
        mask = self._get_mask(imgId['height'], imgId['width'], anns)
        
        # mask = Image.fromarray(mask).convert('P')
        # if self.palette:
        #     mask.putpalette(self.palette)
        
        if self.transform is not None:
          img, mask = self.transform(img.permute(1, 2, 0), mask)
          img = transforms.ToTensor()(img)#.permute(2, 0, 1)
          mask = torch.LongTensor(np.array(mask))
          #print(f"Mask {index}", torch.min(mask), torch.max(mask))
        #print(img.shape, mask.shape)
        
        return img, mask
    
    def _get_mask(self, h, w, anns):
        combined_mask = np.zeros((h, w), dtype=np.int8)
        #ids = np.random.choice(np.arange(10), size=len(anns))
        for i, ann in enumerate(anns):
            mask = self.coco.annToMask(ann)
            # if ann['category_id'] > 80:
              # import pprint
              # print(self.coco.getCatIds(catIds=ann['category_id']))
              # pprint.pprint(ann)
            combined_mask[mask == 1] = ann['category_id']
        return combined_mask