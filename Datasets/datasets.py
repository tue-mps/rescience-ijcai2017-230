import torch
import json
import torchvision.transforms.functional as FT
import os
import os.path
from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset
from PIL import Image
from typing import Any, Callable, Optional, Tuple


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VocDataset(Dataset):
    """
    Method to load in the VOC dataset such that it is readable by PyTorch data_loader function.
    Uses a collate function to make it able to have batches of different image sizes and different amount of
    objects per image.
    Source: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/datasets.py (adapted)
    """

    def __init__(self, data_folder, split):

        self.split = split.upper()

        assert self.split in {'TRAIN', 'VALIDATION', 'TEST'}  # If not one of these, raise error

        self.data_folder = data_folder

        # Read data files.
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        # Read image
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        # Read objects in this image (bounding boxes, labels, difficulties).
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        difficulties = torch.ByteTensor(objects['difficulties'])  # (n_objects)

        image = FT.to_tensor(image)

        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):

        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        targets = []
        images2 = list(image for image in images)

        for x in range(len(images2)):
            dic = {}
            dic['boxes'] = boxes[x]
            dic['labels'] = labels[x]
            dic['difficulties'] = difficulties[x]
            targets.append(dic)

        return images2, targets


class CocoDetection(VisionDataset):
    """
        Method to load in the COCO dataset such that it is readable by PyTorch data_loader function.
        Uses a collate function to make it able to have batches of different image sizes and different amount of
        objects per image. Adapted to be in the same format as the VOC dataset.
        Source: https://github.com/pytorch/vision/blob/master/torchvision/datasets/coco.py (adapted)
        """

    def __init__(
            self,
            root: str,
            annFile: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
            ) -> None:
            super(CocoDetection, self).__init__(root, transforms, transform, target_transform)
            from pycocotools.coco import COCO
            self.coco = COCO(annFile)
            self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, Any]:

        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        corrupted = False

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        boxes = list()
        labels = list()
        areas = list()

        for l in target:
            bbox = l['bbox']
            bbox_corrected = l['bbox']
            bbox_corrected[2] = bbox[0] + bbox[2]        # Given in x, y, w, h
            bbox_corrected[3] = bbox[1] + bbox[3]        # Converted to x1, y1, x2, y2
            boxes.append(bbox_corrected)
            labels.append(l['category_id'])
            areas.append(l['area'])

            if bbox_corrected[0] == bbox_corrected[2]:
                corrupted = True
                break
            elif bbox_corrected[1] == bbox_corrected[3]:
                corrupted = True
                break
            else:
                corrupted = False

        # This is to filter out any images that contain no objects at all, or have a faulty annotation
        # in the bounding boxes.
        if len(target) == 0:
            return None, None, None, None
        elif corrupted == True:
            return None, None, None, None
        else:
            return img, torch.FloatTensor(boxes), torch.LongTensor(labels), torch.FloatTensor(areas)

    def __len__(self) -> int:
        return len(self.ids)


    def collate_fn(batch):

        # if a batch contains a None image (coming from getitem), this image and target is filtered out
        batch = [(a, b, c, d) for (a, b, c, d) in batch if a is not None]

        images = list()
        boxes = list()
        labels = list()
        areas = list()

        for b in batch:
            images.append(FT.to_tensor(b[0]))
            boxes.append(b[1])
            labels.append(b[2])
            areas.append(b[3])

        targets = []
        images2 = list(image for image in images)

        for x in range(len(images2)):
            dic = {}
            dic['boxes'] = boxes[x]
            dic['labels'] = labels[x]
            dic['areas'] = areas[x]
            targets.append(dic)

        return images2, targets
