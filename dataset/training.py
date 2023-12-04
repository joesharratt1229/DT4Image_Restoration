import os
import torch
import torch.utils.data.dataset as dataset
from torchvision import transforms
from PIL import Image

from dataset.transforms import ResizeImageByHeight, ResizeImageByWidth


TRAINING_DIR = os.path.join(os.getcwd(), 'dataset/data/Images_128')

class TrainingDataset(dataset.Dataset):
    def __init__(self) -> None:
        super(TrainingDataset, self).__init__()
        
        self._training_dir = TRAINING_DIR
        self.height_resize = ResizeImageByHeight()
        self.width_resize = ResizeImageByWidth()
        self.transforms = transforms.Compose(
                          [transforms.ToTensor()]
                           )
        
    def __len__(self):
        return len(os.listdir(self._training_dir))
    
    
    def __getitem__(self, 
                    index: int
                    ) -> torch.Tensor:
        
        image_file = os.listdir(TRAINING_DIR)[index]
        image_path = os.path.join(TRAINING_DIR, image_file)
        image = Image.open(image_path)
        
        w, h = image.size

        if self.height_resize.target_height != h:
            image = self.height_resize(image)
        
        if self.width_resize.target_width != w:
            image = self.width_resize(image)

        image = self.transforms(image)

        return image