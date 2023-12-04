from typing import Optional
from PIL import Image
import math


class ResizeImageByHeight:
    def __init__(self,
                 target_height: Optional[int] = None
                 ) -> None:
        
        if target_height is None:
             self.target_height = 128
        else:
             self.target_height = target_height

    def __call__(self, 
                 img: Image
                 ) -> Image:
        
        w, h = img.size
        
        aspect_ratio = w/h
        new_width = int(self.target_height * aspect_ratio)
        new_width = math.ceil(new_width / 2.) * 2
        return img.resize((new_width, self.target_height), Image.BICUBIC)


class ResizeImageByWidth:
        def __init__(self,
                     target_width: Optional[int] = None
                     ) -> None:
             
            if target_width is None:
                self.target_width = 128
            else:
                self.target_width = target_width

        
        def __call__(self, 
                     img: Image
                     ) -> Image:
             
             w, h = img.size 
             aspect_ratio = h/w
             new_height = int(self.target_width * aspect_ratio)
             new_height = math.ceil(new_height/ 2.) * 2
             return img.resize((new_height, self.target_width), Image.BICUBIC)
