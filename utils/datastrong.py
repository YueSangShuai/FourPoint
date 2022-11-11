import random
import numpy as np
import pkg_resources as pkg

from utils.general import colorstr
import logging


class Albumentations:
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self):
        self.transform = None
        try:
            import albumentations as A

            self.transform = A.Compose([
                A.MotionBlur(blur_limit=10, p=1)
                # A.Blur(p=0.01),
                # A.MedianBlur(p=0.01),
                # A.ToGray(p=0.00),
                # A.CLAHE(p=0.00),
                # A.RandomBrightnessContrast(p=0.0),
                # A.RandomGamma(p=0.0),
                # A.ImageCompression(quality_lower=75, p=0.0)
            ],
                bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
            logging.info(colorstr('albumentations: ') + ', '.join(f'{x}' for x in self.transform.transforms if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            logging.info(colorstr('albumentations: ') + f'{e}')

    def __call__(self, im, labels, p=1.0):
        if self.transform and random.random() < p:
            new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # transformed
            im, labels = new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])
        return im, labels
