from .blender import BlenderDataset
from .llff import LLFFDataset
from .mednerf import MednerfDataset
dataset_dict = {'blender': BlenderDataset,
                'mednerf':MednerfDataset,
                'llff': LLFFDataset}