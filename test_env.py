import torch
import mmcv
from mmseg.models import build_segmentor

print("PyTorch Version:", torch.__version__)
print("Is CUDA available?", torch.cuda.is_available())
print("MMCV Version:", mmcv.__version__)

# Let's verify we can import and build the model structure
# (We don't need the weights yet, just verifying the PyTorch code compiles)
from mmcv.utils import Config
cfg = Config.fromfile('configs/ours/ganav_group6_rugd.py')

# Prevent the model from trying to load a massive dataset during a simple architecture test
cfg.model.pretrained = None 

model = build_segmentor(cfg.model)
print("Successfully initialized the GA-Nav Model Architecture!")