import mxnet
import torch

import numpy as np
import decord
import torch

from gluoncv.torch.utils.model_utils import download
from gluoncv.torch.data.transforms.videotransforms import video_transforms, volume_transforms
from gluoncv.torch.engine.config import get_cfg_defaults
from gluoncv.torch.model_zoo import get_model

# url = 'https://github.com/bryanyzhu/tiny-ucf101/raw/master/abseiling_k400.mp4'
# video_fname = download(url)
video_fname = 'videos/abseiling_k400.mp4'
vr = decord.VideoReader(video_fname)
frame_id_list = range(0, 64, 2)
video_data = vr.get_batch(frame_id_list).asnumpy()

crop_size = 224
short_side_size = 256
transform_fn = video_transforms.Compose([video_transforms.Resize(short_side_size, interpolation='bilinear'),
                                         video_transforms.CenterCrop(size=(crop_size, crop_size)),
                                         volume_transforms.ClipToTensor(),
                                         video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


clip_input = transform_fn(video_data)
print('Video data is downloaded and preprocessed.')


config_file = 'model/i3d_resnet50_v1_kinetics400.yaml'
cfg = get_cfg_defaults()
cfg.merge_from_file(config_file)
model = get_model(cfg)
model.eval()
print('%s model is successfully loaded.' % cfg.CONFIG.MODEL.NAME)

with torch.no_grad():
    pred = model(torch.unsqueeze(clip_input, dim=0)).numpy()
print('The input video clip is classified to be class %d' % (np.argmax(pred)))
