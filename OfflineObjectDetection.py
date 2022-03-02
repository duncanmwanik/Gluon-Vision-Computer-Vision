import mxnet as mx
from mxnet import image
from gluoncv.data.transforms.presets.segmentation import test_transform
import gluoncv

# use cpu
ctx = mx.cpu(0)

# load test image
img = image.imread('images/nick.jpg')
img = test_transform(img, ctx)
img = img.astype('float32')

# reconstruct the PSP network model
model = gluoncv.model_zoo.PSPNet(2)

# load the trained model
model.load_parameters('C:/Users/Mo/.mxnet/models/yolo3_darknet53_voc-f5ece5ce.params')

# make inference
output = model.predict(img)
predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()