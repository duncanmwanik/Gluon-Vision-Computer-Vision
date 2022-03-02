from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt
import cv2

# net = model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True)
net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)

while True:
    im_fname = input('>')
    image = cv2.imread(im_fname)

    x, img = data.transforms.presets.yolo.load_test(im_fname, short=512)
    print('Shape of pre-processed image:', x.shape)

    class_IDs, scores, bounding_boxes = net(x)
    ax = utils.viz.plot_bbox(img, bounding_boxes[0], scores[0],
                             class_IDs[0], class_names=net.classes)
    plt.show()
