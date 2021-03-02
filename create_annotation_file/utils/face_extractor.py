from __future__ import print_function

import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from models.prior_box import PriorBox
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms

cudnn.benchmark = True


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # print('Missing keys:{}'.format(len(missing_keys)))
    # print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    # print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    # print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, device_id):
    # print('Loading pretrained model from {}'.format(pretrained_path))
    pretrained_dict = torch.load(pretrained_path, map_location=device_id)
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


class RetinaFaceDetector:

    def __init__(self, device, pretrained_model):
        self.device = device
        self.cfg = {
            'name': 'Resnet50',
            'min_sizes': [[16, 32], [64, 128], [256, 512]],
            'steps': [8, 16, 32],
            'variance': [0.1, 0.2],
            'clip': False,
            'loc_weight': 2.0,
            'gpu_train': True,
            'batch_size': 24,
            'ngpu': 4,
            'epoch': 100,
            'decay1': 70,
            'decay2': 90,
            'image_size': 840,
            'pretrain': True,
            'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
            'in_channel': 256,
            'out_channel': 256
        }

        self.net = RetinaFace(cfg=self.cfg, phase='test')
        self.net = load_model(self.net, pretrained_model, device)
        self.net.eval()
        self.net = self.net.to(device)

    def predict(self, image, confidence_threshold=0.02, top_k=5000, nms_threshold=0.4, keep_top_k=750):
        torch.set_grad_enabled(False)
        img = np.float32(image)
        resize = 1

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        loc, conf, landms = self.net(img)  # forward pass
        # print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        bboxes, landmarks, confident_scores = dets[:, :4], dets[:, 5:], dets[:, 4]

        boxes = []
        if bboxes is None:
            boxes = None
        else:
            for box in bboxes:
                x0, y0, x1, y1 = tuple(box.astype(int))
                height, width = y1 - y0, x1 - x0
                distance = max(height, width)
                if height < distance:
                    gap = distance - height
                    y0 -= gap / 2
                    y1 += gap / 2
                elif width < distance:
                    gap = distance - width
                    x0 -= gap / 2
                    x1 += gap / 2
                if y0 < 0:
                    y1 -= y0
                    y0 = 0
                if x0 < 0:
                    x1 -= x0
                    x0 = 0
                boxes.append([x0, y0, x1, y1])
            boxes = np.array(boxes).astype(int)

        return boxes, landmarks.reshape(-1, 5, 2), confident_scores

detector = None
dir_path = os.path.dirname(os.path.realpath(__file__))

def extract_face(image):
    global detector
    if detector is None:
        weight_file = os.path.join(dir_path, 'weights', 'Resnet50_Final.pth')
        detector = RetinaFaceDetector(torch.device('cpu'), pretrained_model=weight_file)
    bboxes, landmarks, confident = detector.predict(image)
    max_index = np.argmax(confident)
    return bboxes[max_index]



def get_landmark_most_points(landmarks):
    min_x, min_y, max_x, max_y = 9999, 9999, 0, 0
    for landmark in landmarks:
        if min_x > landmark[0]:
            min_x = landmark[0]
        if max_x < landmark[0]:
            max_x = landmark[0]
        if min_y > landmark[1]:
            min_y = landmark[1]
        if max_y < landmark[1]:
            max_y = landmark[1]
    return min_x, min_y, max_x, max_y


class CentralCrop(object):
    """Crop the image in a sample.
        Make sure the head is in the central of image
    Args:
        output_size (tuple or int): Desired output size. If int, square crop is made.
    """

    def __init__(self, output_size, gap_percent=0.05, showing_top=0.65):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.percent = gap_percent
        self.showing_top = showing_top

    def __call__(self, image, landmarks):

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        min_x, min_y, max_x, max_y = get_landmark_most_points(landmarks)

        # if max_x > w:
        #     different = max_x - w
        #     min_x -= different
        # if max_y > h:
        #     different = max_y - h
        #     min_y -= different

        max_face = max(max_y - min_y, max_x - min_x)
        gap = min((max_y - min_y), (max_x - min_x)) * self.percent
        distance = int(max_face + gap * 2)
        if distance > min(w, h):
            distance = min(w, h)

        x = int(min_x - (distance - (max_x - min_x)) / 2)
        if x < 0:
            x = 0
        if x + distance < max_x:
            x = int(max_x - distance)
        if x + distance > w:
            x = w - distance
        y = int((min_y - gap) * self.showing_top)
        if y < 0:
            y = 0
        if y + distance < max_y:
            y = int(max_y - distance)
        if y + distance > h:
            y = h - distance

        image = image[y: y + distance, x: x + distance].copy()

        assert image.shape[0] == image.shape[1]

        landmarks = landmarks - np.array([x, y])

        if new_w > image.shape[1]:
            image = cv2.resize(image, self.output_size, interpolation=cv2.INTER_CUBIC)
        else:
            image = cv2.resize(image, self.output_size, interpolation=cv2.INTER_LANCZOS4)

        landmarks *= np.array([new_w, new_h]) / float(distance)

        # image_debug_utils.show_landmarks(sample['image'], landmarks)
        return image, landmarks
