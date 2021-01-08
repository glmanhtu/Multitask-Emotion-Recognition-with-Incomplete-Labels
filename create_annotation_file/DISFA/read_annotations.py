import math
import os
import pickle
import re

import cv2
import numpy as np
import argparse

import scipy.io
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib
import glob

np.random.seed(0)
parser = argparse.ArgumentParser(description='save annotations')
parser.add_argument('--vis', action='store_true',
                    help='whether to visualize the distribution')
parser.add_argument('--annot_dir', type=str, default='/home/mvu/Documents/datasets/disfa/ActionUnit_Labels',
                    help='annotation dir')
parser.add_argument("--image_dir", type=str, default='/home/mvu/Documents/datasets/disfa/Images/Left')
parser.add_argument("--cropped_dir", type=str, default='/home/mvu/Documents/datasets/mixed/disfa/cropped_aligned')
parser.add_argument('--save_path', type=str, default='/home/mvu/Documents/datasets/mixed/disfa/annotations.pkl')

args = parser.parse_args()

if not os.path.isdir(args.cropped_dir):
    os.makedirs(args.cropped_dir)


def crop_face(image, keypoints, rotate=True, quiet_mode=True):
    lex, ley = (keypoints[36] + keypoints[39]) / 2
    rex, rey = (keypoints[42] + keypoints[45]) / 2
    rmx, rmy = keypoints[54]
    lmx, lmy = keypoints[48]
    nex, ney = keypoints[33]
    # roation using PIL image

    if rotate:
        angle = calculate_angle(lex, ley, rex, rey)
        image, lex, ley, rex, rey, lmx, lmy, rmx, rmy \
            = image_rote(image, angle, lex, ley, rex, rey, lmx, lmy, rmx, rmy)
    eye_width = rex - lex  # distance between two eyes
    ecx, ecy = (lex + rex) / 2.0, (ley + rey) / 2.0  # the center between two eyes
    mouth_width = rmx - lmx
    mcx, mcy = (lmx + rmx) / 2.0, (lmy + rmy) / 2.0  # mouth center coordinate
    em_height = mcy - ecy  # height between mouth center to eyes center
    fcx, fcy = (ecx + mcx) / 2.0, (ecy + mcy) / 2.0  # face center
    # face
    if eye_width > em_height:
        alpha = eye_width
    else:
        alpha = em_height
    g_beta = 2.0
    g_left = fcx - alpha / 2.0 * g_beta
    g_upper = fcy - alpha / 2.0 * g_beta
    g_right = fcx + alpha / 2.0 * g_beta
    g_lower = fcy + alpha / 2.0 * g_beta
    g_face = image.crop((g_left, g_upper, g_right, g_lower))

    return g_face


def image_rote(img, angle, elx, ely, erx, ery, mlx, mly, mrx, mry, expand=1):
    w, h = img.size
    img = img.rotate(angle, expand=expand)  # whether to expand after rotation
    if expand == 0:
        elx, ely = pos_transform_samesize(angle, elx, ely, w, h)
        erx, ery = pos_transform_samesize(angle, erx, ery, w, h)
        mlx, mly = pos_transform_samesize(angle, mlx, mly, w, h)
        mrx, mry = pos_transform_samesize(angle, mrx, mry, w, h)
    if expand == 1:
        elx, ely = pos_transform_resize(angle, elx, ely, w, h)
        erx, ery = pos_transform_resize(angle, erx, ery, w, h)
        mlx, mly = pos_transform_resize(angle, mlx, mly, w, h)
        mrx, mry = pos_transform_resize(angle, mrx, mry, w, h)
    return img, elx, ely, erx, ery, mlx, mly, mrx, mry


def calculate_angle(elx, ely, erx, ery):
    """
    calculate image rotate angle
    :param elx: lefy eye x
    :param ely: left eye y
    :param erx: right eye x
    :param ery: right eye y
    :return: rotate angle
    """
    dx = erx - elx
    dy = ery - ely
    angle = math.atan(dy / dx) * 180 / math.pi
    return angle


def pos_transform_resize(angle, x, y, w, h):
    """
    after rotation, new coordinate with expansion
    :param angle:
    :param x:
    :param y:
    :param w:
    :param h:
    :return:
    """
    angle = angle * math.pi / 180
    matrix = [math.cos(angle), math.sin(angle), 0.0, -math.sin(angle), math.cos(angle), 0.0]

    def transform(x, y, matrix=matrix):
        (a, b, c, d, e, f) = matrix
        return a * x + b * y + c, d * x + e * y + f  # calculate output size

    xx = []
    yy = []
    for x_, y_ in ((0, 0), (w, 0), (w, h), (0, h)):
        x_, y_ = transform(x_, y_)
        xx.append(x_)
        yy.append(y_)
    ww = int(math.ceil(max(xx)) - math.floor(min(xx)))
    hh = int(math.ceil(max(yy)) - math.floor(min(yy)))
    # adjust center
    cx, cy = transform(w / 2.0, h / 2.0)
    matrix[2] = ww / 2.0 - cx
    matrix[5] = hh / 2.0 - cy
    tx, ty = transform(x, y)
    return tx, ty


def pos_transform_samesize(angle, x, y, w, h):
    """
    after rotation, new coordinate without expansion
    :param angle:
    :param x:
    :param y:
    :param w:
    :param h:
    :return:
    """
    angle = angle * math.pi / 180
    matrix = [math.cos(angle), math.sin(angle), 0.0, -math.sin(angle), math.cos(angle), 0.0]

    def transform(x, y, matrix=matrix):
        (a, b, c, d, e, f) = matrix
        return a * x + b * y + c, d * x + e * y + f

    cx, cy = transform(w / 2.0, h / 2.0)
    matrix[2] = w / 2.0 - cx
    matrix[5] = h / 2.0 - cy
    x, y = transform(x, y)
    return x, y


def PIL_image_convert(cv2_im):
    cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)
    return pil_im


def read_au(txt_prefix):
    au_list = [1, 2, 4, 6, 12, 15, 20, 25]
    aus = []
    for i in au_list:
        txt_file = txt_prefix + "{}.txt".format(i)
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        lines = [x.strip() for x in lines]
        lines = [int(x.split(',')[1]) for x in lines]
        lines = [0 if x < 2 else 1 for x in lines]  # if intensity is equal to or greater than 2, it is positive sample
        aus.append(lines)
    aus = np.stack(aus, axis=1)
    return aus


def disfa_crop_face(img_path):
    res = re.findall(r'LeftVideo(.+)_comp.+frame-(\d+)\.jpg', img_path)
    vid_id, frame_id = res[0]
    aam_landmark_path = os.path.join(os.path.dirname(args.annot_dir), 'Landmark_Points')
    landmarks_file = os.path.join(aam_landmark_path, vid_id, 'tmp_frame_lm',
                                  "{}_{:04d}_lm.mat".format(vid_id, int(frame_id) - 1))
    if not os.path.isfile(landmarks_file):
        landmarks_file = os.path.join(aam_landmark_path, vid_id, 'tmp_frame_lm', "l0{:04d}_lm.mat".format(int(frame_id)))
    assert os.path.isfile(landmarks_file)
    landmarks = scipy.io.loadmat(landmarks_file)['pts']
    img = Image.open(img_path).convert("RGB")
    img_cropped_path = os.path.join(args.cropped_dir, vid_id, f'{frame_id}.jpg')
    os.makedirs(os.path.dirname(img_cropped_path), exist_ok=True)
    if not os.path.exists(img_cropped_path):
        crop_aligned_face = crop_face(img, landmarks)
        crop_aligned_face.save(img_cropped_path)
    return img_cropped_path


def plot_pie(AU_list, pos_freq, neg_freq):
    ploting_labels = [x + '+ {0:.2f}'.format(y) for x, y in zip(AU_list, pos_freq)] + [x + '- {0:.2f}'.format(y) for
                                                                                       x, y in zip(AU_list, neg_freq)]
    cmap = matplotlib.cm.get_cmap('coolwarm')
    colors = [cmap(x) for x in pos_freq] + [cmap(x) for x in neg_freq]
    fracs = np.ones(len(AU_list) * 2)
    plt.pie(fracs, labels=ploting_labels, autopct=None, shadow=False, colors=colors, startangle=78.75)
    plt.title("AUs distribution")
    plt.show()


AU_list = ['AU1', 'AU2', 'AU4', 'AU6', 'AU12', 'AU15', 'AU20', 'AU25']
annot_dir = args.annot_dir
data_file = {}
videos = sorted(os.listdir(annot_dir))
# in total 27 videos
ids = np.random.permutation(len(videos))
videos = [videos[i] for i in ids]
train_videos = videos[:21]
val_videos = videos[21:]
data_file['Training_Set'] = {}
data_file['Validation_Set'] = {}
for video in train_videos:
    aus = read_au(annot_dir + '/{}/{}_au'.format(video, video))
    frames = sorted(glob.glob(os.path.join(args.image_dir, "LeftVideo" + video + "_comp.avi", '*.jpg')))
    frames_id = [int(x.split("/")[-1].split(".")[0].split("-")[-1]) - 1 for x in frames]
    assert len(aus) >= len(frames)
    frames = [frames[id] for id in frames_id]
    frames = [disfa_crop_face(x) for x in frames]
    aus = aus[frames_id]
    data_file['Training_Set'][video] = {'label': aus, 'path': frames}
for video in val_videos:
    aus = read_au(annot_dir + '/{}/{}_au'.format(video, video))
    frames = sorted(glob.glob(os.path.join(args.image_dir, "LeftVideo" + video + "_comp.avi", '*.jpg')))
    frames_id = [int(x.split("/")[-1].split(".")[0].split("-")[-1]) - 1 for x in frames]
    assert len(aus) >= len(frames)
    frames = [frames[id] for id in frames_id]
    frames = [disfa_crop_face(x) for x in frames]
    aus = aus[frames_id]
    data_file['Validation_Set'][video] = {'label': aus, 'path': frames}

if args.vis:
    total_dict = {**data_file['Training_Set'], **data_file['Validation_Set']}
    all_samples = np.concatenate([total_dict[x]['label'] for x in total_dict.keys()], axis=0)
    pos_freq = np.sum(all_samples, axis=0) / all_samples.shape[0]
    neg_freq = -np.sum(all_samples - 1, axis=0) / all_samples.shape[0]
    print("pos_weight:", neg_freq / pos_freq)
    plot_pie(AU_list, pos_freq, neg_freq)
pickle.dump(data_file, open(args.save_path, 'wb'))
