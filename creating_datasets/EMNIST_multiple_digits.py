
# NB! This code is from this Github repository: https://github.com/shaohua0116/MultiDigitMNIST/tree/master
# I changed the EMNIST Digits import, changed the colors of the digits and added finding bounding boxes.


import subprocess
import os
import os.path as osp
import numpy as np
from imageio import imwrite
import argparse
import tensorflow_datasets as tfds
from einops import rearrange
import cv2
import random
import matplotlib.pyplot as plt
from PIL import Image


def extract_mnist():

    # Loading in emnist (train
    images_train, labels_train = tfds.as_numpy(tfds.load(
        'emnist/digits',
        split='train',
        batch_size=-1,
        as_supervised=True,
    ))

    # Test data
    images_test, labels_test = tfds.as_numpy(tfds.load(
        'emnist/digits',
        split='test',
        batch_size=-1,
        as_supervised=True,
    ))

    images_train = images_train / 255
    images_train = rearrange(images_train, 'b h w c -> b w h c')
    images_test = images_test / 255.0
    images_test = rearrange(images_test, 'b h w c -> b w h c')

    return np.concatenate((images_train, images_test)), \
        np.concatenate((labels_train, labels_test))


def sample_coordinate(high, size):
    if high > 0:
        return np.random.randint(high, size=size)
    else:
        return np.zeros(size).astype(np.int)

def bounding_box (image, xstart, ystart):
    
    non_zero = np.nonzero(image)
    min_values = np.min(non_zero, axis=1) 
    maximal_values = np.max(non_zero, axis=1) 

    width = maximal_values[1] - min_values[1]  # xmax - xmin
    heigth = maximal_values[0] - min_values[0] # ymax - ymin
    xcentre = min_values[1] + width / 2
    ycentre = min_values[0] + heigth / 2

    return xstart+xcentre, ystart+ycentre, width, heigth


def generator(config):
    # extract mnist images and labels
    image, label = extract_mnist()
    h, w = image.shape[1:3]

    # split: train, val, test
    rs = np.random.RandomState(config.random_seed)
    num_original_class = len(np.unique(label))
    num_class = len(np.unique(label))**config.num_digit
    classes = list(np.array(range(num_class)))
    rs.shuffle(classes)
    num_train, num_val, num_test = [
            int(float(ratio)/np.sum(config.train_val_test_ratio)*num_class)
            for ratio in config.train_val_test_ratio]
    train_classes = classes[:num_train]
    val_classes = classes[num_train:num_train+num_val]
    test_classes = classes[num_train+num_val:]

    # label index
    indexes = []
    for c in range(num_original_class):
        indexes.append(list(np.where(label == c)[0]))

    # generate images for every class
    assert config.image_size[1]//config.num_digit >= w
    np.random.seed(config.random_seed)

    if not os.path.exists(config.multimnist_path):
        os.makedirs(config.multimnist_path)

    split_classes = [train_classes, val_classes, test_classes]
    count = 1
    
    for i, split_name in enumerate(['train', 'val', 'test']):

        images_path = osp.join(osp.join(config.multimnist_path, 'images'), split_name)
        labels_path = osp.join(osp.join(config.multimnist_path, 'labels'), split_name)
        print('Generat images for {} at {}'.format(split_name, images_path))
        
        if not os.path.exists(images_path):
            os.makedirs(images_path)
        if not os.path.exists(labels_path):
            os.makedirs(labels_path)
            
        for j, current_class in enumerate(split_classes[i]):
            class_str = str(current_class)
            class_str = '0'*(config.num_digit-len(class_str))+class_str
            print('(progress: {}/{})'.format(count, len(classes)))
                
            for k in range(config.num_image_per_class):
                
                # sample images
                digits = [int(class_str[l]) for l in range(config.num_digit)]
                imgs = [np.squeeze(image[np.random.choice(indexes[d])]) for d in digits]
                background = np.zeros((config.image_size)).astype(float)
                
                # sample coordinates
                ys = sample_coordinate(config.image_size[0]-h, config.num_digit)
                xs = sample_coordinate(config.image_size[1]//config.num_digit-w,
                                       size=config.num_digit)
                xs = [l*config.image_size[1]//config.num_digit+xs[l]
                      for l in range(config.num_digit)]

                # combine images
                for i in range(config.num_digit):
                    background[ys[i]:ys[i]+h, xs[i]:xs[i]+w] = imgs[i]

                # Find bounding boxes for digits
                new_image_bboxes = []
                for b in range(len(imgs)):
                    xcentre, ycentre, width, height = bounding_box(imgs[b], xs[b], ys[b])
                    new_image_bboxes.append([int(class_str[b]), xcentre, ycentre, width, height])

                # write the image
                background = abs(background - 1)
                background = np.squeeze(background)

                blueOrBlack = random.random()
                if blueOrBlack > 0.5:
                    cm = plt.get_cmap('Blues_r') # Color map
                    background = cm(background)

                # Saving the image
                image_path = osp.join(images_path, '{}_{}.png'.format(class_str, k))
                Image.fromarray((background[:, :] * 255).astype(np.uint8)).save(image_path)


                # write the label and bounding boxes
                label_path = osp.join(labels_path, '{}_{}.txt'.format(class_str, k))
                with open(label_path, "w") as file:
                    for label_bbox in new_image_bboxes:
                        file.write(str(label_bbox[0]) + " " + 
                                   str(label_bbox[1]/config.image_size[1]) + " " + 
                                   str(label_bbox[2]/config.image_size[0]) + " " + 
                                   str(label_bbox[3]/config.image_size[1]) + " " + 
                                   str(label_bbox[4]/config.image_size[0]) + "\n")
                    file.close()

            count += 1

    return image, label, indexes


def argparser():

    def str2bool(v):
        return v.lower() == 'true'

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--multimnist_path', type=str, default='./datasets/multimnist')
    parser.add_argument('--num_digit', type=int, default=3)
    parser.add_argument('--train_val_test_ratio', type=int, nargs='+', default=[64, 16, 20], help='percentage')
    parser.add_argument('--image_size', type=int, nargs='+', default=[64, 96])
    parser.add_argument('--num_image_per_class', type=int, default=10)
    parser.add_argument('--random_seed', type=int, default=123)
    config = parser.parse_args()
    return config


def main():

    config = argparser()
    assert len(config.train_val_test_ratio) == 3
    assert sum(config.train_val_test_ratio) == 100
    assert len(config.image_size) == 2
    generator(config)


if __name__ == '__main__':
    main()
