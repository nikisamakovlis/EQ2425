import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def read_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.cvtColor(cv2.imread(os.path.join(folder, filename)), cv2.COLOR_BGR2RGB)
        images.append(img)
    return images


def kp_detector_sift(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray, None)
    img = cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return kp, img


def main():
    images = read_images('data1')
    kp_sift, kp_img = kp_detector_sift(images[0])
    plt.imshow(kp_img)
    plt.show()

main()
if __name__ == '__main()__':
    main()


