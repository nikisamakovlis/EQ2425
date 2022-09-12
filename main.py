import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils


def read_images(folder):
    # this returns a list of grayscale images
    images = []
    for filename in os.listdir(folder):
        img = cv2.cvtColor(cv2.imread(os.path.join(folder, filename)), cv2.COLOR_BGR2GRAY)
        images.append(img)
    return images


def rotate_image(image):
    # returns a list of rotated images from 0 to 360 degrees.
    rotated_images = []
    (h, w) = image.shape[:2]
    (cX, cY) = (w//2, h//2)  # finding the center point
    d = 0  # degree
    for i in range(int(360/15+1)):
        rotation_matrix = cv2.getRotationMatrix2D((cX, cY), d, 1.0)
        rotated_img = cv2.warpAffine(image, rotation_matrix, (w, h))  # apply rotation on image
        rotated_images.append(rotated_img)
        d += 15

    return rotated_images


def kp_detector_sift(img, n_features=0, contrast_threshold=0.185, edge_threshold=145):
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=n_features, contrastThreshold=contrast_threshold,
                                       edgeThreshold=edge_threshold)
    kp, des = sift.detectAndCompute(img, None)  # keypoints and descriptors
    print(len(kp))  # check how many keypoints
    kp_img = cv2.drawKeypoints(img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('kp_img', kp_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return kp, img


def kp_detector_surf(img, hessian_threshold=7000):
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessian_threshold)
    kp, des = surf.detectAndCompute(img, None)  # keypoints and descriptors
    print(len(kp))  # check how many keypoints
    kp_img = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('kp_img', kp_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return kp, des, kp_img


def main():
    images = read_images('data1')
    kp_sift, kp_img = kp_detector_sift(images[0])
    kp_surf, des_surf, img_surf = kp_detector_surf(images[0])
    rotated_imgs = rotate_image(images[0])


if __name__ == '__main__':
    main()


