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
    for i in range(int(360/15)+1):
        rotation_matrix = cv2.getRotationMatrix2D((cX, cY), d, 1.0)
        abs_cos, abs_sin = abs(rotation_matrix[0,0]), abs(rotation_matrix[0,1])
        bound_w = int(h * abs_sin + w * abs_cos)
        bound_h = int(h * abs_cos + w * abs_sin)

        # subtract old image center (bringing image back to origo) and adding the new image center coordinates
        rotation_matrix[0, 2] += bound_w / 2 - cX
        rotation_matrix[1, 2] += bound_h / 2 - cY

        rotated_img = cv2.warpAffine(image, rotation_matrix, (bound_w, bound_h))  # apply rotation on image
        rotated_images.append(rotated_img)
        d += 15

    return rotated_images


def kp_detector_sift(img, n_features=0, contrast_threshold=0.185, edge_threshold=145):
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=n_features, contrastThreshold=contrast_threshold,
                                       edgeThreshold=edge_threshold)
    kp, des = sift.detectAndCompute(img, None)  # keypoints and descriptors
    #print(len(kp))
    kp_img = cv2.drawKeypoints(img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    point_kp = cv2.KeyPoint_convert(kp)
    return kp, point_kp, kp_img


def kp_detector_surf(img, hessian_threshold=7000):
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessian_threshold)
    kp, des = surf.detectAndCompute(img, None)  # keypoints and descriptors
    #print(len(kp))  # check how many keypoints
    kp_img = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    point_kp = cv2.KeyPoint_convert(kp)
    # print(point_kp)
    # breakpoint()

    return kp, point_kp, des, kp_img

def plot_images(columns, rows, imgs, tested_keypoint_detector):
    fig = plt.figure(figsize=(40, 60))
    for i in range(1, columns * rows + 1):
        img = imgs[i]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    # plt.show()
    if tested_keypoint_detector == 'surf':
        save_path = 'Rotated_images_surf1.png'
    else:
        save_path = 'Rotated_images_sift1.png'
    plt.savefig(save_path, dpi=300)


def main():
    images = read_images('data1')
    # part2-a
    kp_sift, points, kp_img = kp_detector_sift(images[0])
    kp_surf, points, des_surf, img_surf = kp_detector_surf(images[0])

    # part2-b
    rotated_imgs = rotate_image(images[0])
    tested_keypoint_detector = 'surf'  # 'sift' or 'surf'

    rotated_kp_list = []
    rotated_points_list = []
    rotated_kd_list = []
    print(len(rotated_imgs))
    for img in rotated_imgs:
        if tested_keypoint_detector == 'surf':
            rotated_kp, rotated_points, _, rotated_kd = kp_detector_surf(img)
        else:
            rotated_kp, rotated_points, rotated_kd = kp_detector_sift(img)
        rotated_kp_list.append(rotated_kp)
        rotated_points_list.append(rotated_points)
        rotated_kd_list.append(rotated_kd)
    print(len(rotated_points_list))

    # plot_images(columns=4, rows=6, imgs=rotated_kd_list, tested_keypoint_detector=tested_keypoint_detector)

    repeatability_list = [0]*len(rotated_points_list[1:])
    original_points = rotated_points_list[0]
    for ind, rotated_img_points in enumerate(rotated_points_list[1:]):  # for each rotated image
        print(f'Searching in the {ind}-th rotated image ...')
        for ind_kp, original_point in enumerate(original_points):
            # print(f'Matching with the {ind_kp}-th keypoint from the original image ...')
            x_original, y_original = original_point
            for ind_kp_rotated, rotated_point in enumerate(rotated_img_points):
                x_rotated, y_rotated = rotated_point
                if np.abs(x_original - x_rotated) <= 2 and np.abs(y_original - y_rotated) <= 2:
                    repeatability_list[ind] += 1
                    print(f'Matched keypoints detected ! - {ind_kp_rotated}-th in the rotated image matched with the {ind_kp}-th keypoint in the original image ...')
                    break
    # print(repeatability_list)
    # surf: [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 6, 0, 0, 0, 0, 0, 402]
    # sift: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 2, 0, 0, 403]
    repeatability_list = [i/len(original_points) for i in repeatability_list]
    print(repeatability_list)
    # surf: [0.0024875621890547263, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0024875621890547263, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0024875621890547263, 0.0, 0.0, 0.0, 0.014925373134328358, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    # sift: [0.0024813895781637717, 0.0, 0.0, 0.0, 0.0, 0.0024813895781637717, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.004962779156327543, 0.007444168734491315, 0.0, 0.0, 0.004962779156327543, 0.0, 0.0, 1.0]



if __name__ == '__main__':
    main()


