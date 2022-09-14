from utils import *


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


def kp_detector_general(images, tested_keypoint_detector):

    kp_list = []
    points_list = []
    kd_list = []
    print(len(images))
    for img in images:
        if tested_keypoint_detector == 'surf':
            kp, points, _, kd = kp_detector_surf(img)
        else:
            kp, points, kd = kp_detector_sift(img)
        kp_list.append(kp)
        points_list.append(points)
        kd_list.append(kd)
    print(len(points_list))

    return kp_list, kd_list, points_list


def calc_repeatability(rotated_points_list):
    repeatability_list = [0] * len(rotated_points_list[1:])
    original_points = rotated_points_list[0]
    for ind, rotated_img_points in enumerate(rotated_points_list[1:]):  # for each rotated image
        print(f'Searching in the {ind}-th rotated image ...')
        for ind_kp, original_point in enumerate(original_points):
            # print(f'Matching with the {ind_kp}-th keypoint from the original image ...')
            x_original, y_original = original_point
            for ind_kp_rotated, rotated_point in enumerate(rotated_img_points):
                # new_origin_x = math.sqrt(img_height**2+img_width**2) / 2 * math.cos(math.radians(45+15*(ind_kp_rotated+1))) \
                #                + img_height * math.cos(math.radians(90-15*(ind_kp_rotated+1)))
                # new_origin_y = img_width * math.cos(math.radians(90-15*(ind_kp_rotated+1)))
                # x_rotated, y_rotated = rotate_point((new_origin_x, new_origin_y), rotated_point, math.radians(-15*(ind_kp_rotated+1)))
                x_rotated, y_rotated = rotated_point
                if np.abs(x_original - x_rotated) <= 2 and np.abs(y_original - y_rotated) <= 2:
                    repeatability_list[ind] += 1
                    print(
                        f'Matched keypoints detected ! - {ind_kp_rotated}-th in the rotated image matched with the {ind_kp}-th keypoint in the original image ...')
                    break
    print(repeatability_list)
    # surf: [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 6, 0, 0, 0, 0, 0, 402]
    # sift: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 2, 0, 0, 403]
    repeatability_list = [i / len(original_points) for i in repeatability_list]
    print(repeatability_list)
    # surf: [0.0024875621890547263, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0024875621890547263, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0024875621890547263, 0.0, 0.0, 0.0, 0.014925373134328358, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    # sift: [0.0024813895781637717, 0.0, 0.0, 0.0, 0.0, 0.0024813895781637717, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.004962779156327543, 0.007444168734491315, 0.0, 0.0, 0.004962779156327543, 0.0, 0.0, 1.0]
    return repeatability_list


def main():
    images = read_images('data1')
    # part2-a
    #kp_sift, points, kp_img = kp_detector_sift(images[0])
    #kp_surf, points, des_surf, img_surf = kp_detector_surf(images[0])

    # part2-b
    tested_keypoint_detector = 'sift'
    rotated_imgs, inverted_matrices = rotate_image(images[0])
    rotated_kp_list, rotated_kd_list, rotated_points_list = kp_detector_general(rotated_imgs, tested_keypoint_detector)

    transformed_points_list = []
    for idx, points_list in enumerate(rotated_points_list):
        points_list = np.array(points_list)
        transformed_points = transform_rotation(inverted_matrices[idx], points_list)
        transformed_points_list.append(transformed_points)

    # repeatability_list = calc_repeatability(transformed_points, images[0].shape[:2][-1], images[0].shape[:2][-2])
    repeatability_list = calc_repeatability(transformed_points_list)
    #plot_images(columns=4, rows=6, imgs=rotated_kd_list, tested_keypoint_detector=tested_keypoint_detector, augmentation='rotated', figsize=(40, 60))

    # # part 2-c
    # scaled_images = scale_image(images[0])
    # scaled_kp_list, scaled_kd_list, scaled_points_list = kp_detector_general(scaled_images, tested_keypoint_detector)
    # #scaled_repeatability_list = calc_repeatability(scaled_points_list)
    # plot_images(columns=3, rows=3, imgs=scaled_kd_list, tested_keypoint_detector=tested_keypoint_detector, augmentation='scaled', figsize=(60, 60))

    # print(rotate_point((0,0), (1,1), math.radians(-45)))

    # print(rotate_point((0, 0), (0,1), math.radians(-90)))










if __name__ == '__main__':
    main()


