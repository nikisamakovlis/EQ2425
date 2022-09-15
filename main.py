from utils import *


def kp_detector_sift(img, n_features=0, contrast_threshold=0.185, edge_threshold=145):
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=n_features, contrastThreshold=contrast_threshold,
                                       edgeThreshold=edge_threshold)
    kp, des = sift.detectAndCompute(img, None)  # keypoints and descriptors
    #print(len(kp))
    kp_img = cv2.drawKeypoints(img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    point_kp = cv2.KeyPoint_convert(kp)
    return kp, point_kp, des, kp_img


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
    des_list = []
    print(len(images))
    for img in images:
        if tested_keypoint_detector == 'surf':
            kp, points, des, kd = kp_detector_surf(img)
        else:
            kp, points, des, kd = kp_detector_sift(img)
        kp_list.append(kp)
        points_list.append(points)
        des_list.append(des)
        kd_list.append(kd)
    print(len(points_list))

    return kp_list, kd_list, des_list, points_list


def calc_repeatability(transformed_points_list, original_points):
    print(len(original_points))
    repeatability_list = [0] * len(transformed_points_list)
    #original_points = rotated_points_list[0]
    for ind, transformed_img_points in enumerate(transformed_points_list):  # for each rotated image
        print(f'Searching in the {ind}-th transformed image ...')
        for ind_kp, original_point in enumerate(original_points):
            # print(f'Matching with the {ind_kp}-th keypoint from the original image ...')
            x_original, y_original = original_point
            for ind_kp_rotated, transformed_point in enumerate(transformed_img_points):
                x_transformed, y_transformed = transformed_point
                if np.abs(x_original - x_transformed) <= 2 and np.abs(y_original - y_transformed) <= 2:
                    repeatability_list[ind] += 1
                    print(
                        f'Matched keypoints detected ! - {ind_kp_rotated}-th in the transformed image matched with the {ind_kp}-th keypoint in the original image ...')
                    break
    print(repeatability_list)
    repeatability_list = [i / len(original_points) for i in repeatability_list]
    print(repeatability_list)
    return repeatability_list


def main():
    images = read_images('data1')
    # part2-a
    tested_keypoint_detector = 'surf'
    if tested_keypoint_detector == 'sift':
        kp_sift, points, des_sift, kp_img = kp_detector_sift(images[0])
    else:
        kp_surf, points, des_surf, img_surf = kp_detector_surf(images[0])

    # # part2-b
    # rotated_imgs, inverted_matrices = rotate_image(images[0])
    # rotated_kp_list, rotated_kd_list, rotated_points_list = kp_detector_general(rotated_imgs, tested_keypoint_detector)
    #
    # transformed_points_list = []
    # for idx, points_list in enumerate(rotated_points_list):
    #     points_list = np.array(points_list)
    #     transformed_points = transform_rotation(inverted_matrices[idx], points_list)
    #     transformed_points_list.append(transformed_points)
    #
    # print(transformed_points_list[0])
    # print(points)
    #
    # repeatability_list = calc_repeatability(transformed_points_list, points)

    # # part 2-c
    scaled_images = scale_image(images[0])
    scaled_kp_list, scaled_kd_list, scaled_des_list, scaled_points_list = kp_detector_general(scaled_images, tested_keypoint_detector)

    # print(scaled_points_list[1])
    transformed_scaled_points_list = []
    for idx, points_list in enumerate(scaled_points_list):
        scaled_points_each_img_list = []
        for point in points_list:
            x, y = point[0], point[1]
            new_x = x/(1.2**idx)
            new_y = y/(1.2**idx)
            scaled_points_each_img_list.append([new_x, new_y])
        transformed_scaled_points_list.append(scaled_points_each_img_list)
    # print(transformed_scaled_points_list[1])
    print(len(transformed_scaled_points_list))

    scaled_repeatability_list = calc_repeatability(transformed_scaled_points_list, points)
    plot_repeatability('scaled', scaled_repeatability_list)

    # # plot_images(columns=3, rows=3, imgs=scaled_kd_list, tested_keypoint_detector=tested_keypoint_detector, augmentation='scaled', figsize=(60, 60))
    #
    # # print(rotate_point((0,0), (1,1), math.radians(-45)))
    #
    # # print(rotate_point((0, 0), (0,1), math.radians(-90)))

if __name__ == '__main__':
    main()


