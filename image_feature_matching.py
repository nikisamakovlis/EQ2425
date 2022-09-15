from main import *
import random


def fixed_threshold_search(kp1, kp2, t):
    matched_list = []
    # matched_kp2 = []

    for threshold in range(0, t, 5):
        matched_num = 0
        for ind_kp, original_point in enumerate(kp1):
            # print(f'Matching with the {ind_kp}-th keypoint from the original image ...')
            # x_original, y_original = original_point
            random.shuffle(kp2)
            for ind_kp_transformed, transformed_point in enumerate(kp2):
                # x_transformed, y_transformed = transformed_point
                if np.linalg.norm(original_point - transformed_point) <= threshold:
                    matched_num += 1
                    # matched_kp2.append(transformed_point)
                #     print(
                #         f'Matched keypoints detected for threshold {threshold} ! - {ind_kp_transformed}-th keypoint in the transformed image matched with the {ind_kp}-th keypoint in the query image ...',
                # )
                    break
        print(f'For theshold{threshold}: matched_num:{matched_num}')
        matched_list.append(matched_num)
    return matched_list


def fixed_threshold(kp1, kp2, threshold):
    matched_num = 0
    matched_kp2 = []
    for ind_kp, original_point in enumerate(kp1):
        # print(f'Matching with the {ind_kp}-th keypoint from the original image ...')
        # x_original, y_original = original_point
        random.shuffle(kp2)
        for ind_kp_transformed, transformed_point in enumerate(kp2):
            # x_transformed, y_transformed = transformed_point
            if np.linalg.norm(original_point - transformed_point) <= threshold:
                matched_num += 1
                matched_kp2.append(transformed_point)
                # matched_kp2.append(transformed_point)
                #     print(
                #         f'Matched keypoints detected for threshold {threshold} ! - {ind_kp_transformed}-th keypoint in the transformed image matched with the {ind_kp}-th keypoint in the query image ...',
                # )
                break
    print(f'For theshold{threshold}: matched_num:{matched_num}')

    return matched_kp2


def nearest_neighbor():
    return


def nearest_neighbor_dr():
    return


def main():
    images = read_images('data1')
    tested_keypoint_detector = 'sift'
    if tested_keypoint_detector == 'sift':
        kp1, points1, des_sift1, kp_img1 = kp_detector_sift(images[0])
        kp2, points2, des_sift2, kp_img2 = kp_detector_sift(images[1])

    else:
        kp1, points1, des_surf1, img_surf1 = kp_detector_surf(images[0])
        kp2, points2, des_surf2, img_surf2 = kp_detector_surf(images[1])

    # # part3-a
    #threshold = 400
    #matched_list = fixed_threshold_search(points1, points2, threshold)
    #plot_threshold(matched_list, threshold)
    #matched_kp2 = fixed_threshold(points1, points2, threshold)

    # create BFMatcher object
    bf = cv2.BFMatcher(crossCheck=True)

    # Match descriptors.
    matches = bf.match(des_sift1, des_sift2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 10 matches.
    img3 = cv2.drawMatches(images[0], kp1, images[1], kp2, matches[:403], None, flags=2)

    plt.imshow(img3), plt.show()

    # matched_image = cv2.drawMatches(images[0], kp1, images[1], kp2, matches, None, flags=2)
    #
    # plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
    # plt.show()




def plot_threshold(matched_list, threshold):
    x = np.arange(0, threshold, 5)
    print(x)
    y = matched_list
    plt.plot(x, y)
    plt.title('Threshold vs Matched Points')
    plt.xlabel('Threshold')
    plt.ylabel('Number of Matched Points')
    plt.show()




if __name__ == '__main__':
    main()





