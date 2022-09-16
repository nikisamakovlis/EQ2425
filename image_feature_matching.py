from main import *
import random


def plot_threshold(matched_list, threshold):
    x = np.arange(0, threshold)
    print(x)
    y = matched_list
    plt.plot(x, y)
    plt.title('Threshold vs Matched Points')
    plt.xlabel('Threshold')
    plt.ylabel('Number of Matched Points')
    plt.show()


def fixed_threshold_search(des1, des2, t):
    # create BFMatcher object
    bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    # bf = cv2.BFMatcher()
    # matches = bf.knnMatch(des1, des2, k=405)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    # print([x.distance for x in matches])

    good = []
    good_count = []
    for threshold in range(0, t):
        # Apply ratio test
        good_thre = []
        count = 0
        for m in matches:
            if m.distance < threshold:
                # print(m)
                good_thre.append(m)
                count += 1
        good.append(good_thre)
        good_count.append(count)

    return good, good_count


def nearest_neighbor(des1, des2):
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=1)
    matches = [i[0] for i in matches]
    return matches


def nearest_neighbor_dr_search(des1, des2, t):
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test

    good = []
    good_count = []
    for threshold in range(0, 11*t):
        threshold = threshold/10
        good_thre = []
        count = 0
        for m, n in matches:
            if m.distance < threshold * n.distance:
                good_thre.append(m)
                count += 1
        good.append(good_thre)
        good_count.append(count)

    return good, good_count


def part3_b(kp1, kp2, des1, des2, images):
    t = 404
    good, good_count = fixed_threshold_search(des1, des2, t)
    print(good_count)
    max_value = max(good_count)
    ind_max = good_count.index(max_value)
    print(f'max_value{max_value}, ind_max{ind_max}')
    plot_threshold(good_count, t)

    # #print(good)
    # print(good[-1])
    # # Draw matches when the threshold is set in range(0,404,50)
    for ind in range(0, 404, 50):
        img = cv2.drawMatches(images[0], kp1, images[1], kp2, good[ind], None, flags=2)
        cv2.imwrite(f'FT_thres{ind}.png', img)
        # cv2.imshow('Feature matching', img3), plt.show()
        # cv2.waitKey(0)
        # # closing all open windows
        # cv2.destroyAllWindows()


def part3_c(kp1, kp2, des1, des2, images):
    good = nearest_neighbor(des1, des2)
    img = cv2.drawMatches(images[0], kp1, images[1], kp2, good, None, flags=2)
    cv2.imwrite(f'NN1.png', img)

def part3_d(kp1, kp2, des1, des2, images, tested_keypoint_detector):
    t = 1
    good, good_count = nearest_neighbor_dr_search(des1, des2, t)

    for ind in range(0, 11):
        img = cv2.drawMatches(images[0], kp1, images[1], kp2, good[ind], None, flags=2)
        cv2.imwrite(f'KNNR_thres{ind/10}_{tested_keypoint_detector}.png', img)



def main():
    images = read_images('data1')
    tested_keypoint_detector = 'surf'
    if tested_keypoint_detector == 'sift':
        kp1, points1, des1, kp_img1 = kp_detector_sift(images[0])
        kp2, points2, des2, kp_img2 = kp_detector_sift(images[1])

    else:
        kp1, points1, des1, img_surf1 = kp_detector_surf(images[0])
        kp2, points2, des2, img_surf2 = kp_detector_surf(images[1])

    # # part3-b
    # part3_b(kp1, kp2, des1, des2, images)

    # part3-c
    #part3_c(kp1, kp2, des1, des2, images)

    # part3-d
    part3_d(kp1, kp2, des1, des2, images, tested_keypoint_detector)











if __name__ == '__main__':
    main()





