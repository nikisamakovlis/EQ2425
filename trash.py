# def fixed_threshold_search(kp1, kp2, t):
#     matched_list = []
#     # matched_kp2 = []
#
#     for threshold in range(0, t, 5):
#         matched_num = 0
#         for ind_kp, original_point in enumerate(kp1):
#             # print(f'Matching with the {ind_kp}-th keypoint from the original image ...')
#             # x_original, y_original = original_point
#             random.shuffle(kp2)
#             for ind_kp_transformed, transformed_point in enumerate(kp2):
#                 # x_transformed, y_transformed = transformed_point
#                 if np.linalg.norm(original_point - transformed_point) <= threshold:
#                     matched_num += 1
#                     # matched_kp2.append(transformed_point)
#                 #     print(
#                 #         f'Matched keypoints detected for threshold {threshold} ! - {ind_kp_transformed}-th keypoint in the transformed image matched with the {ind_kp}-th keypoint in the query image ...',
#                 # )
#                     break
#         print(f'For theshold{threshold}: matched_num:{matched_num}')
#         matched_list.append(matched_num)
#     return matched_list


# def fixed_threshold(kp1, kp2, threshold):
#     matched_num = 0
#     matched_kp2 = []
#     for ind_kp, original_point in enumerate(kp1):
#         # print(f'Matching with the {ind_kp}-th keypoint from the original image ...')
#         # x_original, y_original = original_point
#         random.shuffle(kp2)
#         for ind_kp_transformed, transformed_point in enumerate(kp2):
#             # x_transformed, y_transformed = transformed_point
#             if np.linalg.norm(original_point - transformed_point) <= threshold:
#                 matched_num += 1
#                 matched_kp2.append(transformed_point)
#                 # matched_kp2.append(transformed_point)
#                 #     print(
#                 #         f'Matched keypoints detected for threshold {threshold} ! - {ind_kp_transformed}-th keypoint in the transformed image matched with the {ind_kp}-th keypoint in the query image ...',
#                 # )
#                 break
#     print(f'For theshold{threshold}: matched_num:{matched_num}')
#
#     return matched_kp2