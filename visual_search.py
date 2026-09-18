import cv2
import numpy as np

def locate_with_features(main_image_path, template_image_path, output_path="result_feature.jpg"):
    img_main = cv2.imread(main_image_path)
    img_template = cv2.imread(template_image_path)
    
    if img_main is None or img_template is None:
        print("Error: Could not load images.")
        return

    gray_main = cv2.cvtColor(img_main, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)

    # 1. Initialize SIFT (SIFT handles complex satellite textures much better than ORB)
    sift = cv2.SIFT_create()

    kp_template, des_template = sift.detectAndCompute(gray_template, None)
    kp_main, des_main = sift.detectAndCompute(gray_main, None)

    if des_template is None or des_main is None:
        print("Error: Could not find features.")
        return

    # 2. Use a FLANN-based matcher for high accuracy with SIFT
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Find the top 2 nearest neighbors for each feature point
    matches = flann.knnMatch(des_template, des_main, k=2)

    # 3. Apply Lowe's Ratio Test (CRITICAL: Filters out false matches)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 4. Draw bounding box if we have reliable matches
    MIN_MATCH_COUNT = 4
    if len(good_matches) >= MIN_MATCH_COUNT:
        print(f"Success! Found {len(good_matches)} strictly verified matching points.")

        src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_main[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # RANSAC will ignore any remaining outliers automatically
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        h, w = gray_template.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        # Draw a thick bright green box around the house
        result_img = cv2.polylines(img_main, [np.int32(dst)], True, (0, 255, 0), 4, cv2.LINE_AA)

        cv2.imwrite(output_path, result_img)
        print(f"Saved exact boundary result to {output_path}")
    else:
        print(f"Not enough high-quality matches found ({len(good_matches)}/{MIN_MATCH_COUNT}).")

# Run the function
locate_with_features("big.jpg", "small.jpg")
