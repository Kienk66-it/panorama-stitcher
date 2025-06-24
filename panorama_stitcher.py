import cv2
import numpy as np
import numba
from numba import njit
from concurrent.futures import ThreadPoolExecutor
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import threading
import os
from datetime import datetime

####################
# Camera Manager #
####################

class CameraManager:
    def __init__(self):
        self.cap = None
        self.available_cameras = []
        self.current_camera = -1
        self.is_running = False
        self.captured_images = []
        self.temp_dir = "/tmp/panorama_captures"
        
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
    
    def find_available_cameras(self):
        self.available_cameras = []
        for i in range(6):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.available_cameras.append(i)
                    print(f"📷 Found camera at index {i}")
            cap.release()
        
        if not self.available_cameras:
            print("❌ No cameras found")
        else:
            print(f"✅ Found {len(self.available_cameras)} camera(s)")
        
        return self.available_cameras
    
    def open_camera(self, camera_index):
        if self.cap is not None:
            self.close_camera()
        
        # Sử dụng GStreamer pipeline cho camera USB với độ phân giải HD
        gst_str = (
            f"v4l2src device=/dev/video{camera_index} ! "
            "video/x-raw, width=1280, height=720, framerate=30/1 ! "
            "videoconvert ! appsink"
        )
        self.cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.current_camera = camera_index
            self.is_running = True
            print(f"📷 Camera {camera_index} opened (GStreamer HD)")
            return True
        else:
            print(f"❌ Failed to open camera {camera_index} (GStreamer fallback to OpenCV)")
            print("⚠️ GStreamer pipeline failed, falling back to OpenCV backend. If you want better performance, please check your GStreamer installation.")
            self.cap = cv2.VideoCapture(camera_index)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.current_camera = camera_index
                self.is_running = True
                print(f"📷 Camera {camera_index} opened (OpenCV fallback)")
                return True
            else:
                print(f"❌ Failed to open camera {camera_index}")
                return False
    
    def read_frame(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    
    def capture_image(self):
        frame = self.read_frame()
        if frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"capture_{timestamp}.jpg"
            filepath = os.path.join(self.temp_dir, filename)
            
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, frame_bgr)
            
            self.captured_images.append({
                'path': filepath,
                'image': frame.copy(),
                'timestamp': timestamp
            })
            
            print(f"📸 Captured image {len(self.captured_images)}: {filename}")
            return frame
        return None
    
    def get_captured_count(self):
        return len(self.captured_images)
    
    def get_captured_paths(self):
        return [img['path'] for img in self.captured_images]
    
    def clear_captures(self):
        for img in self.captured_images:
            if os.path.exists(img['path']):
                os.remove(img['path'])
        self.captured_images = []
        print("🗑️ Cleared all captured images")
    
    def close_camera(self):
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        print("📷 Camera closed")

####################
# Optimized Functions #
####################

def adaptive_contrast(img, low_thresh=60, high_thresh=180, alpha_low=1.2, alpha_high=0.85):
    """Increase contrast if mean is low, decrease if mean is high."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    if mean < low_thresh:
        img = cv2.convertScaleAbs(img, alpha=alpha_low, beta=0)
    elif mean > high_thresh:
        img = cv2.convertScaleAbs(img, alpha=alpha_high, beta=0)
    return img

def enhance_features_preprocess(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    img_clahe = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    img_clahe = adaptive_contrast(img_clahe)
    return img_clahe

def read_image_optimized(img_path, max_size=1920):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # img = enhance_features_preprocess(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_gray, img, img_rgb

def SIFT_optimized(img_gray, max_features=5000):
    sift = cv2.SIFT_create(nfeatures=max_features, 
                          contrastThreshold=0.04,
                          edgeThreshold=10)
    kp, des = sift.detectAndCompute(img_gray, None)
    return kp, des

def KNN_matcher_optimized(des1, des2, ratio_threshold=0.7):
    if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
        return np.array([])
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    try:
        knn_matches = flann.knnMatch(des1, des2, k=2)
    except cv2.error:
        return np.array([])
    
    good_matches = []
    for match_pair in knn_matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
    
    return good_matches

def compute_homography_robust(matches, method='ransac', confidence=0.99):
    if len(matches) < 4:
        return None
    
    src_pts = np.float32([kp1.pt for kp1, kp2 in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2.pt for kp1, kp2 in matches]).reshape(-1, 1, 2)
    
    if method == 'linear':
        H, _ = cv2.findHomography(src_pts, dst_pts, method=0)
    elif method == 'ransac':
        H, mask = cv2.findHomography(src_pts, dst_pts, 
                                   method=cv2.RANSAC,
                                   ransacReprojThreshold=3.0,
                                   confidence=confidence)
        if H is not None:
            inliers = np.sum(mask)
            print(f"RANSAC: {inliers}/{len(matches)} inliers")
    else:
        raise ValueError("Method must be 'linear' or 'ransac'")
    
    return H

@njit(parallel=False)
def feather_blend_numba(img1, img2, feather_width=50):
    h, w, c = img1.shape
    result = np.zeros_like(img1)
    
    overlap_start = w
    overlap_end = -1
    
    for j in range(w):
        has_img1 = False
        has_img2 = False
        
        for i in range(h):
            pixel1_sum = img1[i, j, 0] + img1[i, j, 1] + img1[i, j, 2]
            pixel2_sum = img2[i, j, 0] + img2[i, j, 1] + img2[i, j, 2]
            
            if pixel1_sum > 0.01:
                has_img1 = True
            if pixel2_sum > 0.01:
                has_img2 = True
                
            if has_img1 and has_img2:
                break
        
        if has_img1 and has_img2:
            if j < overlap_start:
                overlap_start = j
            if j > overlap_end:
                overlap_end = j
    
    if overlap_start >= w or overlap_end < 0:
        for i in range(h):
            for j in range(w):
                pixel1_sum = img1[i, j, 0] + img1[i, j, 1] + img1[i, j, 2]
                pixel2_sum = img2[i, j, 0] + img2[i, j, 1] + img2[i, j, 2]
                
                if pixel1_sum > 0.01 and pixel2_sum > 0.01:
                    for k in range(c):
                        result[i, j, k] = (img1[i, j, k] + img2[i, j, k]) * 0.5
                elif pixel1_sum > 0.01:
                    for k in range(c):
                        result[i, j, k] = img1[i, j, k]
                elif pixel2_sum > 0.01:
                    for k in range(c):
                        result[i, j, k] = img2[i, j, k]
        return result
    
    overlap_width = overlap_end - overlap_start + 1
    
    for i in range(h):
        for j in range(w):
            pixel1_sum = img1[i, j, 0] + img1[i, j, 1] + img1[i, j, 2]
            pixel2_sum = img2[i, j, 0] + img2[i, j, 1] + img2[i, j, 2]
            
            if pixel1_sum > 0.01 and pixel2_sum > 0.01:
                if overlap_start <= j <= overlap_end and overlap_width > 1:
                    position_ratio = (j - overlap_start) / (overlap_width - 1)
                    weight1 = 1.0 - position_ratio
                    weight2 = position_ratio
                    
                    weight1 = max(0.0, min(1.0, weight1))
                    weight2 = max(0.0, min(1.0, weight2))
                    
                    for k in range(c):
                        result[i, j, k] = img1[i, j, k] * weight1 + img2[i, j, k] * weight2
                else:
                    for k in range(c):
                        result[i, j, k] = (img1[i, j, k] + img2[i, j, k]) * 0.5
            elif pixel1_sum > 0.01:
                for k in range(c):
                    result[i, j, k] = img1[i, j, k]
            elif pixel2_sum > 0.01:
                for k in range(c):
                    result[i, j, k] = img2[i, j, k]
    
    return result

def create_gaussian_pyramid(img, levels=6):
    pyramid = [img.copy()]
    current = img.copy()
    
    for i in range(levels - 1):
        current = cv2.pyrDown(current)
        pyramid.append(current)
    
    return pyramid

def create_laplacian_pyramid(gaussian_pyramid):
    laplacian_pyramid = []
    
    for i in range(len(gaussian_pyramid) - 1):
        size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
        upsampled = cv2.pyrUp(gaussian_pyramid[i+1], dstsize=size)
        laplacian = cv2.subtract(gaussian_pyramid[i], upsampled)
        laplacian_pyramid.append(laplacian)
    
    laplacian_pyramid.append(gaussian_pyramid[-1])
    
    return laplacian_pyramid

def reconstruct_from_laplacian_pyramid(laplacian_pyramid):
    result = laplacian_pyramid[-1].copy()
    
    for i in range(len(laplacian_pyramid) - 2, -1, -1):
        size = (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0])
        result = cv2.pyrUp(result, dstsize=size)
        result = cv2.add(result, laplacian_pyramid[i])
    
    return result

def create_mask_pyramid(shape, seam_x, levels=6):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    
    for y in range(h):
        for x in range(w):
            if x < seam_x:
                mask[y, x] = 1.0
            else:
                dist = abs(x - seam_x)
                if dist < 50:
                    mask[y, x] = 1.0 - (dist / 50.0)
                else:
                    mask[y, x] = 0.0
    
    pyramid = create_gaussian_pyramid(mask, levels)
    return pyramid

def multiband_blend(img1, img2, seam_x, levels=6):
    gauss1 = create_gaussian_pyramid(img1, levels)
    gauss2 = create_gaussian_pyramid(img2, levels)
    
    lap1 = create_laplacian_pyramid(gauss1)
    lap2 = create_laplacian_pyramid(gauss2)
    
    mask_pyramid = create_mask_pyramid(img1.shape, seam_x, levels)
    
    blended_pyramid = []
    for i in range(levels):
        if len(lap1[i].shape) == 3:
            mask = np.stack([mask_pyramid[i]] * lap1[i].shape[2], axis=2)
        else:
            mask = mask_pyramid[i]
        
        blended = lap1[i] * mask + lap2[i] * (1 - mask)
        blended_pyramid.append(blended)
    
    result = reconstruct_from_laplacian_pyramid(blended_pyramid)
    return result

@njit
def is_valid_rectangle(mask, top, bottom, left, right):
    """
    Kiểm tra xem vùng hình chữ nhật có chứa toàn bộ nội dung (100% không có pixel đen)
    """
    rect_mask = mask[top:bottom, left:right]
    if rect_mask.size == 0:
        return False
    return np.all(rect_mask == 1)  # Tất cả pixel phải là 1 (nội dung)

def _find_content_rect_on_downscaled(img_small, threshold=0.01):
    """
    Tìm hình chữ nhật lớn nhất chứa 100% nội dung trên ảnh đã giảm kích thước
    """
    if len(img_small.shape) == 3:
        gray = cv2.cvtColor(img_small, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_small.copy()
    
    mask = (gray > threshold * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return (0, 0, img_small.shape[1], img_small.shape[0])  # Toàn bộ ảnh
    
    all_points = np.vstack(contours)
    x0, y0, w0, h0 = cv2.boundingRect(all_points)

    max_area = 0
    best_rect = None
    step = max(4, min(w0, h0) // 20)
    min_size = min(40, min(w0, h0) // 4)

    for top in range(y0, y0 + h0 - min_size, step):
        for left in range(x0, x0 + w0 - min_size, step):
            for bottom in range(top + min_size, y0 + h0, step):
                for right in range(left + min_size, x0 + w0, step):
                    if bottom <= img_small.shape[0] and right <= img_small.shape[1]:
                        region = mask[top:bottom, left:right]
                        if region.size > 0 and np.all(region == 1):
                            area = (bottom - top) * (right - left)
                            if area > max_area:
                                max_area = area
                                best_rect = (left, top, right - left, bottom - top)
    
    if best_rect is None:
        best_rect = (x0, y0, w0, h0)
    return best_rect


def find_largest_content_rectangle(img, threshold=0.01, scale_factor=0.25, padding=10):
    """
    Tìm hình chữ nhật lớn nhất chỉ chứa 100% nội dung trên ảnh gốc,
    bằng cách xử lý ảnh đã giảm kích thước và scale ngược lại.
    """
    orig_h, orig_w = img.shape[:2]

    # Giảm độ phân giải ảnh để xử lý nhanh
    small_img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    
    x_s, y_s, w_s, h_s = _find_content_rect_on_downscaled(small_img, threshold)

    # Scale ngược lại về ảnh gốc
    x = int(x_s / scale_factor)
    y = int(y_s / scale_factor)
    w = int(w_s / scale_factor)
    h = int(h_s / scale_factor)

    # Thêm padding nhưng không vượt giới hạn ảnh
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(orig_w - x, w + 2 * padding)
    h = min(orig_h - y, h + 2 * padding)

    cropped = img[y:y+h, x:x+w]

    space_saved = ((orig_h * orig_w - cropped.shape[0] * cropped.shape[1]) /
                   (orig_h * orig_w) * 100)

    print(f"✂️  Cropped from {img.shape[:2]} to {cropped.shape[:2]} "
          f"(saved {space_saved:.1f}% space using scaled crop)")

    return cropped

def auto_crop_panorama(panorama, method='content_aware'):
    if method == 'content_aware':
        return find_largest_content_rectangle(panorama)
    elif method == 'simple_crop':
        h, w = panorama.shape[:2]
        crop_ratio = 0.5
        new_w = int(w * crop_ratio)
        new_h = int(h * crop_ratio)
        x_start = (w - new_w) // 2
        y_start = (h - new_h) // 2
        cropped = panorama[y_start:y_start + new_h, x_start:x_start + new_w]
        print(f"🔪 Simple crop applied: from ({w}, {h}) to ({new_w}, {new_h})")
        return cropped
    else:
        return panorama

def match_and_stitch_advanced(left_path, right_path, blend_method='feather', homography_method='ransac', auto_crop=True):
    print("🔄 Loading images...")
    start_time = time.time()
    
    left_gray, left_bgr, left_rgb = read_image_optimized(left_path)
    right_gray, right_bgr, right_rgb = read_image_optimized(right_path)
    
    print(f"📸 Images loaded: {left_rgb.shape}, {right_rgb.shape}")
    
    print("🔍 Extracting SIFT features...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_left = executor.submit(SIFT_optimized, left_gray)
        future_right = executor.submit(SIFT_optimized, right_gray)
        kp_left, des_left = future_left.result()
        kp_right, des_right = future_right.result()
    
    print(f"✅ Features: Left={len(kp_left)}, Right={len(kp_right)}")
    
    print("🔗 Matching features...")
    good_matches = KNN_matcher_optimized(des_left, des_right)
    
    if len(good_matches) < 10:
        raise ValueError("Not enough matches found!")
    
    print(f"✅ Found {len(good_matches)} good matches")
    
    print(f"🎯 Computing homography using {homography_method}...")
    matches_for_homography = [(kp_left[m.queryIdx], kp_right[m.trainIdx]) for m in good_matches]
    H = compute_homography_robust(matches_for_homography, method=homography_method)
    
    if H is None:
        raise ValueError("Failed to compute homography!")
    
    print("🖼️ Warping and stitching...")
    
    left_norm = left_rgb.astype(np.float32) / 255.0
    right_norm = right_rgb.astype(np.float32) / 255.0
    
    h_left, w_left = left_norm.shape[:2]
    h_right, w_right = right_norm.shape[:2]
    
    corners = np.array([[0, 0], [w_left, 0], [w_left, h_left], [0, h_left]], 
                      dtype=np.float32).reshape(-1, 1, 2)
    corners_transformed = cv2.perspectiveTransform(corners, H)
    
    all_corners = np.concatenate([
        corners_transformed.reshape(-1, 2),
        np.array([[0, 0], [w_right, 0], [w_right, h_right], [0, h_right]])
    ])
    
    x_min, y_min = np.floor(all_corners.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0)).astype(int)
    
    max_dim = 8000
    out_w = x_max - x_min
    out_h = y_max - y_min
    if out_w > max_dim or out_h > max_dim:
        raise MemoryError(f"Output panorama too large: {out_w}x{out_h} px.")
    
    translation = np.array([[1, 0, -x_min],
                           [0, 1, -y_min],
                           [0, 0, 1]], dtype=np.float32)
    
    output_size = (out_w, out_h)
    
    H_translated = translation @ H
    warped_left = cv2.warpPerspective(left_norm, H_translated, output_size)
    warped_right = cv2.warpPerspective(right_norm, translation, output_size)
    
    print(f"🎨 Blending images using {blend_method}...")
    
    if blend_method == 'simple':
        result = np.zeros_like(warped_left)
        mask_left = np.sum(warped_left, axis=2) > 0.01
        mask_right = np.sum(warped_right, axis=2) > 0.01
        mask_overlap = mask_left & mask_right
        result[mask_left & ~mask_overlap] = warped_left[mask_left & ~mask_overlap]
        result[mask_right & ~mask_overlap] = warped_right[mask_right & ~mask_overlap]
        result[mask_overlap] = (warped_left[mask_overlap] + warped_right[mask_overlap]) * 0.5
        
    elif blend_method == 'feather':
        result = feather_blend_numba(warped_left, warped_right)
        
    elif blend_method == 'multiband':
        mask_left = np.sum(warped_left, axis=2) > 0.01
        mask_right = np.sum(warped_right, axis=2) > 0.01
        overlap_cols = []
        for j in range(output_size[0]):
            overlap_count = np.sum(mask_left[:, j] & mask_right[:, j])
            if overlap_count > output_size[1] * 0.05:
                overlap_cols.append(j)
        seam_x = (overlap_cols[0] + overlap_cols[-1]) // 2 if overlap_cols else output_size[0] // 2
        result = multiband_blend(warped_left, warped_right, seam_x)
    
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    
    if auto_crop:
        print("✂️ Auto-cropping...")
        result = auto_crop_panorama(result, method='content_aware')
    
    total_time = time.time() - start_time
    print(f"✨ Panorama completed in {total_time:.2f} seconds!")
    
    return result

def stitch_multiple_images_advanced(image_paths, blend_method='feather', homography_method='ransac', auto_crop=True):
    if len(image_paths) < 2:
        raise ValueError("Cần ít nhất 2 ảnh để ghép panorama.")
    panorama = read_image_optimized(image_paths[0])[2]
    tmp_path = "__tmp_panorama.jpg"
    try:
        for idx in range(1, len(image_paths)):
            right_path = image_paths[idx]
            cv2.imwrite(tmp_path, cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR))
            panorama = match_and_stitch_advanced(
                tmp_path, right_path,
                blend_method=blend_method,
                homography_method=homography_method,
                auto_crop=False
            )
            if panorama is None:
                raise RuntimeError(f"Lỗi khi ghép ảnh thứ {idx+1}")
        if auto_crop:
            panorama = auto_crop_panorama(panorama, method='content_aware')
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as e:
                print(f"⚠️ Warning: Could not remove temp file {tmp_path}: {e}")
    return panorama

def create_panorama_advanced(image_paths, output_path=None, 
                           blend_method='feather', homography_method='ransac', auto_crop=True):
    try:
        if isinstance(image_paths, (list, tuple)):
            if len(image_paths) == 2:
                panorama = match_and_stitch_advanced(
                    image_paths[0], image_paths[1],
                    blend_method, homography_method, auto_crop
                )
            else:
                panorama = stitch_multiple_images_advanced(
                    image_paths, blend_method, homography_method, auto_crop
                )
        else:
            panorama = match_and_stitch_advanced(
                image_paths[0], image_paths[1],
                blend_method, homography_method, auto_crop
            )
        if output_path and panorama is not None:
            panorama_bgr = cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, panorama_bgr)
            print(f"💾 Saved panorama to: {output_path}")
        return panorama
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

####################
# GUI Application #
####################

class PanoramaGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Panorama Stitcher")
        
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        self.window_width = min(int(screen_width * 0.8), 1200)
        self.window_height = min(int(screen_height * 0.8), 800)
        
        x = (screen_width - self.window_width) // 2
        y = (screen_height - self.window_height) // 2
        
        self.root.geometry(f"{self.window_width}x{self.window_height}+{x}+{y}")
        self.root.minsize(800, 600)
        
        self.camera_manager = CameraManager()
        
        self.canvas_width = min(int(self.window_width * 0.4), 500)
        self.canvas_height = min(int(self.window_height * 0.35), 350)
        self.camera_width = min(int(self.window_width * 0.3), 400)
        self.camera_height = min(int(self.window_height * 0.25), 300)
        
        self.result_image = None
        self.save_result = self.save_result_file  # <--- Add this line to alias the correct save method for file tab
        self.camera_active = False
        self.camera_thread = None
        self.photo_references = []
        self.last_frame = None
        self.image_paths = []
        self.preview_imgtk = None

        # Tách widget cho từng tab
        self.file_canvas = None
        self.file_progress = None
        self.file_status_label = None
        self.camera_canvas = None
        self.camera_progress = None
        self.camera_status_label = None

        try:
            self.default_font = ('Arial', 10)
            self.title_font = ('Arial', 14, 'bold')
            self.small_font = ('Arial', 8)
        except:
            self.default_font = ('sans-serif', 10)
            self.title_font = ('sans-serif', 14, 'bold')
            self.small_font = ('sans-serif', 8)
        
        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        self.file_frame = ttk.Frame(self.notebook, padding="5")
        self.camera_frame = ttk.Frame(self.notebook, padding="5")
        
        self.notebook.add(self.file_frame, text="File Mode")
        self.notebook.add(self.camera_frame, text="Live Camera")
        
        self.setup_file_mode()
        self.setup_camera_mode()
    
    def setup_file_mode(self):
        self.file_frame.columnconfigure(0, weight=1)
        self.file_frame.columnconfigure(1, weight=1)
        self.file_frame.rowconfigure(2, weight=1)
        
        file_selection_frame = ttk.LabelFrame(self.file_frame, text="Image Selection", padding="10")
        file_selection_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        file_selection_frame.columnconfigure(0, weight=1)
        file_selection_frame.rowconfigure(1, weight=1)
        
        ttk.Button(file_selection_frame, text="Select Images", 
                  command=self.select_multiple_images, width=20).grid(row=0, column=0, sticky=tk.W)
        self.multi_label = ttk.Label(file_selection_frame, text="No images selected", foreground="gray")
        self.multi_label.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        listbox_frame = ttk.Frame(file_selection_frame)
        listbox_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        listbox_frame.columnconfigure(0, weight=1)
        self.listbox = tk.Listbox(listbox_frame, height=8, width=32, selectmode=tk.SINGLE, exportselection=False)
        self.listbox.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.listbox.bind('<<ListboxSelect>>', self.show_selected_image_preview)
        up_btn = ttk.Button(listbox_frame, text="Up", width=6, command=self.move_up)
        down_btn = ttk.Button(listbox_frame, text="Down", width=6, command=self.move_down)
        up_btn.grid(row=0, column=1, padx=3, pady=1, sticky=tk.N)
        down_btn.grid(row=1, column=1, padx=3, pady=1, sticky=tk.S)
        
        options_frame = ttk.LabelFrame(self.file_frame, text="Processing Options", padding="10")
        options_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        options_frame.columnconfigure(1, weight=1)
        options_frame.columnconfigure(3, weight=1)
        
        ttk.Label(options_frame, text="Blending Method:", font=self.default_font).grid(row=0, column=0, sticky=tk.W, padx=5)
        self.blend_var = tk.StringVar(value="feather")
        blend_combo = ttk.Combobox(options_frame, textvariable=self.blend_var,
                                  values=["simple", "feather", "multiband"], width=15, state='readonly')
        blend_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(options_frame, text="Homography Method:", font=self.default_font).grid(row=0, column=2, sticky=tk.W, padx=5)
        self.homography_var = tk.StringVar(value="ransac")
        homography_combo = ttk.Combobox(options_frame, textvariable=self.homography_var,
                                       values=["linear", "ransac"], width=15, state='readonly')
        homography_combo.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        
        self.auto_crop_var = tk.BooleanVar(value=True)
        auto_crop_check = ttk.Checkbutton(options_frame, text="Auto-crop result", 
                                         variable=self.auto_crop_var)
        auto_crop_check.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)
        
        self.manual_crop_btn = ttk.Button(options_frame, text="Manual Crop", 
                                         command=self.manual_crop, state='disabled', width=12)
        self.manual_crop_btn.grid(row=1, column=2, columnspan=2, padx=5, pady=5, sticky=tk.W)
        
        control_frame = ttk.Frame(options_frame)
        control_frame.grid(row=2, column=0, columnspan=4, pady=10)
        self.process_btn = ttk.Button(control_frame, text="Create Panorama", 
                                     command=self.process_images, width=20)
        self.process_btn.grid(row=0, column=0, padx=10)
        self.save_btn = ttk.Button(control_frame, text="Save Result", 
                                  command=self.save_result, state='disabled', width=15)
        self.save_btn.grid(row=0, column=1, padx=10)
        
        preview_frame = ttk.LabelFrame(self.file_frame, text="Image Preview", padding="10")
        preview_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10), pady=(10, 0))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        self.preview_label = ttk.Label(preview_frame)
        self.preview_label.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W), pady=5)
        
        self.setup_result_display(self.file_frame, row=2, column=1, prefix='file')
    
    def setup_camera_mode(self):
        self.camera_frame.columnconfigure(0, weight=1)
        self.camera_frame.columnconfigure(1, weight=1)
        self.camera_frame.rowconfigure(1, weight=1)
        
        control_frame = ttk.LabelFrame(self.camera_frame, text="Camera Controls", padding="8")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=8)
        control_frame.columnconfigure(0, weight=1)
        control_frame.columnconfigure(1, weight=1)
        control_frame.columnconfigure(2, weight=1)
        control_frame.columnconfigure(3, weight=1)
        control_frame.columnconfigure(4, weight=1)
        
        ttk.Label(control_frame, text="Camera:", font=self.default_font).grid(row=0, column=0, padx=3, pady=3, sticky=tk.W)
        self.camera_var = tk.StringVar()
        self.camera_combo = ttk.Combobox(control_frame, textvariable=self.camera_var, 
                                        width=8, state='readonly')
        self.camera_combo.grid(row=0, column=1, padx=3, pady=3)
        
        ttk.Button(control_frame, text="Find", 
                  command=self.find_cameras, width=10).grid(row=0, column=2, padx=3, pady=3)
        
        self.start_camera_btn = ttk.Button(control_frame, text="Start", 
                                          command=self.start_camera, width=10, state='disabled')
        self.start_camera_btn.grid(row=0, column=3, padx=3, pady=3)
        
        self.stop_camera_btn = ttk.Button(control_frame, text="Stop", 
                                         command=self.stop_camera, width=10, state='disabled')
        self.stop_camera_btn.grid(row=0, column=4, padx=3, pady=3)
        
        capture_frame = ttk.Frame(control_frame)
        capture_frame.grid(row=1, column=0, columnspan=5, pady=8)
        
        self.capture_btn = ttk.Button(capture_frame, text="Capture", 
                                     command=self.capture_image, width=12, state='disabled')
        self.capture_btn.grid(row=0, column=0, padx=3)
        
        self.clear_btn = ttk.Button(capture_frame, text="Clear", 
                                   command=self.clear_captures, width=10, state='disabled')
        self.clear_btn.grid(row=0, column=1, padx=3)
        
        self.create_pano_btn = ttk.Button(capture_frame, text="Create Panorama", 
                                         command=self.create_panorama_from_captures, 
                                         width=15, state='disabled')
        self.create_pano_btn.grid(row=0, column=2, padx=3)
        
        self.save_camera_btn = ttk.Button(capture_frame, text="Save Result", 
                                         command=self.save_result, width=12, state='disabled')
        self.save_camera_btn.grid(row=0, column=3, padx=3)
        
        self.camera_status = ttk.Label(capture_frame, text="Camera: Stopped | Captures: 0", 
                                      font=self.small_font)
        self.camera_status.grid(row=0, column=4, padx=15)
        
        ttk.Label(control_frame, text="Blending:", font=self.default_font).grid(row=2, column=0, padx=3, pady=3, sticky=tk.W)
        self.blend_method_var = tk.StringVar(value="feather")
        self.blend_combo = ttk.Combobox(control_frame, textvariable=self.blend_method_var, 
                                       width=12, state='readonly')
        self.blend_combo['values'] = ("feather", "simple", "multiband")
        self.blend_combo.current(0)
        self.blend_combo.grid(row=2, column=1, padx=3, pady=3, sticky=tk.W)
        
        camera_panel = ttk.LabelFrame(self.camera_frame, text="Camera Preview", padding="5")
        camera_panel.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 3))
        camera_panel.columnconfigure(0, weight=1)
        camera_panel.rowconfigure(0, weight=1)
        
        try:
            self.camera_canvas = tk.Canvas(camera_panel, bg='black', relief=tk.SUNKEN, borderwidth=1)
            self.camera_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            self.camera_canvas.create_text(10, 10, anchor="nw",
                                          text="Camera preview",
                                          fill="white", font=self.default_font)
            self.camera_canvas.bind("<Configure>", self.on_camera_canvas_resize)
        except Exception as e:
            print(f"Error creating camera canvas: {e}")
            self.camera_canvas = tk.Canvas(camera_panel, width=320, height=240, bg='black')
            self.camera_canvas.grid(row=0, column=0)
        
        right_panel = ttk.Frame(self.camera_frame)
        right_panel.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(3, 0))
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(1, weight=1)
        
        thumb_frame = ttk.LabelFrame(right_panel, text="Captured Images", padding="5")
        thumb_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        thumb_frame.columnconfigure(0, weight=1)
        
        try:
            self.thumb_canvas = tk.Canvas(thumb_frame, height=100, bg='white')
            thumb_scrollbar = ttk.Scrollbar(thumb_frame, orient="horizontal", command=self.thumb_canvas.xview)
            self.thumb_scrollable_frame = ttk.Frame(self.thumb_canvas)
            
            self.thumb_scrollable_frame.bind(
                "<Configure>",
                lambda e: self.thumb_canvas.configure(scrollregion=self.thumb_canvas.bbox("all"))
            )
            
            self.thumb_canvas.create_window((0, 0), window=self.thumb_scrollable_frame, anchor="nw")
            self.thumb_canvas.configure(xscrollcommand=thumb_scrollbar.set)
            
            self.thumb_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E))
            thumb_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        except Exception as e:
            print(f"Error creating thumbnails: {e}")
            self.thumb_scrollable_frame = ttk.Frame(thumb_frame)
            self.thumb_scrollable_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        self.setup_result_display(right_panel, row=1, prefix='camera')
    
    def setup_result_display(self, parent, row, column=0, prefix='file'):
        result_frame = ttk.LabelFrame(parent, text="Result Preview", padding="5")
        result_frame.grid(row=row, column=column, sticky=(tk.W, tk.E, tk.N, tk.S), pady=8)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        parent.rowconfigure(row, weight=1)
        
        canvas_frame = ttk.Frame(result_frame)
        canvas_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        
        try:
            canvas = tk.Canvas(canvas_frame, width=self.canvas_width, height=self.canvas_height,
                              bg='white', relief=tk.SUNKEN, borderwidth=1)
            v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
            h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=canvas.xview)
            canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
            canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
            h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
            canvas.create_text(self.canvas_width//2, self.canvas_height//2,
                               text="Result will appear here",
                               fill="gray", font=self.default_font)
        except Exception as e:
            print(f"Error creating result canvas: {e}")
            canvas = tk.Canvas(canvas_frame, width=400, height=300, bg='white')
            canvas.grid(row=0, column=0)

        progress_frame = ttk.Frame(result_frame)
        progress_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=8)
        progress_frame.columnconfigure(0, weight=1)

        progress = ttk.Progressbar(progress_frame, mode='indeterminate')
        progress.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=3)

        status_label = ttk.Label(progress_frame, text="Ready", font=self.default_font)
        status_label.grid(row=1, column=0, pady=3)

        # Gán vào thuộc tính đúng tab
        if prefix == 'file':
            self.file_canvas = canvas
            self.file_progress = progress
            self.file_status_label = status_label
        elif prefix == 'camera':
            self.camera_result_canvas = canvas
            self.camera_progress = progress
            self.camera_status_label = status_label

    def select_multiple_images(self):
        paths = filedialog.askopenfilenames(
            title="Select Images for Panorama",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if paths:
            self.image_paths = list(paths)
            self.multi_label.config(text=f"{len(self.image_paths)} images selected")
            self.update_listbox()
            self.clear_preview()
    
    def update_listbox(self):
        self.listbox.delete(0, tk.END)
        for path in self.image_paths:
            self.listbox.insert(tk.END, path.split('/')[-1])
    
    def move_up(self):
        idx = self.listbox.curselection()
        if not idx or idx[0] == 0:
            return
        i = idx[0]
        self.image_paths[i-1], self.image_paths[i] = self.image_paths[i], self.image_paths[i-1]
        self.update_listbox()
        self.listbox.selection_set(i-1)
    
    def move_down(self):
        idx = self.listbox.curselection()
        if not idx or idx[0] == len(self.image_paths)-1:
            return
        i = idx[0]
        self.image_paths[i+1], self.image_paths[i] = self.image_paths[i], self.image_paths[i+1]
        self.update_listbox()
        self.listbox.selection_set(i+1)
        self.show_selected_image_preview()
    
    def show_selected_image_preview(self, event=None):
        idx = self.listbox.curselection()
        if not idx:
            self.clear_preview()
            return
        path = self.image_paths[idx[0]]
        try:
            img = cv2.imread(path)
            if img is None:
                self.clear_preview()
                return
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            preview_frame = self.preview_label.master
            preview_frame.update_idletasks()
            frame_w = preview_frame.winfo_width() or 400
            frame_h = preview_frame.winfo_height() or 300
            maxsize_w = frame_w - 10
            maxsize_h = frame_h - 10
            scale = min(maxsize_w / img.shape[1], maxsize_h / img.shape[0], 1.0)
            new_w, new_h = int(img.shape[1] * scale), int(img.shape[0] * scale)
            img_large = cv2.resize(img, (new_w, new_h))
            im_pil = Image.fromarray(img_large)
            self.preview_imgtk = ImageTk.PhotoImage(im_pil)
            self.preview_label.config(image=self.preview_imgtk)
            self.photo_references.append(self.preview_imgtk)
        except Exception:
            self.clear_preview()
    
    def clear_preview(self):
        self.preview_label.config(image='')
        self.preview_imgtk = None
    
    def process_images(self):
        if self.image_paths and len(self.image_paths) >= 2:
            paths = self.image_paths.copy()
        else:
            messagebox.showerror("Error", "Please select at least 2 images!")
            return
        self._process_paths = paths
        self.process_btn.config(state='disabled')
        self.manual_crop_btn.config(state='disabled')
        self.save_btn.config(state='disabled')
        self.file_progress.start()
        self.file_status_label.config(text="Processing...")
        thread = threading.Thread(target=self.create_panorama_thread)
        thread.start()
    
    def create_panorama_thread(self):
        try:
            self.result_image = create_panorama_advanced(
                self._process_paths,
                blend_method=self.blend_var.get(),
                homography_method=self.homography_var.get(),
                auto_crop=self.auto_crop_var.get()
            )
            if self.result_image is not None:
                self._original_size = self.result_image.shape[:2]
                self.root.after(0, self.display_result_file)
            else:
                self.root.after(0, lambda: self.update_status_file("Failed to create panorama"))
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.root.after(0, lambda msg=error_msg: self.update_status_file(msg))
    
    def display_result_file(self):
        self.file_progress.stop()
        self.process_btn.config(state='normal')
        self.save_btn.config(state='normal')
        self.manual_crop_btn.config(state='normal')

        self.file_canvas.delete("all")
        self.file_canvas.configure(scrollregion=(0, 0, self.canvas_width, self.canvas_height))

        if self.result_image is not None:
            try:
                h, w = self.result_image.shape[:2]
                scale = min(self.canvas_width / w, self.canvas_height / h, 1.0)
                new_w, new_h = int(w * scale), int(h * scale)
                if new_w < 50 or new_h < 50:
                    scale = min(400 / w, 300 / h, 1.0)
                    new_w, new_h = int(w * scale), int(h * scale)
                if new_w < 1 or new_h < 1:
                    self.file_canvas.create_text(self.canvas_width // 2, self.canvas_height // 2,
                                                text="Result too small to display",
                                                fill="red", font=self.default_font)
                    self.file_status_label.config(text="Result too small to display")
                    return
                resized = cv2.resize(self.result_image, (new_w, new_h))
                self.file_canvas.configure(scrollregion=(0, 0, new_w, new_h))
                pil_image = Image.fromarray(resized)
                self.photo = ImageTk.PhotoImage(pil_image)
                self.photo_references.append(self.photo)
                x = new_w // 2
                y = new_h // 2
                self.file_canvas.create_image(x, y, image=self.photo, anchor="center")
                space_info = ""
                if hasattr(self, '_original_size'):
                    orig_pixels = self._original_size[0] * self._original_size[1]
                    curr_pixels = h * w
                    space_saved = (orig_pixels - curr_pixels) / orig_pixels * 100
                    space_info = f", Saved: {space_saved:.1f}%"
                self.file_status_label.config(text=f"Panorama: {w}x{h} px{space_info}")
                self.file_canvas.update_idletasks()
                self.root.update_idletasks()
            except Exception as e:
                print(f"Error displaying result: {e}")
                self.file_canvas.delete("all")
                self.file_canvas.create_text(self.canvas_width // 2, self.canvas_height // 2,
                                            text="Error displaying result",
                                            fill="red", font=self.default_font)
                self.file_status_label.config(text="Error displaying result")
        else:
            self.file_canvas.delete("all")
            self.file_canvas.create_text(self.canvas_width // 2, self.canvas_height // 2,
                                        text="No result to display",
                                        fill="gray", font=self.default_font)
            self.file_status_label.config(text="No image to display")
    
    def update_status_file(self, message):
        self.file_progress.stop()
        self.process_btn.config(state='normal')
        self.file_status_label.config(text=message)
    
    def find_cameras(self):
        self.camera_status.config(text="Searching for cameras...")
        self.root.update()
        
        cameras = self.camera_manager.find_available_cameras()
        if cameras:
            self.camera_combo['values'] = [f"Camera {i}" for i in cameras]
            self.camera_combo.current(0)
            self.start_camera_btn.config(state='normal')
            self.camera_status.config(text=f"Found {len(cameras)} camera(s)")
        else:
            self.camera_combo['values'] = []
            self.camera_status.config(text="No cameras found")
            messagebox.showwarning("Warning", "No cameras found!")
    
    def start_camera(self):
        if not self.camera_combo.get():
            messagebox.showwarning("Warning", "Please select a camera first!")
            return
        
        camera_index = int(self.camera_combo.get().split()[-1])
        
        if self.camera_manager.open_camera(camera_index):
            self.camera_active = True
            self.start_camera_btn.config(state='disabled')
            self.stop_camera_btn.config(state='normal')
            self.capture_btn.config(state='normal')
            self.clear_btn.config(state='normal')
            
            self.camera_thread = threading.Thread(target=self.camera_loop)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            
            self.update_camera_status()
        else:
            messagebox.showerror("Error", f"Failed to open camera {camera_index}")
    
    def stop_camera(self):
        self.camera_active = False
        self.camera_manager.close_camera()
        
        self.start_camera_btn.config(state='normal')
        self.stop_camera_btn.config(state='disabled')
        self.capture_btn.config(state='disabled')
        
        self.camera_canvas.delete("all")
        self.camera_canvas.create_text(self.camera_width//2, self.camera_height//2, 
                                      text="Camera stopped", 
                                      fill="gray", font=self.default_font)
        
        self.update_camera_status()
    
    def camera_loop(self):
        failed_reads = 0
        while self.camera_active:
            frame = self.camera_manager.read_frame()
            if frame is not None:
                failed_reads = 0
                self.last_frame = frame.copy()
                self.root.after(0, lambda f=frame: self.update_camera_display(f))
            else:
                failed_reads += 1
                if failed_reads > 10:
                    self.root.after(0, lambda: self.handle_camera_disconnect())
                    break
            time.sleep(0.033)
    
    def handle_camera_disconnect(self):
        self.camera_active = False
        self.camera_manager.close_camera()
        
        self.start_camera_btn.config(state='normal')
        self.stop_camera_btn.config(state='disabled')
        self.capture_btn.config(state='disabled')
        
        self.camera_canvas.delete("all")
        self.camera_canvas.create_text(self.camera_width//2, self.camera_height//2, 
                                     text="Camera disconnected", 
                                     fill="red", font=self.default_font)
        
        self.camera_status.config(text="Camera: Disconnected | Captures: " + 
                                 str(self.camera_manager.get_captured_count()))
        messagebox.showerror("Error", "Camera connection lost")
    
    def on_camera_canvas_resize(self, event):
        self.camera_width = event.width
        self.camera_height = event.height
        if self.last_frame is not None:
            self.update_camera_display(self.last_frame)
    
    def update_camera_display(self, frame):
        if not self.camera_active:
            return
        
        try:
            h, w = frame.shape[:2]
            scale = min(self.camera_width / w, self.camera_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            if new_w < 10 or new_h < 10:
                return
            
            resized_frame = cv2.resize(frame, (new_w, new_h))
            pil_image = Image.fromarray(resized_frame)
            self.camera_photo = ImageTk.PhotoImage(pil_image)
            self.photo_references.append(self.camera_photo)
            self.camera_canvas.delete("all")
            x = self.camera_width // 2
            y = self.camera_height // 2
            self.camera_canvas.create_image(x, y, image=self.camera_photo, anchor="center")
        except Exception as e:
            print(f"Error updating camera display: {e}")
    
    def capture_image(self):
        if not self.camera_active:
            messagebox.showwarning("Warning", "Camera is not running!")
            return

        captured_frame = None
        if self.last_frame is not None:
            captured_frame = self.last_frame.copy()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"capture_{timestamp}.jpg"
            filepath = os.path.join(self.camera_manager.temp_dir, filename)
            frame_bgr = cv2.cvtColor(captured_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, frame_bgr)
            self.camera_manager.captured_images.append({
                'path': filepath,
                'image': captured_frame.copy(),
                'timestamp': timestamp
            })
            print(f"📸 Captured image {len(self.camera_manager.captured_images)}: {filename}")

        if captured_frame is not None:
            self.root.after(0, lambda: self.add_thumbnail(captured_frame))
            self.update_camera_status()
            if self.camera_manager.get_captured_count() >= 2:
                self.create_pano_btn.config(state='normal')
            self.root.update_idletasks()
    
    def add_thumbnail(self, image):
        try:
            thumb_size = 80
            h, w = image.shape[:2]
            scale = min(thumb_size/w, thumb_size/h)
            new_w, new_h = int(w*scale), int(h*scale)
            
            if new_w < 10 or new_h < 10:
                print("Thumbnail too small, skipping")
                return
            
            thumb_image = cv2.resize(image, (new_w, new_h))
            pil_thumb = Image.fromarray(thumb_image)
            photo_thumb = ImageTk.PhotoImage(pil_thumb)
            
            thumb_frame = ttk.Frame(self.thumb_scrollable_frame)
            thumb_frame.pack(side=tk.LEFT, padx=2)
            
            thumb_label = ttk.Label(thumb_frame, image=photo_thumb)
            thumb_label.image = photo_thumb
            thumb_label.pack()
            
            count = self.camera_manager.get_captured_count()
            num_label = ttk.Label(thumb_frame, text=f"#{count}", font=self.small_font)
            num_label.pack()
            
            self.photo_references.append(photo_thumb)
            
            self.thumb_scrollable_frame.update_idletasks()
            self.thumb_canvas.configure(scrollregion=self.thumb_canvas.bbox("all"))
            self.root.update_idletasks()
        except Exception as e:
            print(f"Error adding thumbnail: {e}")
    
    def clear_captures(self):
        self.camera_manager.clear_captures()
        
        for widget in self.thumb_scrollable_frame.winfo_children():
            widget.destroy()
        
        self.create_pano_btn.config(state='disabled')
        self.save_camera_btn.config(state='disabled')
        
        self.result_image = None
        self.canvas.delete("all")
        self.canvas.create_text(self.canvas_width//2, self.canvas_height//2, 
                               text="Result will appear here", 
                               fill="gray", font=self.default_font)
        
        self.update_camera_status()
    
    def update_camera_status(self):
        status = "Running" if self.camera_active else "Stopped"
        count = self.camera_manager.get_captured_count()
        self.camera_status.config(text=f"Camera: {status} | Captures: {count}")
    
    def create_panorama_from_captures(self):
        captured_paths = self.camera_manager.get_captured_paths()
        
        if len(captured_paths) < 2:
            messagebox.showwarning("Warning", "Need at least 2 images!")
            return
        
        self.camera_progress.start()
        self.camera_status_label.config(text="Processing...")
        self.create_pano_btn.config(state='disabled')
        self.save_camera_btn.config(state='disabled')
        blend_method = self.blend_method_var.get()
        thread = threading.Thread(target=self.stitch_multiple_images, args=(captured_paths, blend_method))
        thread.start()
    
    def stitch_multiple_images(self, image_paths, blend_method):
        try:
            self.camera_status_label.config(text="Processing multiple images...")
            self.root.update()
            current_pano = create_panorama_advanced(
                image_paths,
                blend_method=blend_method,
                homography_method='ransac',
                auto_crop=False  # <--- Không tự động crop khi ghép ảnh từ camera
            )
            if current_pano is not None:
                self.result_image = current_pano
                self.root.after(0, self.display_result_camera)
                self.save_camera_btn.config(state='normal')
                self.camera_status_label.config(text="Multi-image panorama completed!")
            else:
                self.root.after(0, lambda: self.update_status_camera("Failed to create panorama"))
        except Exception as e:
            self.root.after(0, lambda e=e: self.update_status_camera(f"Error: {str(e)}"))
        finally:
            self.camera_progress.stop()
            self.create_pano_btn.config(state='normal')

    def display_result_camera(self):
        self.camera_progress.stop()
        self.create_pano_btn.config(state='normal')
        self.save_camera_btn.config(state='normal')

        self.camera_result_canvas.delete("all")
        self.camera_result_canvas.configure(scrollregion=(0, 0, self.canvas_width, self.canvas_height))

        if self.result_image is not None:
            try:
                h, w = self.result_image.shape[:2]
                scale = min(self.canvas_width / w, self.canvas_height / h, 1.0)
                new_w, new_h = int(w * scale), int(h * scale)
                if new_w < 50 or new_h < 50:
                    scale = min(400 / w, 300 / h, 1.0)
                    new_w, new_h = int(w * scale), int(h * scale)
                if new_w < 1 or new_h < 1:
                    self.camera_result_canvas.create_text(self.canvas_width // 2, self.canvas_height // 2,
                                                         text="Result too small to display",
                                                         fill="red", font=self.default_font)
                    self.camera_status_label.config(text="Result too small to display")
                    return
                resized = cv2.resize(self.result_image, (new_w, new_h))
                self.camera_result_canvas.configure(scrollregion=(0, 0, new_w, new_h))
                pil_image = Image.fromarray(resized)
                self.photo = ImageTk.PhotoImage(pil_image)
                self.photo_references.append(self.photo)
                x = new_w // 2
                y = new_h // 2
                self.camera_result_canvas.create_image(x, y, image=self.photo, anchor="center")
                self.camera_status_label.config(text=f"Panorama: {w}x{h} px")
                self.camera_result_canvas.update_idletasks()
                self.root.update_idletasks()
            except Exception as e:
                print(f"Error displaying result: {e}")
                self.camera_result_canvas.delete("all")
                self.camera_result_canvas.create_text(self.canvas_width // 2, self.canvas_height // 2,
                                                     text="Error displaying result",
                                                     fill="red", font=self.default_font)
                self.camera_status_label.config(text="Error displaying result")
        else:
            self.camera_result_canvas.delete("all")
            self.camera_result_canvas.create_text(self.canvas_width // 2, self.canvas_height // 2,
                                                 text="No result to display",
                                                 fill="gray", font=self.default_font)
            self.camera_status_label.config(text="No image to display")

    def update_status_camera(self, message):
        self.camera_progress.stop()
        self.create_pano_btn.config(state='normal')
        self.camera_status_label.config(text=message)

    def manual_crop(self):
        if self.result_image is None:
            messagebox.showwarning("Warning", "No panorama to crop!")
            return
        
        original_image = self.result_image.copy()
        
        crop_dialog = tk.Toplevel(self.root)
        crop_dialog.title("Crop Options")
        crop_dialog.geometry("360x280")
        crop_dialog.transient(self.root)
        crop_dialog.grab_set()
        
        crop_dialog.geometry("+{}+{}".format(
            self.root.winfo_rootx() + 50,
            self.root.winfo_rooty() + 50
        ))
        
        ttk.Label(crop_dialog, text="Select crop method:", font=self.title_font).pack(pady=15)
        
        desc_frame = ttk.Frame(crop_dialog)
        desc_frame.pack(pady=5, padx=10, fill='x')
        
        ttk.Label(desc_frame, text="• Content-Aware: Find largest area with 100% content", 
                 font=self.small_font, foreground='blue').pack(anchor='w')
        ttk.Label(desc_frame, text="• Simple Crop: Remove black borders only", 
                 font=self.small_font, foreground='green').pack(anchor='w')
        ttk.Label(desc_frame, text="• No Crop: Restore original panorama", 
                 font=self.small_font, foreground='orange').pack(anchor='w')
        
        def apply_crop(method):
            try:
                self.file_status_label.config(text=f"Applying {method} crop...")
                self.root.update()
                
                if method == 'content_aware':
                    cropped = auto_crop_panorama(original_image, method='content_aware')
                elif method == 'simple':
                    cropped = auto_crop_panorama(original_image, method='simple_crop')
                elif method == 'none':
                    self.file_status_label.config(text="Recreating panorama without crop...")
                    self.root.update()
                    cropped = create_panorama_advanced(
                        self._process_paths,
                        blend_method=self.blend_var.get(),
                        homography_method=self.homography_var.get(),
                        auto_crop=False
                    )
                else:
                    cropped = original_image
                
                if cropped is not None:
                    self.result_image = cropped
                    self.display_result_file()
                    self.file_status_label.config(text=f"{method.title()} crop applied successfully!")
                else:
                    self.file_status_label.config(text=f"Failed to apply {method} crop")
                    
            except Exception as e:
                self.file_status_label.config(text=f"Error applying crop: {str(e)}")
                print(f"Crop error: {e}")
            
            crop_dialog.destroy()
        
        button_frame = ttk.Frame(crop_dialog)
        button_frame.pack(pady=15)
        
        ttk.Button(button_frame, text="Content-Aware Crop", 
                  command=lambda: apply_crop('content_aware'), width=18).pack(pady=3)
        ttk.Button(button_frame, text="Simple Border Crop", 
                  command=lambda: apply_crop('simple'), width=18).pack(pady=3)
        ttk.Button(button_frame, text="No Crop (Original)", 
                  command=lambda: apply_crop('none'), width=18).pack(pady=3)
        ttk.Button(button_frame, text="Cancel", 
                  command=crop_dialog.destroy, width=18).pack(pady=8)
    
    def on_closing(self):
        if self.camera_active:
            self.stop_camera()
        
        self.camera_manager.clear_captures()
        self.root.destroy()

    # Add this method for saving result in File Mode
    def save_result_file(self):
        if self.result_image is None:
            return
        try:
            result_size_mb = (self.result_image.nbytes / (1024 * 1024))
            if os.name == 'posix':
                stat = os.statvfs('/')
                free_space_mb = (stat.f_bavail * stat.f_frsize) / (1024 * 1024)
            else:
                import ctypes
                free_bytes = ctypes.c_ulonglong(0)
                ctypes.windll.kernel32.GetDiskFreeSpaceExW(ctypes.c_wchar_p('.'), None, None, ctypes.pointer(free_bytes))
                free_space_mb = free_bytes.value / (1024 * 1024)
            if free_space_mb < (result_size_mb * 2):
                messagebox.showwarning("Disk Space Warning", 
                                      f"Low disk space: {free_space_mb:.1f} MB available.\n"
                                      f"Panorama requires approximately {result_size_mb:.1f} MB.")
        except:
            pass
        output_path = filedialog.asksaveasfilename(
            title="Save Panorama",
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")]
        )
        if output_path:
            try:
                result_bgr = cv2.cvtColor(self.result_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, result_bgr)
                messagebox.showinfo("Success", f"Panorama saved to {output_path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save: {str(e)}")

def main():
    root = tk.Tk()
    app = PanoramaGUI(root)
    root.mainloop()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--gui':
        main()
    elif len(sys.argv) >= 3:
        image_paths = []
        for arg in sys.argv[1:]:
            if arg.startswith('--'):
                break
            image_paths.append(arg)
        idx = 1 + len(image_paths)
        output_path = sys.argv[idx] if len(sys.argv) > idx else 'panorama_advanced.jpg'
        blend_method = sys.argv[idx+1] if len(sys.argv) > idx+1 else 'feather'
        homography_method = sys.argv[idx+2] if len(sys.argv) > idx+2 else 'ransac'
        auto_crop = '--no-crop' not in sys.argv

        print("🚀 Starting advanced panorama stitching...")
        panorama = create_panorama_advanced(
            image_paths, output_path, blend_method, homography_method, auto_crop
        )
        
        if panorama is not None:
            print("✨ Success! Use --display to show result")
            if '--display' in sys.argv:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(15, 8))
                plt.imshow(panorama)
                plt.axis('off')
                title = f'Advanced Panorama - {blend_method} blend, {homography_method} homography'
                if auto_crop:
                    title += ' (auto-cropped)'
                plt.title(title)
                plt.tight_layout()
                plt.show()
    else:
        print("Usage:")
        print("  GUI mode: python script.py --gui")
        print("  CLI mode: python script.py image1.jpg image2.jpg [output.jpg] [blend_method] [homography_method]")
        print("  Blend methods: simple, feather, multiband")
        print("  Homography methods: linear, ransac")
        print("  Add --no-crop to disable auto-cropping")
        print("  Add --display to show result")