"""
feature_extractor.py
Extracts HSV Color Histograms, HOG / Edge Gradient Histograms, Shape/Contour Metrics,
Color Ratios, Arrow Direction, and Domain Indicators for Traffic Sign Classification across 17 Classes.
"""

import cv2
import numpy as np
from PIL import Image


def load_image_cv2(image_input):
    """Convert file path, PIL Image, or numpy array to OpenCV BGR format."""
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Could not load image from path: {image_input}")
        return img
    elif isinstance(image_input, Image.Image):
        rgb = np.array(image_input.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        return image_input
    else:
        raise TypeError(f"Unsupported image type: {type(image_input)}")


def crop_sign_roi(img_bgr):
    """Detect sign boundary using color & edge contours (Red, Blue, Yellow, Orange) and crop with 5% padding."""
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 150)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, np.array([0, 50, 40]), np.array([12, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([165, 50, 40]), np.array([180, 255, 255]))
    blue = cv2.inRange(hsv, np.array([95, 50, 40]), np.array([135, 255, 255]))
    yellow = cv2.inRange(hsv, np.array([12, 50, 40]), np.array([38, 255, 255]))
    color_mask = cv2.bitwise_or(cv2.bitwise_or(red1, red2), cv2.bitwise_or(blue, yellow))

    combined = cv2.bitwise_or(edges, color_mask)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area > (w * h * 0.08):
            x, y, bw, bh = cv2.boundingRect(c)
            pad_x = int(bw * 0.05)
            pad_y = int(bh * 0.05)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + bw + pad_x)
            y2 = min(h, y + bh + pad_y)
            cropped = img_bgr[y1:y2, x1:x2]
            if cropped.shape[0] > 10 and cropped.shape[1] > 10:
                return cropped

    return img_bgr


def extract_hsv_histogram(img_bgr, bins_h=16, bins_s=8, bins_v=8):
    """Extract normalized HSV color histogram."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [bins_h], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [bins_s], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [bins_v], [0, 256])

    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()

    return np.hstack([hist_h, hist_s, hist_v])


def extract_hog_features(img_bgr, target_size=(64, 64), cell_grid=(4, 4), orientation_bins=8):
    """Extract Histogram of Oriented Gradients over spatial grid cells."""
    resized = cv2.resize(img_bgr, target_size)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    angle = np.mod(angle, 180.0)

    cell_h = target_size[1] // cell_grid[1]
    cell_w = target_size[0] // cell_grid[0]

    hog_vector = []
    for row in range(cell_grid[1]):
        for col in range(cell_grid[0]):
            cell_mag = magnitude[row * cell_h:(row + 1) * cell_h, col * cell_w:(col + 1) * cell_w]
            cell_ang = angle[row * cell_h:(row + 1) * cell_h, col * cell_w:(col + 1) * cell_w]

            hist, _ = np.histogram(cell_ang, bins=orientation_bins, range=(0, 180), weights=cell_mag)
            norm = np.linalg.norm(hist) + 1e-6
            hist = hist / norm
            hog_vector.extend(hist)

    return np.array(hog_vector, dtype=np.float32)


def analyze_shape_and_contours(img_bgr):
    """Detect polygon shape, vertex count, aspect ratio, and circularity."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    num_vertices = 0
    aspect_ratio = 1.0
    solidity = 0.0
    circularity = 0.0
    detected_shape = "Unknown"

    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)

        if perimeter > 0 and area > 100:
            epsilon = 0.03 * perimeter
            approx = cv2.approxPolyDP(c, epsilon, True)
            num_vertices = len(approx)

            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / h if h > 0 else 1.0

            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0.0
            circularity = (4 * np.pi * area) / (perimeter ** 2)

            if circularity > 0.70:
                detected_shape = "Circle"
            elif num_vertices == 3 or len(approx) == 3:
                top_y = min([p[0][1] for p in approx])
                center_y = y + h / 2
                if top_y < center_y - h * 0.15:
                    detected_shape = "Upright Triangle"
                else:
                    detected_shape = "Inverted Triangle"
            elif num_vertices in [7, 8]:
                detected_shape = "Octagon"
            elif num_vertices in [4, 5, 6]:
                detected_shape = f"Polygon ({num_vertices} sides)"
            else:
                detected_shape = f"Shape ({num_vertices} vertices)"

    shape_features = np.array([num_vertices, aspect_ratio, solidity, circularity], dtype=np.float32)
    return shape_features, detected_shape


def analyze_dominant_color(img_bgr):
    """Determine dominant color in the sign image."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    red_mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
    red_count = np.sum(red_mask1 > 0) + np.sum(red_mask2 > 0)

    blue_mask = cv2.inRange(hsv, np.array([100, 70, 50]), np.array([130, 255, 255]))
    blue_count = np.sum(blue_mask > 0)

    yellow_mask = cv2.inRange(hsv, np.array([12, 70, 50]), np.array([38, 255, 255]))
    yellow_count = np.sum(yellow_mask > 0)

    color_counts = {"Red": red_count, "Blue": blue_count, "Yellow": yellow_count}
    dominant_color = max(color_counts, key=color_counts.get)
    if color_counts[dominant_color] < (img_bgr.shape[0] * img_bgr.shape[1] * 0.05):
        dominant_color = "Neutral / White"

    return dominant_color


def detect_traffic_light_features(img_bgr):
    """Detect yellow/amber and green light spots inside traffic signal signs."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, np.array([12, 70, 70]), np.array([35, 255, 255]))
    green_mask = cv2.inRange(hsv, np.array([35, 70, 70]), np.array([88, 255, 255]))

    yellow_pixels = int(np.sum(yellow_mask > 0))
    green_pixels = int(np.sum(green_mask > 0))

    has_traffic_lights = (yellow_pixels >= 12 and green_pixels >= 12)
    return has_traffic_lights, yellow_pixels, green_pixels


def detect_arrow_direction(img_bgr):
    """Detect horizontal bias of white arrow pixels (Left vs Right)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 50, 255]))
    
    h, w = white_mask.shape[:2]
    left_half = np.sum(white_mask[:, :w//2] > 0)
    right_half = np.sum(white_mask[:, w//2:] > 0)

    if right_half > left_half * 1.2:
        return "Right"
    elif left_half > right_half * 1.2:
        return "Left"
    return "Balanced"


def detect_bus_icon(img_bgr):
    """Detect bus body outline inside blue information sign."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, np.array([95, 50, 40]), np.array([135, 255, 255]))
    blue_ratio = float(np.sum(blue_mask > 0)) / float(img_bgr.shape[0] * img_bgr.shape[1])

    # Check for white rectangle body in center of blue background
    white_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 50, 255]))
    h, w = white_mask.shape[:2]
    center_roi = white_mask[int(h*0.25):int(h*0.75), int(w*0.2):int(w*0.8)]
    center_white_ratio = float(np.sum(center_roi > 0)) / float(center_roi.size)

    has_bus = (blue_ratio > 0.40) and (center_white_ratio > 0.35)
    return has_bus


def extract_all_features(image_input):
    """Combined feature extraction function returning 1D numpy array and metadata."""
    img_bgr = load_image_cv2(image_input)
    roi_bgr = crop_sign_roi(img_bgr)

    # 1. Color Histogram (32) on cropped sign ROI
    color_hist = extract_hsv_histogram(roi_bgr)

    # 2. HOG Features (128) on cropped sign ROI
    hog_feats = extract_hog_features(roi_bgr)

    # 3. Shape Analysis (4) on full image / ROI
    shape_feats, shape_name = analyze_shape_and_contours(img_bgr)

    # 4. Pixel Intensity Vector (32x32 grayscale downsampled = 1024)
    resized_gray = cv2.resize(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY), (32, 32))
    pixel_vector = (resized_gray.flatten().astype(np.float32)) / 255.0

    # 5. Color Ratios (Red, Blue, Yellow, White, Black)
    hsv_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    tot_pixels = float(roi_bgr.shape[0] * roi_bgr.shape[1])

    red1 = cv2.inRange(hsv_roi, np.array([0, 70, 50]), np.array([10, 255, 255]))
    red2 = cv2.inRange(hsv_roi, np.array([170, 70, 50]), np.array([180, 255, 255]))
    red_ratio = float(np.sum(red1 > 0) + np.sum(red2 > 0)) / tot_pixels

    blue_mask = cv2.inRange(hsv_roi, np.array([95, 50, 40]), np.array([135, 255, 255]))
    blue_ratio = float(np.sum(blue_mask > 0)) / tot_pixels

    yellow_mask = cv2.inRange(hsv_roi, np.array([12, 60, 50]), np.array([38, 255, 255]))
    yellow_ratio = float(np.sum(yellow_mask > 0)) / tot_pixels

    white_mask = cv2.inRange(hsv_roi, np.array([0, 0, 180]), np.array([180, 50, 255]))
    white_ratio = float(np.sum(white_mask > 0)) / tot_pixels

    true_black_mask = (roi_bgr[:, :, 2] < 80) & (roi_bgr[:, :, 1] < 80) & (roi_bgr[:, :, 0] < 80)
    black_ratio = float(np.sum(true_black_mask)) / tot_pixels

    color_ratios = np.array([red_ratio, blue_ratio, yellow_ratio, white_ratio, black_ratio], dtype=np.float32)

    # 6. Specialized Domain Indicators
    has_traffic_lights, yellow_pix, green_pix = detect_traffic_light_features(roi_bgr)
    arrow_dir = detect_arrow_direction(roi_bgr)
    has_bus = detect_bus_icon(roi_bgr)

    # Black silhouette count inside sign interior
    h, w = roi_bgr.shape[:2]
    inner_black = true_black_mask[int(h*0.25):int(h*0.8), int(w*0.2):int(w*0.8)]
    has_pedestrian = (float(np.sum(inner_black)) / float(inner_black.size)) > 0.035

    domain_feats = np.array([
        float(has_traffic_lights),
        float(has_pedestrian),
        float(has_bus),
        float(arrow_dir == "Left"),
        float(arrow_dir == "Right"),
        float(yellow_pix) / 500.0,
        float(green_pix) / 500.0
    ], dtype=np.float32)

    full_vector = np.hstack([color_hist, hog_feats, shape_feats, pixel_vector, color_ratios, domain_feats])
    dominant_color = analyze_dominant_color(roi_bgr)

    flags = {
        "has_traffic_lights": has_traffic_lights,
        "has_pedestrian": has_pedestrian,
        "has_bus": has_bus,
        "arrow_dir": arrow_dir,
        "red_ratio": round(red_ratio, 2),
        "blue_ratio": round(blue_ratio, 2),
        "yellow_ratio": round(yellow_ratio, 2),
        "black_ratio": round(black_ratio, 2)
    }

    return full_vector, shape_name, dominant_color, flags


if __name__ == "__main__":
    from dataset_generator import create_base_traffic_sign
    sample_img = create_base_traffic_sign("school_zone")
    features, shape, color, flags = extract_all_features(sample_img)
    print(f"Extracted feature vector length: {len(features)}")
    print(f"Detected shape: {shape}, Dominant color: {color}, Flags: {flags}")
