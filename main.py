import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

def generate_synthetic_image():
    """Generates a synthetic test image with colored pills/coins of varying sizes."""
    img = np.ones((500, 600, 3), dtype=np.uint8) * 40  # Dark background
    
    # Synthetic objects: (center_x, center_y, radius, BGR_color)
    objects = [
        (150, 150, 35, (50, 50, 220)),   # Red Pill 1
        (200, 160, 35, (40, 40, 210)),   # Red Pill 2 (touching Pill 1)
        (350, 120, 45, (220, 180, 50)),  # Gold Coin 1
        (420, 130, 40, (210, 175, 45)),  # Gold Coin 2 (touching Coin 1)
        (180, 350, 25, (80, 200, 80)),   # Green Pill 1
        (230, 370, 25, (75, 190, 75)),   # Green Pill 2
        (400, 350, 48, (230, 190, 60)),  # Gold Coin 3
    ]
    
    for cx, cy, r, color in objects:
        cv2.circle(img, (cx, cy), r, color, -1)
        noise = np.random.randint(-15, 15, (r*2, r*2, 3), dtype=np.int16)
        y1, y2 = max(0, cy-r), min(img.shape[0], cy+r)
        x1, x2 = max(0, cx-r), min(img.shape[1], cx+r)
        img[y1:y2, x1:x2] = np.clip(img[y1:y2, x1:x2].astype(np.int16) + noise[:y2-y1, :x2-x1], 0, 255).astype(np.uint8)

    return img

def preprocess_and_segment(image):
    """Module 1 & 2: Contrast enhancement, noise removal, and Watershed Segmentation."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    
    # Noise Removal
    blurred = cv2.GaussianBlur(equalized, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Distance Transform & Watershed for separating touching objects
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    markers = cv2.watershed(image.copy(), markers)
    return markers, opening

def extract_features(image, markers):
    """Module 2 & 5: Feature Extraction (HSV Colors & Geometric Circularity)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    features = []
    metadata = []
    
    unique_markers = np.unique(markers)
    
    for marker in unique_markers:
        if marker <= 1:
            continue
            
        mask = np.zeros(markers.shape, dtype="uint8")
        mask[markers == marker] = 255
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
            
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
            
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        mean_hsv = cv2.mean(hsv, mask=mask)[:3]
        
        features.append([mean_hsv[0], mean_hsv[1], mean_hsv[2], area, circularity])
        
        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
        
        metadata.append({"contour": cnt, "center": (cx, cy), "area": area})
        
    return np.array(features), metadata

def main():
    image = generate_synthetic_image()
    markers, binary_mask = preprocess_and_segment(image)
    features, metadata = extract_features(image, markers)
    
    if len(features) == 0:
        print("No objects detected.")
        return

    # Dimensionality Reduction using PCA
    normalized_features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
    pca = PCA(n_components=min(2, normalized_features.shape[1]))
    pca_features = pca.fit_transform(normalized_features)

    # Clustering & Classification
    num_clusters = min(3, len(features))
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(pca_features)

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(pca_features, cluster_labels)
    predicted_labels = knn.predict(pca_features)

    # Render Results
    output = image.copy()
    category_counts = {}

    for i, meta in enumerate(metadata):
        label = predicted_labels[i]
        category_counts[label] = category_counts.get(label, 0) + 1
        cnt = meta["contour"]
        cx, cy = meta["center"]
        
        cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)
        cv2.putText(output, f"ID:{i+1} (Cat {label})", (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    hud_y = 30
    cv2.putText(output, f"Total Objects: {len(metadata)}", (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    for cat, count in category_counts.items():
        hud_y += 22
        cv2.putText(output, f" -> Category {cat} Count: {count}", (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    binary_3ch = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    stacked_view = np.hstack((image, binary_3ch, output))
    
    cv2.imwrite("pill_coin_counter_output.png", stacked_view)
    print("Process complete! Output image saved as 'pill_coin_counter_output.png'.")

if __name__ == "__main__":
    main()