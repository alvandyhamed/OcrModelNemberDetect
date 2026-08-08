import os
import cv2
import json
import glob
import numpy as np
import easyocr

def preprocess_chart(image_path):
    """
    Preprocess technical chart nomograms to reduce grid pattern noise
    and enhance text/number contrast for OCR.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Adaptive thresholding to isolate dark text from grid background
    binarized = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 15, 8
    )
    
    # Morphological grid dot removal
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binarized, cv2.MORPH_OPEN, kernel)
    
    return img, gray, binarized, cv2.bitwise_not(cleaned)

def extract_isolated_digits(gray, reader):
    """
    Extract small/isolated single digits (e.g. axis ticks 0, 1, 2, 3, 4, 5) using ROI contours.
    """
    _, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    isolated_results = []
    h_img, w_img = gray.shape[:2]
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Dimensions typical for single axis tick numbers
        if 5 <= w <= 45 and 8 <= h <= 45:
            # Skip borders
            if x < 5 or y < 5 or (x + w) > (w_img - 5) or (y + h) > (h_img - 5):
                continue
                
            pad = 5
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(w_img, x + w + pad), min(h_img, y + h + pad)
            crop = gray[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue
                
            # Upscale 3x for clear OCR recognition
            crop_large = cv2.resize(crop, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            res = reader.readtext(crop_large, allowlist='0123456789', detail=1)
            
            for _, text, conf in res:
                text_clean = text.strip()
                if text_clean.isdigit() and conf > 0.2:
                    bbox = [
                        [x, y],
                        [x + w, y],
                        [x + w, y + h],
                        [x, y + h]
                    ]
                    isolated_results.append((bbox, text_clean, conf))
                    
    return isolated_results

def extract_chart_data(image_path, reader, output_dir="extracted_output"):
    """
    Complete Hybrid Extraction Pipeline:
    1. Pass 1: Full-scene EasyOCR (Raw & Cleaned)
    2. Pass 2: Contour ROI Isolated Digit Detector
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    img, gray, binarized, cleaned_bg = preprocess_chart(image_path)
    
    # 1. Full scene OCR
    raw_results = reader.readtext(img)
    clean_results = reader.readtext(cleaned_bg)
    
    # 2. Isolated digit contour OCR
    isolated_digits = extract_isolated_digits(gray, reader)
    
    all_detections = []
    seen_regions = []
    
    def is_overlapping(box1, box2, threshold=0.5):
        # Calculate overlap IoU box
        xs1 = [pt[0] for pt in box1]
        ys1 = [pt[1] for pt in box1]
        xs2 = [pt[0] for pt in box2]
        ys2 = [pt[1] for pt in box2]
        
        inter_x1 = max(min(xs1), min(xs2))
        inter_y1 = max(min(ys1), min(ys2))
        inter_x2 = min(max(xs1), max(xs2))
        inter_y2 = min(max(ys1), max(ys2))
        
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            area1 = (max(xs1) - min(xs1)) * (max(ys1) - min(ys1))
            area2 = (max(xs2) - min(xs2)) * (max(ys2) - min(ys2))
            iou = inter_area / float(area1 + area2 - inter_area)
            return iou > threshold
        return False

    def process_res(results, source):
        for bbox, text, conf in results:
            text_clean = text.strip()
            if not text_clean or conf < 0.15:
                continue
            
            box_coords = [[int(pt[0]), int(pt[1])] for pt in bbox]
            
            # Check overlap with existing
            duplicate = False
            for existing in seen_regions:
                if is_overlapping(box_coords, existing["bounding_box"]):
                    duplicate = True
                    break
            if duplicate:
                continue
                
            is_digit = text_clean.replace('.', '').replace('-', '').replace(',', '').isdigit()
            
            item = {
                "text": text_clean,
                "confidence": round(float(conf), 3),
                "is_numeric": is_digit,
                "type": "NUMBER" if is_digit else "TEXT",
                "bounding_box": box_coords,
                "source": source
            }
            seen_regions.append(item)
            all_detections.append(item)

    process_res(raw_results, "raw_ocr")
    process_res(clean_results, "cleaned_ocr")
    process_res(isolated_digits, "contour_digit")
    
    # Sort detections by Y then X
    all_detections.sort(key=lambda d: (d["bounding_box"][0][1], d["bounding_box"][0][0]))
    
    # Draw annotations
    annotated = img.copy()
    for item in all_detections:
        pts = np.array(item["bounding_box"], np.int32).reshape((-1, 1, 2))
        color = (0, 255, 0) if item["is_numeric"] else (255, 200, 0)
        cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2)
        
        top_left = item["bounding_box"][0]
        cv2.putText(
            annotated, f"{item['text']} ({item['confidence']:.2f})", 
            (top_left[0], max(top_left[1] - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA
        )
        
    annotated_path = os.path.join(output_dir, f"{base_name}_annotated.png")
    json_path = os.path.join(output_dir, f"{base_name}_data.json")
    
    cv2.imwrite(annotated_path, annotated)
    
    summary = {
        "file": image_path,
        "total_detected": len(all_detections),
        "total_numbers": sum(1 for d in all_detections if d["is_numeric"]),
        "total_texts": sum(1 for d in all_detections if not d["is_numeric"]),
        "detections": all_detections
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        
    return summary, annotated_path

if __name__ == "__main__":
    print("Initializing EasyOCR Engine...")
    reader = easyocr.Reader(['en'], gpu=False)
    
    chart_files = sorted(glob.glob("Screenshot*.png"))
    for chart in chart_files:
        print(f"\nProcessing {chart}...")
        summary, ann_path = extract_chart_data(chart, reader)
        print(f"  -> Total Found: {summary['total_numbers']} numbers & {summary['total_texts']} text labels.")
        print(f"  -> Saved output: {ann_path}")
        print("  Extracted Numbers:")
        numbers = [d['text'] for d in summary['detections'] if d['is_numeric']]
        print("   ", numbers)
