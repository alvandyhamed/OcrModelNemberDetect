import os
import json
import glob
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
import easyocr

app = Flask(__name__, static_folder='static', template_folder='templates')

UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
OUTPUT_FOLDER = os.path.join(app.root_path, 'extracted_output')
CROPS_FOLDER = os.path.join(OUTPUT_FOLDER, 'crops')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CROPS_FOLDER, exist_ok=True)

reader = None

def get_ocr_reader():
    global reader
    if reader is None:
        print("Loading EasyOCR Engine...")
        reader = easyocr.Reader(['en'], gpu=False)
    return reader

def extract_isolated_digits(gray, reader):
    """
    Extract small/isolated digits (e.g. axis ticks 0, 1, 2, 3, 4, 5) using ROI contours.
    """
    _, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    isolated_results = []
    h_img, w_img = gray.shape[:2]
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if 5 <= w <= 45 and 8 <= h <= 45:
            if x < 5 or y < 5 or (x + w) > (w_img - 5) or (y + h) > (h_img - 5):
                continue
            pad = 5
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(w_img, x + w + pad), min(h_img, y + h + pad)
            crop = gray[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_large = cv2.resize(crop, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            res = reader.readtext(crop_large, allowlist='0123456789', detail=1)
            for _, text, conf in res:
                text_clean = text.strip()
                if text_clean.isdigit() and conf > 0.2:
                    bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                    isolated_results.append((bbox, text_clean, conf))
    return isolated_results

def process_single_image(filepath, param_block_size=15, param_c=8):
    ocr = get_ocr_reader()
    img = cv2.imread(filepath)
    if img is None:
        return None
    
    filename = os.path.basename(filepath)
    base_name = os.path.splitext(filename)[0]
    h_img, w_img = img.shape[:2]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binarized = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, param_block_size, param_c
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binarized, cv2.MORPH_OPEN, kernel)
    cleaned_bg = cv2.bitwise_not(cleaned)
    
    clean_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_cleaned.png")
    cv2.imwrite(clean_path, cleaned_bg)
    
    raw_results = ocr.readtext(img)
    clean_results = ocr.readtext(cleaned_bg)
    isolated_digits = extract_isolated_digits(gray, ocr)
    
    detections = []
    seen_boxes = []
    
    def is_overlapping(box1, box2, threshold=0.5):
        xs1, ys1 = [pt[0] for pt in box1], [pt[1] for pt in box1]
        xs2, ys2 = [pt[0] for pt in box2], [pt[1] for pt in box2]
        ix1, iy1 = max(min(xs1), min(xs2)), max(min(ys1), min(ys2))
        ix2, iy2 = min(max(xs1), max(xs2)), min(max(ys1), max(ys2))
        if ix1 < ix2 and iy1 < iy2:
            iarea = (ix2 - ix1) * (iy2 - iy1)
            area1 = (max(xs1) - min(xs1)) * (max(ys1) - min(ys1))
            area2 = (max(xs2) - min(xs2)) * (max(ys2) - min(ys2))
            return (iarea / float(area1 + area2 - iarea)) > threshold
        return False

    def add_res(res_list, src):
        for bbox, text, conf in res_list:
            clean_t = text.strip()
            if not clean_t or conf < 0.15:
                continue
            coords = [[int(pt[0]), int(pt[1])] for pt in bbox]
            
            if any(is_overlapping(coords, s["polygon"]) for s in seen_boxes):
                continue
                
            xs = [pt[0] for pt in coords]
            ys = [pt[1] for pt in coords]
            x, y, w, h = min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
            
            # Save crop snippet image
            pad = 4
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(w_img, x + w + pad), min(h_img, y + h + pad)
            crop_img = img[y1:y2, x1:x2]
            
            crop_id = len(detections) + 1
            crop_filename = f"{base_name}_crop_{crop_id}.png"
            crop_filepath = os.path.join(CROPS_FOLDER, crop_filename)
            if crop_img.size > 0:
                cv2.imwrite(crop_filepath, crop_img)
            
            is_num = clean_t.replace('.', '').replace('-', '').replace(',', '').isdigit()
            
            item = {
                "id": crop_id,
                "predicted_text": clean_t,
                "user_verified_text": clean_t, # default to prediction
                "confidence": round(float(conf), 3),
                "is_numeric": is_num,
                "type": "NUMBER" if is_num else "TEXT",
                "box": {"x": x, "y": y, "w": w, "h": h},
                "polygon": coords,
                "source": src,
                "crop_url": f"/files/crops/{crop_filename}"
            }
            seen_boxes.append(item)
            detections.append(item)

    add_res(raw_results, "raw_ocr")
    add_res(clean_results, "cleaned_ocr")
    add_res(isolated_digits, "contour_digit")
    
    detections.sort(key=lambda d: (d["box"]["y"], d["box"]["x"]))
    # Re-assign sequential IDs
    for idx, d in enumerate(detections, 1):
        d["id"] = idx

    # Draw visual annotations
    annotated = img.copy()
    for d in detections:
        pts = np.array(d["polygon"], np.int32).reshape((-1, 1, 2))
        color = (0, 255, 0) if d["is_numeric"] else (255, 200, 0)
        cv2.polylines(annotated, [pts], True, color, 2)
        
        top_l = d["polygon"][0]
        cv2.putText(
            annotated, f"#{d['id']} {d['predicted_text']}", 
            (top_l[0], max(top_l[1] - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA
        )
        
    ann_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_annotated.png")
    cv2.imwrite(ann_path, annotated)
    
    data = {
        "filename": filename,
        "base_name": base_name,
        "image_width": w_img,
        "image_height": h_img,
        "image_url": f"/files/uploads/{filename}",
        "cleaned_url": f"/files/output/{base_name}_cleaned.png",
        "annotated_url": f"/files/output/{base_name}_annotated.png",
        "total_count": len(detections),
        "number_count": sum(1 for d in detections if d["is_numeric"]),
        "text_count": sum(1 for d in detections if not d["is_numeric"]),
        "detections": detections
    }
    
    json_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_data.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    return data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/samples')
def get_samples():
    samples = sorted(glob.glob("Screenshot*.png"))
    return jsonify([os.path.basename(s) for s in samples])

@app.route('/api/process', methods=['POST'])
def process():
    data = request.json or {}
    filename = data.get('filename')
    if filename:
        filepath = os.path.join(app.root_path, filename)
        if not os.path.exists(filepath):
            filepath = os.path.join(UPLOAD_FOLDER, filename)
    else:
        return jsonify({"error": "No filename provided"}), 400
        
    result = process_single_image(filepath)
    if result:
        return jsonify(result)
    return jsonify({"error": "Failed to process image"}), 500

@app.route('/api/update_label', methods=['POST'])
def update_label():
    body = request.json or {}
    base_name = body.get('base_name')
    detection_id = body.get('detection_id')
    new_text = body.get('text')
    new_type = body.get('type')  # 'NUMBER' or 'TEXT'
    
    json_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_data.json")
    if not os.path.exists(json_path):
        return jsonify({"error": "Data file not found"}), 404
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    updated = False
    for d in data['detections']:
        if d['id'] == detection_id:
            if new_text is not None:
                d['user_verified_text'] = new_text.strip()
            if new_type in ["NUMBER", "TEXT"]:
                d['type'] = new_type
                d['is_numeric'] = (new_type == "NUMBER")
            else:
                d['is_numeric'] = d['user_verified_text'].replace('.', '').replace('-', '').isdigit()
                d['type'] = "NUMBER" if d['is_numeric'] else "TEXT"
            updated = True
            break
            
    if updated:
        data['number_count'] = sum(1 for d in data['detections'] if d["is_numeric"])
        data['text_count'] = sum(1 for d in data['detections'] if not d["is_numeric"])
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return jsonify({"success": True, "data": data})
    return jsonify({"error": "Detection ID not found"}), 404

@app.route('/api/export_labelstudio/<base_name>')
def export_labelstudio(base_name):
    json_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_data.json")
    if not os.path.exists(json_path):
        return jsonify({"error": "File not found"}), 404
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    w_img = data.get('image_width', 1000)
    h_img = data.get('image_height', 1000)
    
    results = []
    for d in data['detections']:
        bx = d['box']
        # Convert pixel coordinates to Label Studio percentage (0..100)
        px = (bx['x'] / float(w_img)) * 100.0
        py = (bx['y'] / float(h_img)) * 100.0
        pw = (bx['w'] / float(w_img)) * 100.0
        ph = (bx['h'] / float(h_img)) * 100.0
        
        label_val = d['user_verified_text']
        
        results.append({
            "id": f"bbox_{d['id']}",
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "value": {
                "x": round(px, 3),
                "y": round(py, 3),
                "width": round(pw, 3),
                "height": round(ph, 3),
                "rotation": 0,
                "rectanglelabels": ["NUMBER" if d['is_numeric'] else "TEXT"]
            }
        })
        results.append({
            "id": f"bbox_{d['id']}",
            "from_name": "transcription",
            "to_name": "image",
            "type": "textarea",
            "value": {
                "x": round(px, 3),
                "y": round(py, 3),
                "width": round(pw, 3),
                "height": round(ph, 3),
                "rotation": 0,
                "text": [label_val]
            }
        })
        
    ls_task = [{
        "data": {
            "image": f"/data/upload/{data['filename']}"
        },
        "predictions": [{
            "model_version": "hybrid_chart_ocr_v1",
            "result": results
        }]
    }]
    
    export_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_labelstudio_import.json")
    with open(export_path, 'w', encoding='utf-8') as f:
        json.dump(ls_task, f, indent=2, ensure_ascii=False)
        
    return send_file(export_path, as_attachment=True)

@app.route('/api/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
        
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)
    
    result = process_single_image(path)
    return jsonify(result)

@app.route('/files/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(app.root_path, filename)

@app.route('/files/output/<path:filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/files/crops/<path:filename>')
def serve_crop(filename):
    return send_from_directory(CROPS_FOLDER, filename)

if __name__ == '__main__':
    print("Starting Chart OCR Web Server on http://127.0.0.1:5050...")
    app.run(host='0.0.0.0', port=5050, debug=False)
