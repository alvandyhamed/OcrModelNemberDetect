import os
import io
import zipfile
import json
import glob
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory, Response, make_response
import easyocr

import minio_helper

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # 1GB limit

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

def process_chart_bytes(img_bytes, filename, save_to_minio=False, dataset_name=None):
    ocr = get_ocr_reader()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None
        
    base_name = os.path.splitext(os.path.basename(filename))[0]
    h_img, w_img = img.shape[:2]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binarized = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 15, 8
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binarized, cv2.MORPH_OPEN, kernel)
    cleaned_bg = cv2.bitwise_not(cleaned)
    
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
            
            pad = 4
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(w_img, x + w + pad), min(h_img, y + h + pad)
            crop_img = img[y1:y2, x1:x2]
            
            crop_id = len(detections) + 1
            crop_filename = f"{base_name}_crop_{crop_id}.png"
            
            _, crop_buf = cv2.imencode('.png', crop_img)
            crop_bytes = crop_buf.tobytes() if crop_img.size > 0 else b''
            
            crop_url = f"/files/crops/{crop_filename}"
            if save_to_minio and dataset_name:
                crop_key = f"processed_datasets/{dataset_name}/crops/{crop_filename}"
                minio_helper.upload_file_bytes(crop_key, crop_bytes, 'image/png')
                crop_url = f"/files/minio/{crop_key}"
            else:
                crop_filepath = os.path.join(CROPS_FOLDER, crop_filename)
                if crop_img.size > 0:
                    cv2.imwrite(crop_filepath, crop_img)
            
            is_num = clean_t.replace('.', '').replace('-', '').replace(',', '').isdigit()
            
            item = {
                "id": crop_id,
                "predicted_text": clean_t,
                "user_verified_text": clean_t,
                "confidence": round(float(conf), 3),
                "is_numeric": is_num,
                "type": "NUMBER" if is_num else "TEXT",
                "box": {"x": x, "y": y, "w": w, "h": h},
                "polygon": coords,
                "source": src,
                "crop_url": crop_url
            }
            seen_boxes.append(item)
            detections.append(item)

    add_res(raw_results, "raw_ocr")
    add_res(clean_results, "cleaned_ocr")
    add_res(isolated_digits, "contour_digit")
    
    detections.sort(key=lambda d: (d["box"]["y"], d["box"]["x"]))
    for idx, d in enumerate(detections, 1):
        d["id"] = idx

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
        
    _, ann_buf = cv2.imencode('.png', annotated)
    ann_bytes = ann_buf.tobytes()
    
    _, clean_buf = cv2.imencode('.png', cleaned_bg)
    clean_bytes = clean_buf.tobytes()
    
    if save_to_minio and dataset_name:
        ann_key = f"processed_datasets/{dataset_name}/annotated/{base_name}_annotated.png"
        clean_key = f"processed_datasets/{dataset_name}/cleaned/{base_name}_cleaned.png"
        minio_helper.upload_file_bytes(ann_key, ann_bytes, 'image/png')
        minio_helper.upload_file_bytes(clean_key, clean_bytes, 'image/png')
        
        annotated_url = f"/files/minio/{ann_key}"
        cleaned_url = f"/files/minio/{clean_key}"
        image_url = f"/files/minio/raw_uploads/{dataset_name}/{filename}"
    else:
        ann_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_annotated.png")
        clean_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_cleaned.png")
        cv2.imwrite(ann_path, annotated)
        cv2.imwrite(clean_path, cleaned_bg)
        
        annotated_url = f"/files/output/{base_name}_annotated.png"
        cleaned_url = f"/files/output/{base_name}_cleaned.png"
        image_url = f"/files/uploads/{filename}"
        
    data = {
        "filename": filename,
        "base_name": base_name,
        "image_width": w_img,
        "image_height": h_img,
        "image_url": image_url,
        "cleaned_url": cleaned_url,
        "annotated_url": annotated_url,
        "total_count": len(detections),
        "number_count": sum(1 for d in detections if d["is_numeric"]),
        "text_count": sum(1 for d in detections if not d["is_numeric"]),
        "detections": detections
    }
    
    if save_to_minio and dataset_name:
        json_key = f"processed_datasets/{dataset_name}/json/{base_name}_data.json"
        minio_helper.save_json_to_minio(json_key, data)
    else:
        json_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_data.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
    return data

@app.route('/')
def index():
    response = make_response(render_template('index.html'))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/api/samples')
def get_samples():
    samples = sorted(glob.glob("Screenshot*.png"))
    return jsonify([os.path.basename(s) for s in samples])

@app.route('/api/upload_image', methods=['POST'])
def upload_single_image():
    if 'file' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
        
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    with open(filepath, 'rb') as f:
        img_bytes = f.read()
        
    result = process_chart_bytes(img_bytes, file.filename)
    if result:
        return jsonify({"success": True, "data": result})
    return jsonify({"error": "Failed to process uploaded image"}), 500

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
        
    with open(filepath, 'rb') as f:
        img_bytes = f.read()
        
    result = process_chart_bytes(img_bytes, filename)
    if result:
        return jsonify(result)
    return jsonify({"error": "Failed to process image"}), 500

@app.route('/api/update_label', methods=['POST'])
def update_label():
    body = request.json or {}
    base_name = body.get('base_name')
    detection_id = body.get('detection_id')
    new_text = body.get('text')
    new_type = body.get('type')
    
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

# --- MinIO Dataset API Endpoints ---

@app.route('/api/minio/upload_zip', methods=['POST'])
def minio_upload_zip():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No ZIP file uploaded"}), 400
        file = request.files['file']
        if not file.filename:
            return jsonify({"error": "No selected file"}), 400
        if not file.filename.lower().endswith('.zip'):
            return jsonify({"error": "File must be a .zip archive"}), 400
            
        clean_name = os.path.splitext(file.filename)[0].replace(' ', '_')
        dataset_name = "".join(c for c in clean_name if c.isalnum() or c in ('_', '-')).strip('_')
        if not dataset_name:
            dataset_name = "dataset_zip"
        
        temp_path = os.path.join(UPLOAD_FOLDER, f"temp_{dataset_name}.zip")
        file.save(temp_path)
        
        try:
            extracted_files = minio_helper.extract_zip_file_to_minio(temp_path, dataset_name)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        return jsonify({
            "success": True,
            "dataset_name": dataset_name,
            "total_extracted": len(extracted_files),
            "files": extracted_files
        })
    except Exception as e:
        print(f"ZIP Upload Exception: {e}")
        return jsonify({"error": f"ZIP Upload Error: {str(e)}"}), 500

@app.route('/api/minio/datasets')
def minio_list_datasets():
    datasets = minio_helper.list_raw_datasets()
    return jsonify(datasets)

@app.route('/api/minio/process_dataset_stream', methods=['POST'])
def minio_process_dataset_stream():
    """
    Server-Sent Events (SSE) streaming batch processor.
    Streams real-time progress updates for every single chart image processed.
    """
    body = request.json or {}
    dataset_name = body.get('dataset_name')
    if not dataset_name:
        return jsonify({"error": "No dataset_name provided"}), 400
        
    image_keys = minio_helper.list_dataset_images(dataset_name)
    total_images = len(image_keys)

    def generate_events():
        yield f"data: {json.dumps({'type': 'start', 'total': total_images, 'dataset_name': dataset_name})}\n\n"
        
        for idx, key in enumerate(image_keys, 1):
            filename = os.path.basename(key)
            try:
                img_bytes = minio_helper.get_object_bytes(key)
                result = process_chart_bytes(
                    img_bytes, filename, 
                    save_to_minio=True, dataset_name=dataset_name
                )
                
                event_payload = {
                    "type": "progress",
                    "current": idx,
                    "total": total_images,
                    "filename": filename,
                    "numbers_found": result["number_count"] if result else 0,
                    "texts_found": result["text_count"] if result else 0,
                    "percentage": round((idx / float(total_images)) * 100, 1)
                }
                yield f"data: {json.dumps(event_payload)}\n\n"
            except Exception as e:
                error_payload = {
                    "type": "error_item",
                    "current": idx,
                    "total": total_images,
                    "filename": filename,
                    "error": str(e)
                }
                yield f"data: {json.dumps(error_payload)}\n\n"
                
        yield f"data: {json.dumps({'type': 'complete', 'total': total_images, 'dataset_name': dataset_name})}\n\n"

    return Response(generate_events(), mimetype='text/event-stream')

@app.route('/api/minio/dataset_details/<dataset_name>')
def minio_dataset_details(dataset_name):
    s3 = minio_helper.get_s3_client()
    prefix = f"processed_datasets/{dataset_name}/json/"
    paginator = s3.get_paginator('list_objects_v2')
    
    all_json_data = []
    for page in paginator.paginate(Bucket=minio_helper.MINIO_BUCKET, Prefix=prefix):
        for obj in page.get('Contents', []):
            json_data = minio_helper.read_json_from_minio(obj['Key'])
            if json_data:
                json_data['json_key'] = obj['Key']
                all_json_data.append(json_data)
                
    return jsonify({
        "dataset_name": dataset_name,
        "charts": all_json_data
    })

@app.route('/api/minio/update_crop_label', methods=['POST'])
def minio_update_crop_label():
    body = request.json or {}
    json_key = body.get('json_key')
    detection_id = body.get('detection_id')
    new_text = body.get('text')
    new_type = body.get('type')
    
    if not json_key:
        return jsonify({"error": "json_key is required"}), 400
        
    data = minio_helper.read_json_from_minio(json_key)
    if not data:
        return jsonify({"error": "JSON object not found in MinIO"}), 404
        
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
        minio_helper.save_json_to_minio(json_key, data)
        return jsonify({"success": True, "data": data})
        
    return jsonify({"error": "Detection ID not found"}), 404

@app.route('/api/minio/export_training_zip/<dataset_name>')
def minio_export_training_zip(dataset_name):
    s3 = minio_helper.get_s3_client()
    prefix = f"processed_datasets/{dataset_name}/json/"
    paginator = s3.get_paginator('list_objects_v2')
    
    zip_buffer = io.BytesIO()
    all_labelstudio_tasks = []
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for page in paginator.paginate(Bucket=minio_helper.MINIO_BUCKET, Prefix=prefix):
            for obj in page.get('Contents', []):
                data = minio_helper.read_json_from_minio(obj['Key'])
                if not data:
                    continue
                    
                filename = data['filename']
                base_name = data['base_name']
                w_img = data.get('image_width', 1000)
                h_img = data.get('image_height', 1000)
                
                ls_results = []
                for d in data['detections']:
                    bx = d['box']
                    px = (bx['x'] / float(w_img)) * 100.0
                    py = (bx['y'] / float(h_img)) * 100.0
                    pw = (bx['w'] / float(w_img)) * 100.0
                    ph = (bx['h'] / float(h_img)) * 100.0
                    
                    ls_results.append({
                        "id": f"bbox_{d['id']}",
                        "from_name": "label",
                        "to_name": "image",
                        "type": "rectanglelabels",
                        "value": {
                            "x": round(px, 3), "y": round(py, 3),
                            "width": round(pw, 3), "height": round(ph, 3),
                            "rotation": 0,
                            "rectanglelabels": ["NUMBER" if d['is_numeric'] else "TEXT"]
                        }
                    })
                    ls_results.append({
                        "id": f"bbox_{d['id']}",
                        "from_name": "transcription",
                        "to_name": "image",
                        "type": "textarea",
                        "value": {
                            "x": round(px, 3), "y": round(py, 3),
                            "width": round(pw, 3), "height": round(ph, 3),
                            "rotation": 0,
                            "text": [d['user_verified_text']]
                        }
                    })
                    
                all_labelstudio_tasks.append({
                    "data": {"image": f"/data/upload/{filename}"},
                    "predictions": [{"model_version": "hybrid_chart_ocr_v1", "result": ls_results}]
                })
                
                zf.writestr(f"annotations/{base_name}_data.json", json.dumps(data, indent=2, ensure_ascii=False))
                
        zf.writestr("labelstudio_import_all.json", json.dumps(all_labelstudio_tasks, indent=2, ensure_ascii=False))
        
    zip_buffer.seek(0)
    return Response(
        zip_buffer.getvalue(),
        mimetype='application/zip',
        headers={"Content-Disposition": f"attachment;filename={dataset_name}_verified_training_dataset.zip"}
    )

@app.route('/files/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(app.root_path, filename)

@app.route('/files/output/<path:filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/files/crops/<path:filename>')
def serve_crop(filename):
    return send_from_directory(CROPS_FOLDER, filename)

@app.route('/files/minio/<path:object_key>')
def serve_minio_object(object_key):
    try:
        data = minio_helper.get_object_bytes(object_key)
        ext = os.path.splitext(object_key)[1].lower()
        mimetype = 'image/png' if ext == '.png' else ('image/jpeg' if ext in ['.jpg', '.jpeg'] else 'application/octet-stream')
        return Response(data, mimetype=mimetype)
    except Exception as e:
        return jsonify({"error": f"MinIO object not found: {str(e)}"}), 404

if __name__ == '__main__':
    print("Starting Chart OCR & MinIO Dataset Web Server on http://127.0.0.1:5050...")
    app.run(host='0.0.0.0', port=5050, debug=False)
