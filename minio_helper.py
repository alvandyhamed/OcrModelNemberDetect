import os
import io
import zipfile
import json
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

MINIO_ENDPOINT = os.environ.get('MINIO_ENDPOINT', 'http://minio:9000')
MINIO_ACCESS_KEY = os.environ.get('MINIO_ACCESS_KEY', 'minioadmin')
MINIO_SECRET_KEY = os.environ.get('MINIO_SECRET_KEY', '1MP6rLEyoGKVAoUaNJoMDW2F')
MINIO_BUCKET = os.environ.get('MINIO_BUCKET', 'chart-ocr-datasets')

def get_s3_client():
    session = boto3.Session(
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
    )
    return session.client(
        's3',
        endpoint_url=MINIO_ENDPOINT,
        config=Config(
            signature_version='s3v4',
            s3={'addressing_style': 'path'}
        ),
        region_name='us-east-1'
    )

def ensure_bucket_exists():
    try:
        s3 = get_s3_client()
        buckets = [b['Name'] for b in s3.list_buckets().get('Buckets', [])]
        if MINIO_BUCKET not in buckets:
            s3.create_bucket(Bucket=MINIO_BUCKET)
    except Exception as e:
        print(f"MinIO bucket check warning: {e}")

def upload_file_bytes(object_key, file_bytes, content_type='application/octet-stream'):
    ensure_bucket_exists()
    s3 = get_s3_client()
    s3.put_object(
        Bucket=MINIO_BUCKET,
        Key=object_key,
        Body=file_bytes,
        ContentType=content_type
    )
    return f"{MINIO_ENDPOINT}/{MINIO_BUCKET}/{object_key}"

def extract_zip_file_to_minio(zip_source, dataset_prefix):
    ensure_bucket_exists()
    s3 = get_s3_client()
    extracted_files = []
    
    with zipfile.ZipFile(zip_source, 'r') as z:
        for file_info in z.infolist():
            if file_info.is_dir():
                continue
            filename = file_info.filename
            if filename.startswith('__MACOSX') or filename.endswith('.DS_Store') or os.path.basename(filename).startswith('.'):
                continue
                
            clean_rel_path = filename.lstrip('/')
            object_key = f"raw_uploads/{dataset_prefix}/{clean_rel_path}"
            
            with z.open(file_info) as f:
                file_data = f.read()
            
            ext = os.path.splitext(filename)[1].lower()
            content_type = 'image/png' if ext == '.png' else ('image/jpeg' if ext in ['.jpg', '.jpeg'] else 'application/octet-stream')
            
            s3.put_object(
                Bucket=MINIO_BUCKET,
                Key=object_key,
                Body=file_data,
                ContentType=content_type
            )
            extracted_files.append({
                "filename": os.path.basename(filename),
                "key": object_key,
                "size": len(file_data)
            })
            
    return extracted_files

def list_raw_datasets():
    ensure_bucket_exists()
    s3 = get_s3_client()
    paginator = s3.get_paginator('list_objects_v2')
    datasets = set()
    
    for page in paginator.paginate(Bucket=MINIO_BUCKET, Prefix='raw_uploads/'):
        for obj in page.get('Contents', []):
            key = obj['Key']
            parts = key.split('/')
            if len(parts) > 2:
                datasets.add(parts[1])
                
    return sorted(list(datasets))

def list_dataset_images(dataset_name):
    ensure_bucket_exists()
    s3 = get_s3_client()
    paginator = s3.get_paginator('list_objects_v2')
    image_keys = []
    
    prefix = f"raw_uploads/{dataset_name}/"
    for page in paginator.paginate(Bucket=MINIO_BUCKET, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            ext = os.path.splitext(key)[1].lower()
            if ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                image_keys.append(key)
                
    return sorted(image_keys)

def get_object_bytes(object_key):
    s3 = get_s3_client()
    response = s3.get_object(Bucket=MINIO_BUCKET, Key=object_key)
    return response['Body'].read()

def save_json_to_minio(object_key, data):
    ensure_bucket_exists()
    s3 = get_s3_client()
    json_bytes = json.dumps(data, indent=2, ensure_ascii=False).encode('utf-8')
    s3.put_object(
        Bucket=MINIO_BUCKET,
        Key=object_key,
        Body=json_bytes,
        ContentType='application/json'
    )

def read_json_from_minio(object_key):
    s3 = get_s3_client()
    try:
        response = s3.get_object(Bucket=MINIO_BUCKET, Key=object_key)
        return json.loads(response['Body'].read().decode('utf-8'))
    except Exception:
        return None
