from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
import cv2
import numpy as np
import base64
import zipfile
import importlib
import glob
from io import BytesIO
import shutil
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALGORITHM_FOLDER = 'algorithms'
OUTPUT_FOLDER = 'results'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ALGORITHM_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALGORITHM_FOLDER'] = ALGORITHM_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

def load_algorithm(algorithm_name):
    try:
        module = importlib.import_module(f'algorithms.{algorithm_name}')
        return getattr(module, algorithm_name)
    except Exception as e:
        print(f'Error loading algorithm: {e}')
        return None

def process_images(visible_paths, infrared_paths, algorithm, output_name):
    os.makedirs(os.path.join(app.config['OUTPUT_FOLDER'], output_name), exist_ok=True)
    algorithm_func = load_algorithm(algorithm)
    results = []
    
    for v_path, i_path in zip(visible_paths, infrared_paths):
        try:
            img = algorithm_func(v_path, i_path)
            filename = os.path.basename(v_path)
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_name, filename)
            cv2.imwrite(output_path, img)
            results.append(output_path)
        except Exception as e:
            print(f'Error processing {filename}: {e}')
    
    return results

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/algorithms')
def get_algorithms():
    algorithms = [os.path.splitext(f)[0] for f in os.listdir(ALGORITHM_FOLDER) 
                 if f.endswith('.py') and not f.startswith('__')]
    return jsonify(algorithms)

@app.route('/upload', methods=['POST'])
def handle_upload():
    try:
        output_name = request.form['outputName']
        algorithm = request.form['algorithm']
        
        visible_files = request.files.getlist('visible')
        infrared_files = request.files.getlist('infrared')
        
        # 清空上传目录（可选）
        shutil.rmtree(app.config['UPLOAD_FOLDER'])
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        visible_paths = []
        infrared_paths = []
        
        # 修改文件保存逻辑（确保只保存文件名）
        for f in visible_files:
            # 仅保留文件名（去除路径）
            filename = secure_filename(os.path.basename(f.filename)) 
            path = os.path.normpath(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            f.save(path)
            visible_paths.append(path)
            
        for f in infrared_files:
            filename = secure_filename(os.path.basename(f.filename))
            path = os.path.normpath(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            f.save(path)
            infrared_paths.append(path)
        
        # 添加路径验证
        print("可见光文件路径:", visible_paths)
        print("红外文件路径:", infrared_paths)
        if not all(os.path.exists(p) for p in visible_paths + infrared_paths):
            raise FileNotFoundError("部分文件未正确保存")
        
        # 处理图像
        results = process_images(visible_paths, infrared_paths, algorithm, output_name)
        
        # 创建压缩包
        zip_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{output_name}.zip')
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for res in results:
                zipf.write(res, os.path.basename(res))
        
        return jsonify({
            'status': 'success',
            'downloadUrl': f'/download/{output_name}'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/download/<filename>')
def download_results(filename):
    zip_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{filename}.zip')
    return send_file(zip_path, as_attachment=True)

@app.route('/preview/<path:filename>')
def preview_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)