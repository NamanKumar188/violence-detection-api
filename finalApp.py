from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename

from test import model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'ogg'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = model(filepath)
        if result[0] == 'not_violence':
            result[0] = 'Non Violence'
        else:
            result[0] = 'Violence' 
        x = result[2]
        result[2] = f"{x:.2f}%"
        x = result[4]
        result[4] = f"{x:.2f}%"
        
        response = {
            'prediction': str(result[0]),
            'violent_segments':str(result[1]),
            'violent_percentage':str(result[2]),
            'not_violent_segments':str(result[3]),
            'not_violent_percentage' :str(result[4])
        }
        return jsonify(response), 200
    else:
        return jsonify({'error': 'File type not allowed'}), 400



if __name__ == '__main__':
    app.run(debug=True)
