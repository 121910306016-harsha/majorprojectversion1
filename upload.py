from flask import Flask, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLD ='C:/Users/chint/OneDrive/Desktop/sample/Projectversion1/uploadedk'
ALLOWED_EXTENSIONS = {'dat', 'hea'}
UPLOAD_FOLDER = os.path.join(APP_ROOT, UPLOAD_FOLD)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        dat_file = request.files['dat_file']
        hea_file = request.files['hea_file']
        dat_file.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(dat_file.filename)))
        hea_file.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(hea_file.filename)))
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
            return redirect(url_for('uploaded_file', filename=filename))
    return ''' 
    '''
if __name__ == "__main__":
    app.run(debug=True)