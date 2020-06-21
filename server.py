import os
import cv2
import numpy as np
from flask import Flask, jsonify, request
from detector import LicensePlateDetector

app = Flask(__name__)

ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'gif']


@app.route('/')
def hello_world():
    return 'Hello, World!'


def is_image_file(file_name: str):
    """
    Description : Checks if the image is valid or not
    :param file_name: Nme of the file
    :return: bool
    """
    file_name = file_name.lower()
    if file_name.split('.')[-1] in ALLOWED_EXTENSIONS:
        return True
    return False


@app.route('/api/v1/getnumberplate', methods=['POST'])
def get_prediction():
    """
    Description : Detects the license plate from image and return number
    :return: Returns a license plate number from given image or return null
    """
    output_path = 'predictions'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    file = request.files.get('file', '')
    image = file.read()
    if image and is_image_file(file.filename):
        np_img = np.fromstring(image, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        temp_image = img.copy()
        img_h, img_w = img.shape[0], img.shape[1]
        output_detection = LicensePlateDetector.process_image(temp_image, img_w, img_h)
        if output_detection:
            output_detected_image, number_plate_list = LicensePlateDetector.draw_bounding_box(temp_image,
                                                                                              output_detection)
            image_save_path = os.path.join(output_path, file.filename)
            cv2.imwrite(image_save_path, output_detected_image)
            output_text_list = LicensePlateDetector.extract_text(number_plate_list)
            return jsonify({'license_plate': output_text_list})
        return jsonify({'message': 'Sorry, no license plate detected or image is of poor quality or no Indian vehicle found'})
    else:
        return jsonify({'message':'not an image file'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
