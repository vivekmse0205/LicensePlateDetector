# Spritle License Plate Detector

Automatic License plate detection using Computer Vision technique.

## Algorithm
   * Object detection algorithm to detect location of license plate in image.
   * OCR/Object detection to extract the license plate information.

## Installation
Download the whole assignment from - https://drive.google.com/file/d/1hheVh7vX0XonjeYrzDzitTS84cqyQGcK/view?usp=sharing
First install all necessary dependencies. 
Also in detector.py need to manually set the path of locally installed tesseract at line number 77.
Tesseract installation steps:
* Ubuntu : https://github.com/tesseract-ocr/tesseract/wiki
* Windows : https://github.com/UB-Mannheim/tesseract/wiki 

```bash
pip install -r requirements.txt
```

## Usage

```python
 * Run python server.py
 * Goto localhost:5000/api/v1/getnumberplate
 * Send a post request of image file
```

## Curl Command
```CURL Command
curl --location --request POST 'localhost:5000/api/v1/getnumberplate' 
--form 'file=<image file location>'

# Example
curl --location --request POST "localhost:5000/api/v1/getnumberplate" 
-F "file=@test/1.jpg"
*The above command tested in windows 8.1 machine with all dependencies installed.
```
