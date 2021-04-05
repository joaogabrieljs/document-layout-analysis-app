import subprocess
import base64
import numpy as np
from flask import Blueprint, jsonify, request
import matplotlib.pyplot as plt
from io import BytesIO
from project.predictor import make_predictions
from pdf2image import convert_from_bytes
from PyPDF2 import PdfFileReader, PdfFileWriter

dla_blueprint = Blueprint("", __name__)


@dla_blueprint.route("/analyse-image-json", methods=["POST"])
def analyse_image_json():
  # Read pdf from request and convert to PyPdf2 PdfFileReader
  pdfFileFromRequest = request.files["pdf_file"].read()
  pdfFile = PdfFileReader(BytesIO(pdfFileFromRequest))
  
  # Resize all pdf pages
  for pageNumber in range(pdfFile.getNumPages()):
    pdfFile.getPage(pageNumber).scaleTo(400, 700)

  pdfWriter = PdfFileWriter()
  pdfWriter.addPage(pdfFile.getPage(0))
  with open('pdfResized.pdf', 'wb') as f:
    pdfWriter.write(f)

  imagesFromPdf = convert_from_bytes(BytesIO(pdfFileFromRequest).read(), size=(400, 700))

  for image in imagesFromPdf:
    openCVImage = np.array(image) 
    openCVImage = openCVImage[:, :, ::-1].copy() # Convert RGB to BGR
    jsonData = make_predictions(openCVImage, True)
    boundingBoxes = jsonData.get('predictions').get('pred_boxes')

    for boundingBox in boundingBoxes:
      pointX = int(boundingBox[0])
      pointY = int(boundingBox[1])
      width = int(boundingBox[2]) - pointX
      height = int(boundingBox[3]) - pointY
      subprocess.call(['pdftotext', '-x', str(pointX), '-y', str(pointY), '-W', str(width), '-H', str(height), 'pdfResized.pdf', '-enc', 'UTF-8'])

    predictedImage = jsonData.get('img')
    predictedImage = base64.b64decode(predictedImage)

    with open('predictedImage.png', 'wb') as f:
      f.write(predictedImage)
    return 'true'