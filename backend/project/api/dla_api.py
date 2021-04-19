from subprocess import check_output
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
  #pdfWriter.addAttachment('pdfResized.pdf', pdfFile)
  pdfWriter.addPage(pdfFile.getPage(0))
  with open('pdfResized.pdf', 'wb') as f:
    pdfWriter.write(f)

  imagesFromPdf = convert_from_bytes(BytesIO(pdfFileFromRequest).read(), size=(400, 700))

  for image in imagesFromPdf:
    openCVImage = np.array(image) 
    openCVImage = openCVImage[:, :, ::-1].copy() # Convert RGB to BGR
    jsonData = make_predictions(openCVImage, True)
    predictions = jsonData.get('predictions')
    boundingBoxes = sort_bounding_boxes(predictions.get('pred_boxes'))

    jsonParagraphs = {}
    for index, boundingBox in enumerate(boundingBoxes):
      paragraph = {}
      pointX = int(boundingBox[0])
      pointY = int(boundingBox[1])
      width = int(boundingBox[2]) - pointX
      height = int(boundingBox[3]) - pointY
      paragraph['text'] = check_output(['pdftotext', '-x', str(pointX), '-y', str(pointY), '-W', str(width), '-H', str(height), 'pdfResized.pdf', '-enc', 'UTF-8', '-'])
      paragraph['score'] = predictions.get('scores')[index]
      paragraph['type'] = get_prediction_type(predictions.get('pred_classes')[index])
      jsonParagraphs[index] = paragraph
  return jsonify(jsonParagraphs)

def sort_bounding_boxes(boundingBoxes):
  npArrayBoundingBoxes = np.array(boundingBoxes)   
  index = np.lexsort((npArrayBoundingBoxes[:,0], npArrayBoundingBoxes[:,1])) 
  sortedBoundingBoxes = npArrayBoundingBoxes[index]
  return sortedBoundingBoxes

def get_prediction_type(type):
  switcher = {
    0: 'text',
    1: 'title',
    2: 'figure',
    3: 'table'
  }
  return switcher.get(type, 'nothing')

