import fitz
import os, time
import shutil


import os, io
from google.cloud import vision
import pandas as pd
from google.cloud.vision import types

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"theravens-5405e084ede1.json"

client = vision.ImageAnnotatorClient()

def performocr():
    pdf_files ="static/4th_and_6th_Sem_Third_Mid_Term.pdf"
    doc = fitz.open(pdf_files)   #open/read pdf file
    pages = doc.pageCount   #count pages in pdf file
    for i in range(pages):
        page = doc.loadPage(i)  #load pages one by one
        pix = page.getPixmap()  #get image of page
        output = os.path.join("static/images", "Page "+str(i+1)+".jpg")    #output path for image
        pix.writePNG(output)    #write pdf page as image file
    os.mkdir("static/images")
    dir1 = "static/images"
    image_file=[_ for _ in os.listdir(dir1) if _.endswith('jpg')]
    #print(image_file)
    print("Print the Pdf content")

    for file_name in image_file:
        image_path = r'static/images'

        with io.open(os.path.join(image_path,file_name), 'rb') as image_file:
            content = image_file.read()

        # construct an iamge instance
        image = vision.types.Image(content=content)

        # annotate Image Response
        response = client.text_detection(image=image)  # returns TextAnnotation
        df = pd.DataFrame(columns=['locale', 'description'])

        texts = response.text_annotations
        for text in texts:
            df = df.append(
                dict(
                    locale=text.locale,
                    description=text.description
                ),
                ignore_index=True
            )
        print(df['description'][0])



    shutil.rmtree(dir1)
performocr()
        


