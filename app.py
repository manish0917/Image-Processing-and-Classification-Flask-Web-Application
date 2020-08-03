# import the necessary packages
import ip
import os,io
from pyimagesearch.motion_detection.singlemotiondetector import SingleMotionDetector
#import object_size
from imutils.video import VideoStream
import numpy as np
import threading
import argparse
import datetime
import imutils
import time
import cv2
import pyautogui
import sys
from flask import Flask, render_template, Response, flash, request, redirect, url_for, send_from_directory, Markup
from werkzeug.utils import secure_filename
from PIL import Image, ImageStat
import base64
#Dupicate Image
import duplicate
#PDF TO IMAGE CONVERSION
import camelot
import PyPDF2
import pdfocr
import ocrstructured
#tensorflow
import tensorflow.keras
from tensorflow.keras import backend as k
from tensorflow.keras.models import Sequential
#from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array

#google vision
from google.cloud import vision
import pandas as pd
from google.cloud.vision import types
#gva env
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"theravens-5405e084ede1.json"
client = vision.ImageAnnotatorClient()



swidth,sheight=pyautogui.size()


fswidth=swidth
# initialize the output frame and a lock used to ensure thread-safe exchanges of the output frames 
#(useful when multiple browsers/tabs are viewing the stream)
outputFrame = None
lock = threading.Lock()
count=0

# initialize a flask object
app = Flask(__name__)

# initialize the video stream and allow the camera sensor to warmup
#vs = VideoStream(usePiCamera=1).start()
vs = VideoStream(src=0).start()
time.sleep(2.0)
duplist=[]
"""UPLOADED IMAGE"""
UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
print(os.path.join(app.config['UPLOAD_FOLDER'], 'model.h5'))


@app.route('/adminpdf',methods=['GET','POST'])
def adminpdf():
    import json
    global duplist

    with open('imagedata.txt') as json_file:
        data = json.load(json_file)
        try:
            if request.method == 'POST':  
                #Duplication
                print('case1')
                         
                if request.values.get('filename') != None:
                    filename=request.values.get('filename')
                    ocrtext=pdfocr.performocr(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                    ocrstructure=ocrstructured.ocrformat(ocrtext)
            
                    try:
                        pdfFileObj = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
                        tables = camelot.read_pdf(os.path.join(app.config['UPLOAD_FOLDER'], filename),pages='1-'+str(pdfReader.getNumPages()))
                        #tables.export(os.path.join(app.config['UPLOAD_FOLDER'], 'temp.csv'), f='csv', compress=True)
                        print(pdfReader.getNumPages())
                        tablist=[]
            
                        for i in range(pdfReader.getNumPages()):
                            tablist.append(Markup(tables[i].df.to_html(classes='data')))
            
                        tab=tables[0].df
                        dftrue=tab.empty

                        ptextlist=[]
                        for i in range(pdfReader.getNumPages()):
                            ptextlist.append(pdfReader.getPage(i).extractText())

                        print(ptextlist)

                        print(tablist)
                
                        return render_template('pdfdesk.html',pdfname=filename,ocrtext=ocrtext,ocrstructure=ocrstructure,text=ptextlist,dftrue=dftrue,tables=tablist)
                    except:
                        print("Something Went wrong")

                    return render_template('pdfdesk.html',pdfname=filename,ocrtext=ocrtext,ocrstructure=ocrstructure)


                    
        except:
            print('case5')
            return render_template('pdfdesk.html',vinfo="Something is wrong!!")

        return render_template('adminpdf.html',data=data)



def validimg(img):
    import json

    images=img
    #ip getinfo
    ipdata=ip.getDetails()
                
    #print(ipdata)


    data = {}
    data['image'] = img
    data['ip'] = ipdata['ip']
    data['city'] = ipdata['city']
    data['region'] = ipdata['region']
    data['country'] = ipdata['country']
    data['loc'] = ipdata['loc']
    data['postal'] = ipdata['postal']
    
    

    with open('imagedata.txt', 'w') as outfile:
        json.dump(data, outfile)

@app.route('/admin',methods=['GET','POST'])
def admin():
    import json
    global duplist

    with open('imagedata.txt') as json_file:
        data = json.load(json_file)
        try:
            if request.method == 'POST':  
                #Duplication
                print('case1')

                         
                if request.values.get('filename') != None:
                    filename=request.values.get('filename')
                    modelinfo=get_model(filename)
                    print('case1')
                    #opencv image object to perform IMG-Processing
                    cvimg=cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))

                    try:
                        print('case2')
                        # construct an iamge instance
                        with io.open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'rb') as image_file:
                            content = image_file.read()
                        image = vision.types.Image(content=content)
                        print('caseB')
                        '''/// OCR /// '''
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
            

                        # Performs label detection on the image file
                        response = client.label_detection(image=image)
                        labels = response.label_annotations

                        '''
                        print('Labels:')
                        for label in labels:
                        print(label.description)
                        '''
                        print('case C')
                        #Multiple Object Detection
                        objects = client.object_localization(image=image).localized_object_annotations
                        print('case A')
            
                        print('Number of objects found: {}'.format(len(objects)))
                        for object_ in objects:
                            print('{} (confidence: {})'.format(object_.name, object_.score))
                            #object_.bounding_poly.normalized_vertices
                            # Assigning vertices to polygon
                            #cv2.imwrite('static/img/'+filename, cvimg) 
                            #print('static/img/'+filename)
                        print('case3')
                        return render_template('admindesk.html',imgname=filename,ocr=df['description'][0],labels=labels, objects=objects,modelinfo=modelinfo)
                    except:
                        print('case4')
                        return render_template('admindesk.html',imgname=filename,modelinfo=modelinfo)
        except:
            print('case5')
            return render_template('admindesk.html',vinfo="Something is wrong!!")

        return render_template('admin.html',data=data)
    






#tensorflow cnn model
def get_model(imgfile):
    global model
    info=None
    model=tensorflow.keras.models.load_model(os.path.join(app.config['UPLOAD_FOLDER'], 'my_model'))
    print('model loaded')
    img=image.load_img(os.path.join(app.config['UPLOAD_FOLDER'], imgfile),target_size=(200,200))
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    images=np.vstack([x])
    val=model.predict([images])
    if val==0:
        info='valid'
    else:
        info='not valid'
        
        
    return info






def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/', methods=['GET', 'POST'])
def index():
    global duplist
    try:
        if request.method == 'POST':  
            #Duplication
            print('case1')

            if request.values.get('duplicate') != None:
                
                #print(request.values.get('duplicate'))
                duplicates=request.values.get('duplicate')
                fim=os.path.join(app.config['UPLOAD_FOLDER'], duplicates)
                duplist=duplicate.findDuplicate(fim)
                print(duplist)
                return render_template('indexAI.html',vinfo="Duplicates found!!",duplist=duplist)
                
                
            if request.values.get('delete') != None:
                dfile=request.values.get('delete')
                delf=os.path.join(app.config['UPLOAD_FOLDER'], dfile)
                os.remove(delf)
                duplist.remove(dfile)
                return render_template('indexAI.html',vinfo="Duplicates found!!",duplist=duplist)
            
            #object size redirect
            # check if the post request has the file part
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            print('case2')
            file = request.files['file']
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            print('case3')
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                

                try:
                    with Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename),'r') as im:
                        pix_val = list(im.getdata())  # get pixel value in RGB format
                    print('case N')
                    a= [x for sets in pix_val for x in sets] #Convert list of tuples into one list 

                    myRoundedList =  [round(x,-1) for x in a]  #Round integers to nearest 10

                    b=[tuple(myRoundedList[i:i+3]) for i in range(0, len(myRoundedList), 3)]  #Group list to a tuple of 3 integers 

                    list_of_pixels = list(b)

                    print('case O')
                    im2 = Image.new(im.mode, im.size) #Create a new image 
                    print('case OO')

                    im2.putdata(list_of_pixels) #put image data into the new image 
                    print('case OOO')

                    im2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))  #save the file 
                    print('case p')
                except:
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                validimg(filename)
                print('case Q')
                modelinfo=get_model(filename)
                print('case4')
                #opencv image object to perform IMG-Processing
                cvimg=cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))

                try:
                    print('case5')
                    # construct an iamge instance
                    with io.open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'rb') as image_file:
                        content = image_file.read()
                    image = vision.types.Image(content=content)

                    '''/// OCR /// '''
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
            

                    # Performs label detection on the image file
                    response = client.label_detection(image=image)
                    labels = response.label_annotations

                    '''
                    print('Labels:')
                    for label in labels:
                    print(label.description)
                    '''

                    #Multiple Object Detection
                    objects = client.object_localization(image=image).localized_object_annotations

            
                    print('Number of objects found: {}'.format(len(objects)))
                    for object_ in objects:
                        print('{} (confidence: {})'.format(object_.name, object_.score))
                        #object_.bounding_poly.normalized_vertices
                        # Assigning vertices to polygon
                        #cv2.imwrite('static/img/'+filename, cvimg) 
                        #print('static/img/'+filename)
                    print('case6')
                    

                    return render_template('index.html',imgname=filename,ocr=df['description'][0],labels=labels, objects=objects,modelinfo=modelinfo)
                except:
                    print('case7')
                    return render_template('index.html',imgname=filename,modelinfo=modelinfo)
    except:
        print('case8')
        return render_template('index.html',vinfo="Something is wrong!!")


    """Video streaming home page."""
    return render_template('index.html')


@app.route('/Pdf', methods=['GET', 'POST'])
def pdf():
      
    if request.method == 'POST':        

        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
            # if user does not select file, browser also
            # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            validimg(filename)

            return render_template('pdf.html',pdfname=filename)


#    except:
 #       return render_template('pdf.html',vinfo="Something is wrong!!")

    return render_template('pdf.html')


'''
//***********/// Streaming ///**********///
'''

#WebCam
@app.route('/camera')
def camera():
    """Video streaming page."""
    return render_template('cam.html')

""" motion dtector and video recorder """
def detect_motion(frameCount=32):
    
    # grab global references to the video stream, output frame, and lock variables
    global vs, outputFrame, lock, count
    
    # initialize the motion detector and the total number of frames
    # read thus far
    md = SingleMotionDetector(accumWeight=0.1)
    total = 0

    # loop over frames from the video stream
    while True:
        # read the next frame from the video stream, resize it,
        # convert the frame to grayscale, and blur it
        frame = vs.read()
        
        #frame = imutils.resize(frame, width=720)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        

        # grab the current timestamp and draw it on the frame
        """
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        """


        # if the total number of frames has reached a sufficient number 
        # to construct a reasonable background model, then continue to process the frame
        if total > frameCount:
            
            # detect motion in the image
            motion = md.detect(gray)
            
            # check to see if motion was found in the frame
            """
            if motion is not None:
                # unpack the tuple and draw the box surrounding the
                # "motion area" on the output frame
                (thresh, (minX, minY, maxX, maxY)) = motion
                cv2.rectangle(frame, (minX, minY), (maxX, maxY),
                    (255, 255, 255), 1)
            """

        # update the background model and increment the total number
        # of frames read thus far
        md.update(gray)
        total += 1
        count += 1
        
        # acquire the lock, set the output frame, and release the
        # lock
        with lock:
            outputFrame = frame.copy()



def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock
    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue
            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            # ensure the frame was successfully encoded
            if not flag:
                continue
        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')
  




@app.route("/cam")
def cam():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")



""" Saving GIF and 3D Process """

@app.route('/gif')
def gif():
    global vs, count
    # Initialize variables
    #video = cv2.VideoCapture(0) 

    t_end = time.time() + 1*10
    t = 10
    t_time = time.time()
    # initialize the FourCC, video writer, dimensions of the frame, and
    # zeros array
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = None
    (h, w) = (None, None)
    zeros = None

    #rec = cv2.VideoWriter("Vid%d.avi" % count,fourcc,10,(400,400))
    
    running = True
    now_t=int((time.strftime("%S",time.localtime())))
    while time.time() < t_end:
        # grab the frame from the video stream and resize it to have a
        # maximum width of 300 pixels
        frame = vs.read()

        #frame = imutils.resize(frame, width=300)
        # check if the writer is None
        if writer is None:
            # store the image dimensions, initialize the video writer,
            # and construct the zeros array
            (h, w) = frame.shape[:2]
            writer = cv2.VideoWriter("Vid%d.avi" % count, fourcc,50,
                (w,h), True)
            zeros = np.zeros((h, w), dtype="uint8")
        # break the image into its RGB components
        (B, G, R) = cv2.split(frame)
        R = cv2.merge([zeros, zeros, R])
        G = cv2.merge([zeros, G, zeros])
        B = cv2.merge([B, zeros, zeros])

        # construct the final output frame
        output = np.zeros((h,w, 3), dtype="uint8")
        output[0:h, 0:w] = frame
        
        # write the output frame to file
        writer.write(output)

        #timer
        

        if now_t<int(time.strftime("%S",time.localtime())):
            t=t-1

        now_t=int(time.strftime("%S",time.localtime()))+1


        cv2.putText(frame,str(t),(10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        #time Loop
        now_t=int(time.strftime("%S",time.localtime()))
        #print(t_time)



    # When everything done, release  
    # the video capture and video  
    # write objects 
    writer.release() 

    return render_template("index.html")
    
            



#Image Capturing
@app.route('/capture')
def capture():
    global vs, count
    image = vs.read()
    cv2.imwrite("Img%d.jpg" % count, image)
    return render_template('index.html')
            

""" IMG DIR """
@app.route('/aigallary', methods=['GET', 'POST'])
def aigallary():

    
    try:
       
        if request.method == 'POST':
            """
            if request.values.get('width') !=None and request.values.get('w-unit') !=None:
                print(request.values.get('width'))
                object_size.detect_size(os.path.join(app.config['UPLOAD_FOLDER'], request.values.get('w-image')))
                return render_template('index.html',vinfo=request.values.get('width'))
            """
            if request.values.get('aifile') !="":
                aifile=request.values.get('aifile')
                #print(aifile)
                # construct an iamge instance
                with io.open(os.path.join(app.config['UPLOAD_FOLDER'], aifile), 'rb') as image_file:
                    content = image_file.read()
                image = vision.types.Image(content=content)

                '''/// OCR /// '''
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
            

                # Performs label detection on the image file
                response = client.label_detection(image=image)
                labels = response.label_annotations

                '''
                print('Labels:')
                for label in labels:
                    print(label.description)
                '''

                #Multiple Object Detection
                objects = client.object_localization(image=image).localized_object_annotations
                '''
                print('Number of objects found: {}'.format(len(objects)))
                for object_ in objects:
                    print('{} (confidence: {})'.format(object_.name, object_.score))
                    print('Normalized bounding polygon vertices: ')
                    for vertex in object_.bounding_poly.normalized_vertices:
                    print(' - ({}, {})'.format(vertex.x, vertex.y))
                '''
                return render_template('index.html',imgname=aifile,ocr=df['description'][0],labels=labels, objects=objects)
            


    except:
        jpg=[]
        for file in os.listdir("static/"):
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg") or file.endswith(".gif") or file.endswith(".JPG") or file.endswith(".PNG") or file.endswith(".JPEG") or file.endswith(".GIF"):
                jpg.append( file)

        return render_template('gallary.html',images=jpg,vinfo="Something Went wrong!!")

    #Fatching Images
    jpg=[]
    for file in os.listdir("static/"):
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg") or file.endswith(".gif") or file.endswith(".JPG") or file.endswith(".PNG") or file.endswith(".JPEG") or file.endswith(".GIF"):
            jpg.append( file)

    
    return render_template('gallary.html',images=jpg)

""" PDF DIR """
@app.route('/docs', methods=['GET', 'POST'])
def docs():
    try:
        
        if request.method == 'POST':
            if request.values.get('aifile') !="":
                aifile=request.values.get('aifile')
                #print(aifile)
                pdfFileObj = open(os.path.join(app.config['UPLOAD_FOLDER'], aifile), 'rb')
                # pdf reader object
                pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
            
                tables = camelot.read_pdf(os.path.join(app.config['UPLOAD_FOLDER'], aifile),pages='1-'+str(pdfReader.getNumPages()))
                #tables.export(os.path.join(app.config['UPLOAD_FOLDER'], 'temp.csv'), f='csv', compress=True)
                print(pdfReader.getNumPages())
                tablist=[]
                try:
                    for i in range(pdfReader.getNumPages()):
                        tablist.append(Markup(tables[i].df.to_html(classes='data')))
                except:
                    tablist.append("No Table Detected")
            
                tab=tables[0].df
                dftrue=tab.empty

                ptextlist=[]
                for i in range(pdfReader.getNumPages()):
                    ptextlist.append(pdfReader.getPage(i).extractText())
                return render_template('pdf.html',pdfname=aifile,dftrue=dftrue,tables=tablist,titles=tab.columns.values,text=ptextlist)
        
    except:
        pdf=[]
        for file in os.listdir("static/"):
            if file.endswith(".pdf") or file.endswith(".PDF"):
                pdf.append(file)
        return render_template('docs.html',pdfs=pdf,vinfo="Something Went wrong!!!")
    pdf=[]
    for file in os.listdir("static/"):
        if file.endswith(".pdf") or file.endswith(".PDF"):
            pdf.append(file)
    
    return render_template('docs.html',pdfs=pdf)



#if __name__ == '__main__':
#    app.run(host='0.0.0.0', debug=True, threaded=True)
# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
        help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
        help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32,
        help="# of frames used to construct the background model")
    args = vars(ap.parse_args())
    # start a thread that will perform motion detection
    t = threading.Thread(target=detect_motion, args=(
        args["frame_count"],))
    t.daemon = True
    t.start()
    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
        threaded=True, use_reloader=False)
# release the video stream pointer
vs.stop()