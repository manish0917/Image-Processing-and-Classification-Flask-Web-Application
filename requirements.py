import subprocess

flask = ["pip", "install", "flask"]
imutils = ["pip", "install", "imutils"]
pyautogui = ["pip", "install", "pyautogui"]
PIL = ["pip", "install", "PIL"]
werkzeug = ["pip", "install", "werkzeug"]
camelot = ["pip", "install", "camelot"]
PyPDF2 = ["pip", "install", "PyPDF2"]
tensorflow = ["pip", "install", "tensorflow"]
google_cloud = ["pip", "install", "google.cloud"]
google_cloud_vision = ["pip", "install", "google.cloud.vision"]



### install flask package
out = subprocess.check_output(flask, shell = True)
### install imutils package
out = subprocess.check_output(imutils, shell = True)
### install pyautogui package
out = subprocess.check_output(pyautogui, shell = True)
### install PIL package
out = subprocess.check_output(PIL, shell = True)
### install werkzeug package
out = subprocess.check_output(werkzeug, shell = True)
### install camelot package
out = subprocess.check_output(camelot, shell = True)
### install PyPDF2 package
out = subprocess.check_output(PyPDF2, shell = True)
### install tensorflow package
out = subprocess.check_output(tensorflow, shell = True)
### install google_cloud package
out = subprocess.check_output(google_cloud, shell = True)
### install google_cloud_vision package
out = subprocess.check_output(google_cloud_vision, shell = True)


