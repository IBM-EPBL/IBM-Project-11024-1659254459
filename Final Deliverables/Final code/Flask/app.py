from flask import Flask,render_template,request
import cv2
from tensorflow.keras.models import load_model
import tensorflow 
import numpy as np

app = Flask(__name__,template_folder="templates")
model=load_model("Flask/analysis.h5")
#print(model)

@app.route('/',methods=['GET'])
def index():
  return render_template('home.html')

@app.route('/home',methods=['GET'])
def home():
  return render_template('home.html')

@app.route('/intro',methods=['GET'])
def intro():
  return render_template('intro.html')

@app.route('/upload',methods=['GET'])
def upload():
  return render_template('upload.html')

@app.route('/uploader',methods=['GET','POST'])
def uploader():
  if request.method == "POST":
    f = request.files['filename']
    f.save("Flask/videos/save.mp4")
  cap=cv2.VideoCapture("Flask/videos/save.mp4")
  while(True):
    _,frame = cap.read()
    frame=cv2.flip(frame,1)
    while(True):
      (grabbed,frame) = cap.read()
      if not grabbed:
        break
      output = frame.copy()
      frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
      frame = cv2.resize(frame,(64,64))
      x=np.expand_dims(frame,axis=0)
      result = np.argmax(model.predict(x),axis=1)
      index=['Cyclone','Earthquake','Flood','Wildfire']
      result = str(index[result[0]])
      #print(result)
      cv2.putText(output,"activity: {}".format(result),(10,120),cv2.FONT_HERSHEY_PLAIN,1,(0,25,255),1)
      cv2.imshow("Output",output)
    if cv2.waitKey(0) & 0xFF==ord('q'):
      break
  print("[INFO]cleaning up...")
  cap.release()
  cv2.destroyAllWindows()
  return render_template("upload.html")

@app.route('/livecam',methods=['GET','POST'])
def livecam():
    cap=cv2.VideoCapture(0)  
    while(True):
        (grabbed,frame) = cap.read()
        if not grabbed:
            break  
        output = frame.copy()
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame,(64,64))
        x=np.expand_dims(frame,axis=0)
        result = np.argmax(model.predict(x),axis=-1)
        index=['Cyclone','Earthquake','Flood','Wildfire']
        result = str(index[result[0]])
        #print(result)
        cv2.putText(output,"activity: {}".format(result),(10,120),cv2.FONT_HERSHEY_PLAIN,1,(0,25,255),1)
        cv2.imshow("Output",output)    
        if cv2.waitKey(0) & 0xFF==ord('q'):
            break
    
    print("[INFO]cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    return render_template("upload.html")


if __name__ == '__main__':
  app.run(host='0.0.0.0',port=8000,debug=False)