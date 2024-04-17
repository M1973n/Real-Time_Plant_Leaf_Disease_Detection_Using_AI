import os
from flask import Flask, redirect, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd


disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')

model = CNN.CNN(39)    
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model.eval()

def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index


app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        print(file_path)
        pred = prediction(file_path)
        title = disease_info['disease_name'][pred]
        description =disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        return render_template('submit.html' , title = title , desc = description , prevent = prevent , 
                               image_url = image_url , pred = pred ,sname = supplement_name , simage = supplement_image_url , buy_link = supplement_buy_link)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image = list(supplement_info['supplement image']),
                           supplement_name = list(supplement_info['supplement name']), disease = list(disease_info['disease_name']), buy = list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run(debug=True)
    
    
    # For_the_real_time_camera.........................................................................................................
    


# disease_info_path = "C:\Users\manis\OneDrive\Desktop\Plant-Disease-Detection-main Backup (1)\Plant-Disease-Detection-main\Flask Deployed App\disease_info.csv"
disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
# supplement_info_path = "C:\Users\manis\OneDrive\Desktop\Plant-Disease-Detection-main Backup (1)\Plant-Disease-Detection-main\Flask Deployed App\supplement_info.csv"
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')

model = CNN.CNN(39)    
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model.eval()

def prediction(image_path):
    image = image_path
    # image = cv2.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    print("first output; ",output)
    
    output = output.detach().numpy()
    print("\nsecond output; ",output)
    index = np.argmax(output)
    return index


def main():
    # if request.method == 'POST':
        wCam, hCam = 600, 380
        frameR = 100 # Frame Reduction
        smoothening = 7

        pTime = 0
        plocX, plocY = 0, 0
        clocX, clocY = 0, 0
        cap = cv2.VideoCapture(0)
        cap.set(3, wCam)
        cap.set(7, hCam)
        while (True):
            
            
            check , image = cap.read()
            
            # Display
            cv2.imshow("Capturing", image)
            cv2.putText(image, "press c to take the picture or q to quit", (10, 450), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
            # Frame Rate
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(image, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
            (255, 0, 0), 3)
            key = cv2.waitKey(1)
            if key == ord('c'):
                reimage = cv2.resize(image,(224, 224))
                pred = prediction(reimage)
                print("\nprediction is returned index", pred)
                cap.release()
                cv2.destroyAllWindows()
                title = disease_info['disease_name'][pred]
                print("\nprediction is ", title)
                description =disease_info['description'][pred]
                prevent = disease_info['Possible Steps'][pred]
                image_url = disease_info['image_url'][pred]
                supplement_name = supplement_info['supplement name'][pred]
                supplement_image_url = supplement_info['supplement image'][pred]
                supplement_buy_link = supplement_info['buy link'][pred]
                print("\nDescription: ", description,"\nprevent: ",prevent,"\nimage_url: ",image_url,
                      "\nsupplement_name: ",supplement_name,"\nsupplement_image_url: ",supplement_image_url
                      ,"\nsupplement_buy_link",supplement_buy_link)

            if key == ord('c'):
                break

            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break


if __name__ == '__main__':
    main()

  
