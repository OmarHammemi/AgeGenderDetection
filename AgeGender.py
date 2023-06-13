import cv2
import pandas as pd
import os

folder_path = 'C:/Users/T14s/Desktop/photo influencers'  # Replace with the actual path to your folder
file_names = os.listdir(folder_path)
ages,genders,fileNames=[],[],[]
for file_name in file_names:
    try:
        print(file_name)
        image = cv2.imread('C:/Users/T14s/Desktop/photo influencers/'+file_name)
        image = cv2.resize(image, (720, 640))
        face1 = "C:/Users/T14s/Desktop/GenderAgeDetection/pretrained_moels/opencv_face_detector.pbtxt"
        face2 = "C:/Users/T14s/Desktop/GenderAgeDetection/pretrained_moels/opencv_face_detector_uint8.pb"
        age1 = "C:/Users/T14s/Desktop/GenderAgeDetection/pretrained_moels/age_deploy.prototxt"
        age2 = "C:/Users/T14s/Desktop/GenderAgeDetection/pretrained_moels/age_net.caffemodel"
        gen1 = "C:/Users/T14s/Desktop/GenderAgeDetection/pretrained_moels/gender_deploy.prototxt"
        gen2 = "C:/Users/T14s/Desktop/GenderAgeDetection/pretrained_moels/gender_net.caffemodel"
        
        MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        face = cv2.dnn.readNet(face2, face1)
        age = cv2.dnn.readNet(age2, age1)
        gen = cv2.dnn.readNet(gen2, gen1)
        la = ['(0-2)', '(4-7)', '(8-14)', '(15-22)',
            '(23-35)', '(36-45)', '(46-58)', '(60+)']
        lg = ['Male', 'Female']
        # Copy image
        fr_cv = image.copy()
        # Face detection
        fr_h = fr_cv.shape[0]
        fr_w = fr_cv.shape[1]
        blob = cv2.dnn.blobFromImage(fr_cv, 1.0, (300, 300),
                                    [104, 117, 123], True, False)
        face.setInput(blob)
        detections = face.forward()

        faceBoxes = []
        for i in range(detections.shape[2]):
            #Bounding box creation if confidence > 0.7
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                x1 = int(detections[0, 0, i, 3]*fr_w)
                y1 = int(detections[0, 0, i, 4]*fr_h)
                x2 = int(detections[0, 0, i, 5]*fr_w)
                y2 = int(detections[0, 0, i, 6]*fr_h)
                faceBoxes.append([x1, y1, x2, y2])
                cv2.rectangle(fr_cv, (x1, y1), (x2, y2),
                            (0, 255, 0), int(round(fr_h/150)), 8)
        faceBoxes
        if not faceBoxes:
            print("No face detected")
        
        for faceBox in faceBoxes:
            face = fr_cv[max(0, faceBox[1]-15):
                        min(faceBox[3]+15, fr_cv.shape[0]-1),
                        max(0, faceBox[0]-15):min(faceBox[2]+15,
                                    fr_cv.shape[1]-1)]
            blob = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            gen.setInput(blob)
            genderPreds = gen.forward()
            gender = lg[genderPreds[0].argmax()]

            age.setInput(blob)
            agePreds = age.forward()
            age = la[agePreds[0].argmax()]
            print(age)
            ages.append(age)
            genders.append(gender)
            fileNames.append(file_name)
    except:
        pass
print('done')
print(len(ages),len(genders),len(fileNames))
data=pd.DataFrame({'photo':fileNames,'gender':genders,'age':ages})
data.to_csv('C:/Users/T14s/Desktop/Tawa/Scripts/AgeGendera.csv')