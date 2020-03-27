# USAGE
# python test_network.py --model leafDiease.model --image M:\pythonD\leaf_diseaseDatabse\plantVillage\Pepper__bell___Bacterial_spot\101.JPG   
## python test_network.py --model leafDiease.model --image M:\pythonD\leaf_diseaseDatabse\plantVillage\Pepper__bell___healthy\100.JPG
## python test_network.py --model leafDiease.model --image M:\pythonD\leaf_diseaseDatabse\plantVillage\Potato___Early_blight\10.JPG
## python test_network.py --model M:\pythonD\15Days\AlexNetModel\AlexNetModel.hdf5 --image M:\pythonD\PlantVillageFullDataBase\Tomato__Tomato_mosaic_virus\101.JPG
##python test_network.py --model M:\pythonD\15Days\kegal\cnn_model.pkl --image M:\pythonD\DandI\tomato\Tomato_Late_blight\1002.JPG




# import the necessary packages
import warnings
warnings.simplefilter("ignore")
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
##import argparse
import imutils
import cv2
from os import listdir
import pickle
ht=75
wd=75
className = []

##totClass = 38

##pathh = r"M:\pythonD\Delete"
pathh = r"M:\pythonD\PlantVillageFullDataBase\Tomato"
classNames = listdir(pathh)
totClass = len(classNames)
print(classNames)
print(totClass)
##print(MainFolder)
##for classNam in MainFolder:
##    className.append(classNam)
##print(className)

# construct the argument parse and parse the arguments
##ap = argparse.ArgumentParser()
##ap.add_argument("-m", "--model", required=True,
##	help="path to trained model model")
##ap.add_argument("-i", "--image", required=True,
##	help="path to input image")
##args = vars(ap.parse_args())
##modell = "leafDisease.model"
##pathhh = "M:/pythonD/DandI/tomato/Tomato_Late_blight"
# load the image

mdl = r"M:\pythonD\15Days\7cls\LeafDisease.h5"
##im = r"M:\pythonD\Delete\Tomato___Bacterial_spot\01a3cf3f-94c1-44d5-8972-8c509d62558e___GCREC_Bact.Sp 3396.JPG"
##im = r"M:\pythonD\Delete\Potato___Early_blight\065fc68f-88c9-4fc3-b0a6-a6f5e1072eaa___RS_Early.B 7174.JPG"
##im = r"M:\pythonD\DataBase\PlantVillage\PlantVillage\validation\Potato___Early_blight\0a0744dc-8486-4fbb-a44b-4d63e6db6197___RS_Early.B 7575.JPG"
##im = r"M:\pythonD\DataBase\PlantVillage\PlantVillage\train\Tomato___Bacterial_spot\00b7e89a-e129-4576-b51f-48923888bff9___GCREC_Bact.Sp 6202.JPG"
##im = r"M:\pythonD\DataBase\PlantVillage\PlantVillage\train\Tomato___healthy\017a4026-813a-4983-887a-4052bb78c397___RS_HL 0218.JPG"
##im = r"M:\pythonD\DataBase\PlantVillage\PlantVillage\train\Tomato___Target_Spot\01e0b8b1-e713-4c6d-973b-f7636280c58a___Com.G_TgS_FL 9816.JPG"
##im = r"M:\pythonD\DataBase\PlantVillage\PlantVillage\validation\Tomato___Target_Spot\0a3b6099-c254-4bc3-8360-53a9f558a0c4___Com.G_TgS_FL 8259.JPG"
##im = r"M:\pythonD\DataBase\PlantVillage\PlantVillage\validation\Potato___Early_blight\06ac6596-8d65-46dd-a343-a2209f3480e4___RS_Early.B 6921.JPG"
##im = r"M:\pythonD\PlantVillageFullDataBase\Tomato\Tomato__Tomato_YellowLeaf__Curl_Virus\57.JPG"
##im = r"M:\pythonD\PlantVillageFullDataBase\Tomato\Tomato_Early_blight\110.JPG"
##im = r"M:\pythonD\PlantVillageFullDataBase\Tomato\Tomato_Late_blight\1001.JPG"
##im = r"M:\pythonD\PlantVillageFullDataBase\Tomato\Tomato_Leaf_Mold\101.JPG"
##im = r"M:\pythonD\PlantVillageFullDataBase\Tomato\Tomato_Septoria_leaf_spot\1005.JPG"
##im = r"M:\pythonD\PlantVillageFullDataBase\Tomato\Tomato_Spider_mites_Two_spotted_spider_mite\15.JPG"
##im = r"M:\pythonD\PlantVillageFullDataBase\Tomato\Tomato__Tomato_mosaic_virus\111.JPG"
im = r"M:\pythonD\tomTargetSPot.jpg"
image = cv2.imread(im)
orig = image.copy()

# pre-process the image for classification
try:
    image = cv2.resize(image, (ht, wd))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
except Exception as e:
    print("Error Occured : ",e)
    


# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(mdl)
##model = pickle.load(open('M:/pythonD/15Days/kegal/cnn_model.pkl', 'rb'))
# classify the input image
##classNames = ["zero", "one","two","three","four","five","six","seven","eight","nine","ozero", "oone","otwo","othree","ofour","ofive","osix","oseven","oeight","onine","tzero", "tone","ttwo","tthree","tfour","tfive","tsix","tseven","teight","tnine","tzero", "tone","ttwo","tthree","tfour","tfive","tsix","tseven"]
(zero, one,two, three,four,five,six,seven, eight,nine) = model.predict(image)[0]
prob = [zero, one,two, three,four,five,six,seven, eight,nine]

maxProb = max(prob)
maxIndex = prob.index(maxProb)
label = classNames[maxIndex]
proba = maxProb
for i in range(0,totClass):
    print(f'{classNames[i]} : {prob[i]}')
    
label = "{}: {:.2f}%".format(label, proba * 100)

# draw the label on the image
output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)

# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)
