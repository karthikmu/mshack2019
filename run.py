
from sklearn.externals import joblib
import os
from keras.preprocessing import image
import cv2
from keras.preprocessing.image import img_to_array
import numpy as np
import sys

def convert_image_to_array(image_dir):
    try:

        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, tuple((256, 256)))
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None


def main():

    model = joblib.load('cnn_model.pkl')

    model.compile(loss="binary_crossentropy", optimizer='adam',metrics=["accuracy"])

    # imgPath = os.getcwd() + '/49bb8caf-4c31-4bae-b345-1b03f5fd8e34.JPG'

    imgPath = sys.argv[1]
    inp = convert_image_to_array(imgPath)
    inp = inp[np.newaxis,:,:,:]

    pd = model.predict(inp)

    label_name = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
     'Potato___Early_blight' ,'Potato___Late_blight', 'Potato___healthy',
     'Tomato_Bacterial_spot', 'Tomato_Early_blight' ,'Tomato_Late_blight',
     'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
     'Tomato_Spider_mites_Two_spotted_spider_mite' ,'Tomato__Target_Spot',
     'Tomato__Tomato_YellowLeaf__Curl_Virus' ,'Tomato__Tomato_mosaic_virus',
     'Tomato_healthy']

    cls = np.where(pd[0,:]==1)[0][0]

    print(label_name[cls])

    return (label_name[cls])

if __name__ == "__main__":
    main()