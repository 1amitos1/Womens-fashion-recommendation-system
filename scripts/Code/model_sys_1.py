import os

from keras import Model
from keras.models import model_from_json
import gc
import cv2
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np


class Recommendation_Model:
    def __init__(self):
        self.model_fashioNet= self.fashionnet_model_loading()
        self.pre_dict = {0: "Blouses_Shirts", 1: "Dresses", 2: "Shorts", 3: "Skirts"}

    def fashionnet_model_loading(self):
        # load json and create model
        json_file = open(r"E:\FINAL_PROJECT_DATA\FashioNet\Models\model.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model_fashioNet = model_from_json(loaded_model_json)
        # load weights into new model
        self.model_fashioNet.load_weights(r"E:\FINAL_PROJECT_DATA\FashioNet\Models\save_weights.h5")
        print("Loaded FashioNet model from disk")
        #self.image_embedding_model = VGG16(weights='imagenet', include_top=False)
        return self.model_fashioNet

    def read_image(self,img_path, H=224, W=224):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (W, H))  # you can resize to  (128,128) or (256,256)
        return img

    def prediction_processes(self,pred):
        pred = pred[0]
        self.pre_dict={0:"Blouses_Shirts",1:"Dresses",2:"Shorts",3:"Skirts"}
        categorical = """
                    Blouses_Shirts :  0
                    Dresses :  1
                    Shorts :  2
                    Skirts :  3
                """

        #print(f"pred len={len(pred)}\n")
        #for i in range(len(pred)):
            # print(f"pred{i}={round(pred[i], 5)}\n")
            #print(f"pred{i}={pred[i]}\n")

        # Get the maximum element from a Numpy array
        maxElement = np.amax(pred)
        #print('Max element from Numpy Array : ', maxElement)
        ## Get the indices of maximum element in numpy array
        result = np.where(pred == np.amax(pred))
        #print('Returned tuple of arrays :', result)
        #print('List of Indices of maximum element :', result[0])
        key = result[0][0]
        #print(f"Res = {self.pre_dict.get(key)}")
        return key

    def prediction(self,test_img_path):
        try:
            H, W = 224, 224
            img = self.read_image(test_img_path, H, W)
            img = img.reshape((-1, 224, 224, 3))
            pred = self.model_fashioNet.predict(img, batch_size=128, verbose=1)
            key = self.prediction_processes(pred)
        except TypeError:
            key = None

        return key

    def get_user_prediction(self,user_data_path=None):
        """
        :param user_data_path: path to user pic
        :return: dict with classification items for each category
        """
        #user_data_path= r"E:\FINAL_PROJECT_DATA\FashioNet\TEST_PICTURE"
        #print(type(user_data_path))


        Blouses_Shirts_index=0
        Dresses_index=0
        Shorts_index=0
        Skirts_index=0
        for p in user_data_path:
            #get prediction on curent picture
            key = self.prediction(p)
            #print(key)
            if(key == None):
                continue
            if(key == 0):
                Blouses_Shirts_index = Blouses_Shirts_index + 1
            elif(key == 1):
                Dresses_index = Dresses_index +1
            elif(key == 2):
                Shorts_index = Shorts_index+1
            elif (key == 3):
                Skirts_index = Skirts_index +1
            else:
                print("[-][-]ERROR\nExit..")

        #update vote

        vote_count_dict = {"Blouses_Shirts": 0, "Dresses": 0, "Shorts": 0, "Skirts": 0}
        vote_count_dict.update({"Blouses_Shirts":Blouses_Shirts_index})
        vote_count_dict.update({"Dresses": Dresses_index})
        vote_count_dict.update({ "Shorts": Shorts_index})
        vote_count_dict.update({"Skirts": Skirts_index})
        v_list=[]
        v_list.append(Blouses_Shirts_index)
        v_list.append(Dresses_index)
        v_list.append(Shorts_index)
        v_list.append(Skirts_index)
        maxElement = np.amax(v_list)
        #print('First Max element from Numpy Array : ', maxElement)
        ## Get the indices of maximum element in numpy array
        result = np.where(v_list == np.amax(v_list))
        print('Returned tuple of arrays :', result)
        print('List of  First Indices of maximum element :', result[0])
        try:
            index_max_1 = int(result[0])
            v_list[int(result[0])] = 0
        except TypeError:
            index_max_1 = int(result[0][0])
            v_list[int(result[0][0])] = 0

        maxElement = np.amax(v_list)
        #print('Second Max element from Numpy Array : ', maxElement)
        ## Get the indices of maximum element in numpy array
        result = np.where(v_list == np.amax(v_list))
        # print('Returned tuple of arrays :', result)
        #print('List of the second Indices of maximum element :', result[0])

        try:
            index_max_2 = int(result[0])
            v_list[int(result[0])] = 0
        except TypeError:
            index_max_2 = int(result[0][0])
            v_list[int(result[0][0])] = 0
        #print(f"index_1:{index_max_1}\nindex_2:{index_max_2}")

        return [vote_count_dict,[index_max_1,index_max_2]]

    def get_model_embedding_model(self):
        """
        creating a new model from fashioNet take the 3erd layer for
        image embeddings
        :return: embedding_model
        """
        return Model(inputs=self.model_fashioNet.inputs, outputs=self.model_fashioNet.layers[3].output)

    def get_image_embedding(self,path):
        # process image
        img = image.load_img(path, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        #vgg16_feature = self.image_embedding_model.predict(img_data)
        ######creating new model from fashionnet layers
        model = self.get_model_embedding_model()
        v_feature = model.predict(img_data)
        #return feature vector [0:25088] for AnnoyIndex
        feature_vec = v_feature.flatten()
        return feature_vec[0:25088]






#r_model = Recommendation_Model()
#r_model.get_user_prediction()