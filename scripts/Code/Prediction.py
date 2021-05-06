from keras.models import model_from_json
import gc
import cv2
import numpy as np
from keras.optimizers import adam, Adam
from sklearn import metrics
from Models.Data_Generator import DataGenerator


def read_image(img_path,H=224,W=224):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (W,H)) # you can resize to  (128,128) or (256,256)
    return img

def load_model():
    # load json and create model
    json_file = open(r"E:\FINAL_PROJECT_DATA\FashioNet\Models\model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model_final = model_from_json(loaded_model_json)
    # load weights into new model
    model_final.load_weights(r"E:\FINAL_PROJECT_DATA\FashioNet\Models\save_weights.h5")
    adam = Adam(lr=1e-3)
    model_final.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    print("Loaded model from disk")
    return model_final

def get_y_true_label(y_true_vec):
    pre_dict = {0: "Blouses", 1: "Dresses", 2: "Shorts", 3: "Skirts"}
    pre_dict2 = {"Blouses":'B',"Dresses":'D' ,"Shorts":'SH',"Skirts":'SK'}
    # Get the maximum element from a Numpy array
    maxElement = np.amax(y_true_vec)
    #print('Max element from Numpy Array : ', maxElement)
    # Get the indices of maximum element in numpy array
    result = np.where(y_true_vec == np.amax(y_true_vec))
    #print('Returned tuple of arrays :', result)
    #print('List of Indices of maximum element :', result[0])
    ##print(f"[+][+] label = {pre_dict.get(int(result[0]))}\n")
    key = result[0][0]

    #LABEL = pre_dict.get(int(result[0]))
    LABEL = pre_dict.get(int(key))
    ##print(f"1[+][+][+] label = {pre_dict2.get(LABEL)}\n")
    return pre_dict2.get(LABEL)

def prediction_processes(pred):
    pred = pred[0]
    pre_dict={0:"Blouses",1:"Dresses",2:"Shorts",3:"Skirts"}
    pre_dict2 = {"Blouses": 'B', "Dresses": 'D', "Shorts": 'SH', "Skirts": 'SK'}
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
    #print(f"Res = {pre_dict.get(key)}")
    #LABEL = pre_dict.get(int(result[0]))
    LABEL = pre_dict.get(int(key))
    #print(f"2[+][+][+] label = {pre_dict2.get(LABEL)}\n")
    return pre_dict2.get(LABEL)


def test_sklearn_metrics(y_true,y_pred):

    #
    # # Constants
    # C = "Cat"
    # F = "Fish"
    # H = "Hen"
    #print(y_true)
    #print(y_pred)
    # True values
    #y_true = [C, C, C, C, C, C, F, F, F, F, F, F, F, F, F, F, H, H, H, H, H, H, H, H, H]
    # Predicted values
    #y_pred = [C, C, C, C, H, F, C, C, C, C, C, C, H, H, F, F, C, C, C, H, H, H, H, H, H]
    # Print the confusion matrix
    print(metrics.confusion_matrix(y_true, y_pred))
    # Print the precision and recall, among other metrics
    print(metrics.classification_report(y_true, y_pred, digits=3))



model_final = load_model()

directory =r"F:\data for recommender system\FINAL_DATA_SORT\Test"
X= DataGenerator(directory,batch_size=2, shuffle=True, data_augmentation=False)

###############################################3
start = 0
end = 1
sample = len(X.X_path)
print("Number of sample={}\n".format(sample))

y_true = []
y_predictions_list = []

while end <= len(X.X_path[0:1]):
    path = X.X_path[start:end]
    start = end
    end = end + 1
    batch_x, batch_y = X.data_generation(path)
    batch_y = batch_y[0]
    true_label_index = get_y_true_label(batch_y)

    y_true.append(true_label_index)

    #print(batch_x)
    pred = model_final.predict(batch_x, verbose=0)
    y_pred = prediction_processes(pred)

    y_predictions_list.append(y_pred)

test_sklearn_metrics(y_true,y_predictions_list)


##############################################33

#test_img_path=r"E:\FINAL_PROJECT_DATA\FashioNet\shirt.jpg"
# test_img_path=r"E:\FINAL_PROJECT_DATA\FashioNet\TEST_PICTURE\Blouses_Shirts1.jpg"
# #test_img_path=r"E:\FINAL_PROJECT_DATA\FashioNet\TEST_PICTURE\long_dress.jpeg"
# H, W = 224, 224
# img = read_image(test_img_path,H,W)
# img = img.reshape((-1,224, 224, 3))
#
# pred = model_final.predict(img, batch_size=128,verbose=1)
# pred = model_final.evaluate()
#
# pred = pred[0]
#
# categorical="""
#     Blouses_Shirts :  0
#     Dresses :  1
#     Shorts :  2
#     Skirts :  3
# """
# print(f"pred len={len(pred)}\n")
# for i in range(len(pred)):
#     #print(f"pred{i}={round(pred[i], 5)}\n")
#     print(f"pred{i}={pred[i]}\n")
#
#
#
#
# # Get the maximum element from a Numpy array
# maxElement = np.amax(pred)
# print('Max element from Numpy Array : ', maxElement)
# # Get the indices of maximum element in numpy array
# result = np.where(pred == np.amax(pred))
# print('Returned tuple of arrays :', result)
# print('List of Indices of maximum element :', result[0])