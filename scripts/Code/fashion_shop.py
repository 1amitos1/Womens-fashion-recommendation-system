import os
from keras.models import model_from_json
import gc
import cv2
import numpy as np
import random
import pandas as pd
from MAIN_SYSTEM.model_sys_1 import Recommendation_Model
from annoy import AnnoyIndex
import random
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np

from MAIN_SYSTEM.user import User


class fashion_shop:
    def __init__(self, csv_file , user):
        self.pre_dict = {0: "Blouses", 1: "Dresses", 2: "Shorts", 3: "Skirts"}
        self.csv_file_path = csv_file
        self.Dresses_path_dict = {}
        self.Blouses_path_dict = {}
        self.Skirts_path_dict = {}
        self.Shorts_path_dict = {}
        self.fashion_recommendation_model = Recommendation_Model()
        self.user = user
        # Fashion_Recommendation_Model.get_user_prediction()
        # self.show_data()

    def get_new_item_id(self):
        data_path = r"C:\Users\amit hayoun\Desktop\reant_anyway_data"
        index_a = 0
        index_b = 0
        index_c = 0
        index_d = 0
        for category in os.listdir(data_path):
            category_path = os.path.join(data_path, category)
            # print(category_path)
            for pic in os.listdir(category_path):
                # print(f"pic ={pic}")
                full_pic_path = os.path.join(category_path, pic)
                if category == "Dresses":
                    cat_index = 100 + index_a
                    index_a = index_a + 1
                    self.Dresses_path_dict.update({cat_index: full_pic_path})
                elif category == "Blouses":
                    cat_index = 200 + index_b
                    index_b = index_b + 1
                    self.Blouses_path_dict.update({cat_index: full_pic_path})
                elif category == "Shorts":
                    cat_index = 300 + index_c
                    index_c = index_c + 1

                    self.Shorts_path_dict.update({cat_index: full_pic_path})
                elif category == "Skirts":
                    cat_index = 400 + index_d
                    index_d = index_d + 1
                    self.Skirts_path_dict.update({cat_index: full_pic_path})

        # print(len(self.Blouses_path_dict))  # 23,-(200,223)
        # print(len(self.Dresses_path_dict))  # 20 (100,120)
        # print(len(self.Shorts_path_dict))  # 20 (300,320)
        # print(len(self.Skirts_path_dict))  # 18 (400)
        # print(len(path_dict.values()))
        # print(path_dict.get(1))
    def delete_columns(self):
        df = pd.read_csv(self.csv_file_path)

        # # Check if Dataframe has a column with Label name 'City'
        # if 'Unnamed: 0' and 'Unnamed: 0.1' in df.columns:

        df.drop(['Unnamed: 0'], axis='columns', inplace=True)
        df.drop(['Division Name'], axis='columns', inplace=True)
        df.drop(['Title'], axis='columns', inplace=True)
        df.drop(['Review Text'], axis='columns', inplace=True)
        df.drop(['Department Name'], axis='columns', inplace=True)

        print(df.sample(30))
        df.to_csv('new_data_format.csv')
    def get_new_user_id(self):
        list1 = ['A', 'B', 'C', 'D', 'E']
        b = random.randint(0, 4)
        age_dict = {'A':32,'B':42,'C':36,'D':28,'E':25}
        return [list1[b],age_dict.get(list1[b])]

    def show_data(self):
        df = pd.read_csv(self.csv_file_path, delimiter=",")
        print(df.columns)
        print(df.sample(30))

    def create_User_And_Item_ID_(self):
        df = pd.read_csv(self.csv_file_path, delimiter=",")
        df.drop(['Clothing ID'], axis='columns', inplace=True)
        df.drop(['Unnamed: 0'], axis='columns', inplace=True)
        df.drop(['Age'], axis='columns', inplace=True)
        # random generation of user and item id

        user_list = []
        item_ID = []
        path_lsit = []
        age_list= []
        for i in range(len(df)):
            # choosing user ID
            if df.values[i][-1] == "Dresses":
                random_index = random.randint(0, len(self.Dresses_path_dict))
                random_index = random_index + 100
                path_lsit.append(self.Dresses_path_dict.get(random_index))
                item_ID.append(random_index)
                l = self.get_new_user_id()
                user_list.append(l[0])
                age_list.append(l[1])
            elif df.values[i][-1] == "Blouses":
                random_index = random.randint(0, len(self.Blouses_path_dict))
                random_index = random_index + 200
                path_lsit.append(self.Blouses_path_dict.get(random_index))
                item_ID.append(random_index)
                l = self.get_new_user_id()
                user_list.append(l[0])
                age_list.append(l[1])
            elif df.values[i][-1] == "Shorts":
                random_index = random.randint(0, len(self.Shorts_path_dict))
                random_index = random_index + 300
                path_lsit.append(self.Shorts_path_dict.get(random_index))
                item_ID.append(random_index)
                l = self.get_new_user_id()
                user_list.append(l[0])
                age_list.append(l[1])
            elif df.values[i][-1] == "Skirts":
                random_index = random.randint(0, len(self.Skirts_path_dict))
                random_index = random_index + 400
                path_lsit.append(self.Skirts_path_dict.get(random_index))
                item_ID.append(random_index)
                l = self.get_new_user_id()
                user_list.append(l[0])
                age_list.append(l[1])
            else:
                #print("[-][-] ERROR\n")
                df.values[i][-1] = "Skirts"
                random_index = random.randint(0, len(self.Skirts_path_dict))
                random_index = random_index + 400
                path_lsit.append(self.Skirts_path_dict.get(random_index))
                item_ID.append(random_index)
                l = self.get_new_user_id()
                user_list.append(l[0])
                age_list.append(l[1])

        df.insert(0, "UserID", user_list, True)
        df.insert(1, "Age", age_list, True)
        df.insert(2, "Clothing ID", item_ID, True)
        df.insert(3, "Pic_path", path_lsit, True)


        #write sample data
        w_df = df.sample(250)
        w_df.to_csv("__data_for_simulation.csv")
    def Running_simulation(self):
        # step_1
        self.get_new_item_id()
        #step_2
        self.create_User_And_Item_ID_()


    # recommendation process
    def model_fashion_recognitions(self,path_list):
        """
        :param path_list:
        :return:tow category that the model most frequency found
        """
        category_dict = {0: "Blouses_Shirts", 1: "Dresses", 2: "Shorts", 3: "Skirts"}
        vote_dict, index = self.fashion_recommendation_model.get_user_prediction(path_list)
        category_1 = category_dict.get(index[0])
        category_2 = category_dict.get(index[1])
        return [category_1,category_2]

    # def create_inventory(self,cat_1_path,cat_2_path,category):
    #     """
    #
    #     :param cat_1_path:DataFrame object ,user data for the specific category
    #     :param cat_2_path:DataFrame object ,user data for the specific category
    #     :param category: list of tow category
    #     :return:
    #     """
    #     df = pd.read_csv(self.csv_file_path, delimiter=",")
    #     # print(cat_1_path.columns)
    #     #
    #     # print(cat_1_path.head(5))
    #     # print(category[0])
    #     # print(cat_2_path.head(5))
    #     # print(category[1])
    #
    #     df1 = df.loc[df['Class Name'] == category[0]]
    #     df2 = df.loc[df['Class Name'] == category[1]]
    #
    #     path1_list = []
    #     path2_list=[]
    #     for i in range(0,len(df1)):
    #         #print(df.values[i][2])
    #         if(df1.values[i][2] in cat_1_path['Pic_path'].values):
    #             continue
    #         else:
    #             path1_list.append(df1.values[i][2])
    #
    #     for i in range(0,len(df2)):
    #         #print(df.values[i][2])
    #         if(df2.values[i][2] in cat_2_path['Pic_path'].values):
    #             continue
    #         else:
    #             path2_list.append(df2.values[i][2])
    #
    #     # write all category path to a file
    #
    #     f = open(f'inventory_{category[0]}.txt', 'a')
    #     for path in path1_list:
    #         if path == 'nan':
    #             continue
    #         f.write(f"{path}\n")
    #     f.close()
    #     f = open(f'inventory_{category[1]}.txt', 'a')
    #     for path in path2_list:
    #         if str(path) == 'nan':
    #             continue
    #         f.write(f"{path}\n")
    #     f.close()

    def create_inventory(self,category):
        item_path_dict={}
        # get anny vector at size 25088
        size = 25088
        #size = 802816
        annoy_inventory_1 = self.get_annoy_index(size)
        # get the top tow user items from user by rating
        user_feature_pic = self.user.get_top_item_by_user_rating(category)
        # get feature img from model
        feature_vec_i1 = self.fashion_recommendation_model.get_image_embedding(user_feature_pic[0])
        feature_vec_i2 = self.fashion_recommendation_model.get_image_embedding(user_feature_pic[0])
        #add
        annoy_inventory_1.add_item(0, feature_vec_i1)
        annoy_inventory_1.add_item(0, feature_vec_i2)
        item_path_dict.update({0: user_feature_pic[0]})
        item_path_dict.update({0: user_feature_pic[1]})
        #print(self.csv_file_path)
        #print(os.path.join(self.csv_file_path,category))
        category_path = os.path.join(self.csv_file_path,category)
        #print(category_path)
        i = 2
        for pic_name in os.listdir(category_path):
            path = os.path.join(category_path,pic_name)
            feature_vec = self.fashion_recommendation_model.get_image_embedding(path)
            annoy_inventory_1.add_item(i, feature_vec)
            item_path_dict.update({i:path})
            i = i+1
        annoy_inventory_1.build(10)  # 10 trees
        save_name = category +".ann"
        #annot_inventory_1.save('test.ann')
        annoy_inventory_1.save(save_name)
        return item_path_dict

    def get_annoy_index(self,size):
        f = size
        t = AnnoyIndex(f, 'angular')  # Length of item vector that will be indexed
        return t

    def print_image(self,item_path_dict,l):

        # print(item_path_dict.get(l[0]))
        # print(item_path_dict.get(l[1]))
        # print(item_path_dict.get(l[2]))


        # Show user item category
        im1 = cv2.imread(item_path_dict.get(l[0]))
        imS1 = cv2.resize(im1, (224, 224))
        cv2.imshow('user item', imS1)

        im2 = cv2.imread(item_path_dict.get(l[1]))
        imS2 = cv2.resize(im2, (224, 224))

        im3 = cv2.imread(item_path_dict.get(l[2]))
        imS3 = cv2.resize(im3, (224, 224))

        im4 = cv2.imread(item_path_dict.get(l[3]))
        imS4 = cv2.resize(im4, (224, 224))
        #numpy_horizontal_concat = np.concatenate((imS1, imS2,imS3,imS4), axis=1)
        numpy_horizontal_concat = np.concatenate((imS2, imS3, imS4), axis=1)
        cv2.imshow('Numpy Horizontal Concat', numpy_horizontal_concat)

        cv2.waitKey(0)

    def gen_recommendation_from_inventory(self,item_path_dict,category):
        size = 25088
        #size = 802816
        u = AnnoyIndex(size, 'angular')
        save_name = category + ".ann"
        u.load(save_name)  # super fast, will just mmap the file
        #print(u.get_nns_by_item(0, 4))  # will find the 1000 nearest neighbors
        l = u.get_nns_by_item(0, 4)
        self.print_image(item_path_dict,l)

    def run_recommendation(self):
        #get user data
        u_data = self.user.get_user_data()
        ## model go over all user picture and return 2 category
        print(f"[+][+]Step-1:\n\tFashion recognition model start scanning user data\n")
        path = u_data['Pic_path']
        category = self.model_fashion_recognitions(path)
        #category = ['Dresses','Skirts']
        print(f"\tModel found tow category:{category}\n")

        # filter user data by category items , get tow category
        #print(f"[+][+]Step-2:\n\tFilter user data by category:{category}\n")
        #f_data_cat_1 , f_data_cat_2  = u.filter_category(category=category)

        print(f"[+][+]Step-2:\n\tcreate fashion shop inventory for tow category: {category}"
              f"\n\tUsing AnnoyIndex\n")
        # get user item pic_path list, and subtract from general data , to create inventory
        item_path_dict1 = self.create_inventory(category[0])
        item_path_dict2 = self.create_inventory(category[1])


        print(f"[+][+]Step-3:\n\tgen_recommendation_from_inventory\n")
        print(f"\tgen_recommendation_from_inventory for category {category[0]}\n")
        self.gen_recommendation_from_inventory(item_path_dict1 ,category[0])
        print(f"\tgen_recommendation_from_inventory for category {category[1]}\n")
        self.gen_recommendation_from_inventory(item_path_dict2, category[1])


# csv_file_change = r"E:\FINAL_PROJECT_DATA\FashioNet\MAIN_SYSTEM\data_for_simulation.csv"
# csv_file = r"E:\FINAL_PROJECT_DATA\FashioNet\MAIN_SYSTEM\new_data_format.csv"
#
# f = fashion_shop(csv_file,None)
#
# # Step-1 create simulation data
#
# f.show_data()
# f.Running_simulation()

# Step-2 create recommendation
users_data_path =r'E:\FINAL_PROJECT_DATA\FashioNet\MAIN_SYSTEM\__data_for_simulation.csv'
u = User("B",users_data_path)


path = r"C:\Users\amit hayoun\Desktop\SHOP_DATA"
f = fashion_shop(path,u)
f.run_recommendation()

