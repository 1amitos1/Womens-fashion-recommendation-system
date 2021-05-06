import os
from keras.models import model_from_json
import gc
import cv2
import numpy as np
import random
import pandas as pd
from MAIN_SYSTEM.model_sys_1 import Recommendation_Model


class User:
    def __init__(self, id,csv_data_path):
        self.user_id = id
        self.csv_data_path = csv_data_path
        self.user_data = self.get_user_data()

    def get_user_data(self):
        df = pd.read_csv(self.csv_data_path, delimiter=",")
        data = df.loc[df["UserID"] == self.user_id]
        #print(data.head(20))
        #print(df.columns)
        return data.copy()

    def filter_category(self,category):
        """
        :param category:: list with tow category predicted by the model
        :return: list filter data
        """
        data1 = self.user_data.loc[self.user_data["Class Name"] == category[0]]
        data2 = self.user_data.loc[ self.user_data["Class Name"] == category[1]]
        #print(data1.head(10))
        #print(data2.head(10))
        return [data1,data2]

    def get_top_item_by_user_rating(self,category):
        """
        read from user data, and return most high rating item in this
        category
        :param category: string[Dresses,Shorts,Skirts,Blouses_Shirts]
        :return:tow path to user item
        # """
        # print(category)
        # print(self.user_data.head())
        #print(self.user_data.columns)
        data1 = self.user_data.loc[self.user_data["Class Name"] == category]
        #print(data1.sort_values(by=['Rating'],ascending=False).head(2))
        data1 = data1.sort_values(by=['Rating'],ascending=False)

        try:
            item_1 = data1.iloc[0]['Pic_path']
            item_2 = data1.iloc[1]['Pic_path']
        except IndexError:
            item_2 =item_1
        return [item_1,item_2]



# path =r'E:\FINAL_PROJECT_DATA\FashioNet\MAIN_SYSTEM\data_for_simulation.csv'
# u = User("A",path)
# u.filter_category(["Dresses","Blouses"])
# u.get_top_item_by_user_rating("Dresses")