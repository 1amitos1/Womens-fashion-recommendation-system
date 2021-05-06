import pandas as pd


def extract_data_from_csv():
    """
    open csv file and extract 4 class
    [Blouses_Shirts :  0,  Dresses :  1 ,Shorts :  2 ,Skirts :  3]
    and save in a new csv file
    :return:
    """
    filepath=r"E:\FINAL_PROJECT_DATA\FashioNet\recommendation_fashion_data\Womens Clothing E-Commerce Reviews.csv"
    df = pd.read_csv(filepath,delimiter=",")

    Dresses= df.loc[df['Class Name'] == "Dresses"]
    Blouses= df.loc[df['Class Name'] == "Blouses"]
    Skirts = df.loc[df['Class Name'] == "Skirts"]
    Shorts = df.loc[df['Class Name'] == "Shorts"]

    Dresses.to_csv("baba2.csv",mode='a')
    Shorts.to_csv("baba2.csv",mode='a')
    Skirts.to_csv("baba2.csv",mode='a')
    Blouses.to_csv("baba2.csv",mode='a')

def delete_columns():

    df = pd.read_csv(r"E:\FINAL_PROJECT_DATA\FashioNet\baba2.csv")

    # Check if Dataframe has a column with Label name 'City'
    if 'Unnamed: 0' and 'Unnamed: 0.1' in df.columns :
        print(True)
        df.drop(['Unnamed: 0'] , axis='columns', inplace=True)
        df.drop(['Unnamed: 0.1'], axis='columns', inplace=True)
    print(df.sample(30))
    df.to_csv('baba3.csv')

delete_columns()
