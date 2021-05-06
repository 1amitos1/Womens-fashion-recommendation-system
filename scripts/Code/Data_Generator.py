import os
import cv2
from keras.utils import Sequence
from keras.utils import np_utils

from Models.Network_building import *


class DataGenerator(Sequence):
    """Data Generator inherited from keras.utils.Sequence
    Args:
        directory: the path of data set, and each sub-folder will be assigned to one class
        batch_size: the number of data points in each batch
        shuffle: whether to shuffle the data per epoch
    Note:
        If you want to load file with other data format, please fix the method of "load_data" as you want
    """

    def __init__(self, directory, batch_size=1, shuffle=True, data_augmentation=True):
        # Initialize the params
        self.batch_size = batch_size
        self.directory = directory
        self.shuffle = shuffle
        self.data_aug = data_augmentation
        # Load all the save_path of files, and create a dictionary that save the pair of "data:label"
        self.X_path, self.Y_dict = self.search_data()
        # Print basic statistics information
        self.print_stats()


    def search_data(self):
        X_path = []
        Y_dict = {}
        # list all kinds of sub-folders
        self.dirs = sorted(os.listdir(self.directory))
        one_hots = np_utils.to_categorical(range(len(self.dirs)))
        for i, folder in enumerate(self.dirs):
            folder_path = os.path.join(self.directory, folder)
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                # append the each file path, and keep its label
                X_path.append(file_path)
                Y_dict[file_path] = one_hots[i]
        return X_path, Y_dict

    def print_stats(self):
        # calculate basic information
        self.n_files = len(self.X_path)
        self.n_classes = len(self.dirs)
        self.indexes = np.arange(len(self.X_path))
        np.random.shuffle(self.indexes)
        # Output states
        print("Found {} files belonging to {} classes.".format(self.n_files, self.n_classes))
        for i, label in enumerate(self.dirs):
            print('%10s : ' % (label), i)
        return None

    def __len__(self):
        # calculate the iterations of each epoch
        steps_per_epoch = np.ceil(len(self.X_path) / float(self.batch_size))
        return int(steps_per_epoch)

    def __getitem__(self, index):
        """Get the data of each batch
        """
        # get the indexs of each batch
        batch_indexs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # using batch_indexs to get path of current batch
        batch_path = [self.X_path[k] for k in batch_indexs]
        # get batch data
        batch_x, batch_y = self.data_generation(batch_path)
        return batch_x, batch_y

    def on_epoch_end(self):
        # shuffle the data at each end of epoch
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def read_image(self,img_path, H, W):
        #print(f"in read img\n{img_path}\n")
        # img_path=img_path+".jpg"
        #print(f"in read img--{img_path}")
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (W, H))  # you can resize to  (128,128) or (256,256)

        return img

    def data_generation(self, batch_path):
        #print(f"in data generation\n {batch_path}\n")
        # load data into memory, you can change the np.load to any method you want
        batch_x = [self.load_data(x) for x in batch_path]
        batch_y = [self.Y_dict[x] for x in batch_path]
        # transfer the data format and take one-hot coding for labels
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        return batch_x, batch_y

    def load_data(self, path):
        # load the processed .npy files which have 5 channels (1-3 for RGB, 4-5 for optical flows)
        #data = np.load(path, mmap_mode='r')
        #print(f"[+][+] in load data:path-->{path}]n")
        data = self.read_image(path, 224, 224)
        data = np.float32(data)
        return data

#
# model = get_compile_model()
# model.save_weights(r'C:\Users\amit hayoun\Desktop\weights_at_epoch_%d.h5' % (0))
#
# #path_train = r"E:\FINAL_PROJECT_DATA\Rec_project\data_set\dataset_example\train"
#
# path_train = r"F:\data for recommender system\FINAL_DATA_SORT\Train"
# path_val = r"F:\data for recommender system\FINAL_DATA_SORT\Val"
# num_epochs=3
# batch_size=3
# checkpoint_dir=""
# # callbacks_list = [keras.callbacks.ModelCheckpoint(
# # keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir + "/ckpt-{epoch}", save_freq="epoch"))]
# callbacks_list =None
# num_workers=4
#
# train_generator = DataGenerator(directory=path_train,batch_size=batch_size,data_augmentation=False)
# val_generator = DataGenerator(directory=path_val, batch_size=batch_size,data_augmentation=False)
#
#
# #self.model_to_save.save_weights(r'/root/gate_flow_slow_fast/weight_save_2/weights_at_epoch_%d.h5' % (epoch+1+15))
#
# hist = model.fit_generator(
#     generator=train_generator,
#     validation_data=val_generator,
#     callbacks=callbacks_list,
#     verbose=1,
#     epochs=num_epochs,
#     workers=num_workers ,
#     max_queue_size=4,
#     steps_per_epoch=len(train_generator),
#     validation_steps=len(val_generator))