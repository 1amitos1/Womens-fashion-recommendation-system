import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from annoy import AnnoyIndex
import glob

from fastai.vision import show_image, open_image

np.random.seed(42)


class SaveFeatures():
    features=None
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)
        self.features = None
    def hook_fn(self, module, input, output):
        out = output.detach().cpu().numpy()
        if isinstance(self.features, type(None)):
            self.features = out
        else:
            self.features = np.row_stack((self.features, out))
    def remove(self):
        self.hook.remove()


# loads cnn pkl file
# def load_cnn(pkl_filename):
#     learn = load_learner(path='Img/', file=pkl_filename)
#     return learn


# returns the embeddings for a single image,
# from a single given CNN's last FC layer
def get_embeddings_for_image(cnn, img_path):
    hook = hook_output(cnn.model_fashioNet[-1][-3])
    cnn.predict(open_image(img_path))
    hook.remove()
    return hook.stored.cpu()[0]


# returns the concatenated embeddings for a single image,
# from the given list of CNNs' last FC layer
def get_combined_embeddings_for_image(cnns, img_path):
    embeddings = []
    for cnn in cnns:
        embeddings.append(get_embeddings_for_image(cnn, img_path))
    return np.concatenate(embeddings)


# returns the embeddings for multiple image, from
# a single given CNN's last FC layer
def get_embeddings_for_images(cnn, img_paths):
    sf = SaveFeatures(cnn.model_fashioNet[-1][-3])
    cnn.data.add_test(img_paths)
    cnn.get_preds(DatasetType.Test)
    sf.remove()
    return sf.features


# returns the embeddings for multiple image, from
# a list of given CNNs' last FC layer
def get_combined_embeddings_for_images(cnns, img_paths):
    embeddings = []
    for cnn in cnns:
        embeddings.append(get_embeddings_for_images(cnn, img_paths))

    return np.concatenate(embeddings, axis=1)


# creates an ANN index from the given list of embeddings
def create_ann_index(embeddings, dim=512, trees=10):
    ann_index = AnnoyIndex(dim)
    for i in range(len(embeddings)):
        ann_index.add_item(i, embeddings[i])
    ann_index.build(trees)
    return ann_index


# queries the given vector against the given ANN index
def query_ann_index(ann_index, dataset, embeddings, n=5):
    nns = ann_index.get_nns_by_vector(embeddings, n=n, include_distances=True)
    img_paths = [dataset[i] for i in nns[0]]
    return img_paths, nns[1]


# displays the list of given image paths
def display_images(img_paths, sim_scores):
    for i, img_path in enumerate(img_paths):
        print(sim_scores[i])
        show_image(open_image(img_path))


# Get and display recs
def get_recs(img_path, cnns, ann_index, dataset, n=5):
    embedding = get_combined_embeddings_for_image(cnns, img_path)
    img_paths, sim_scores = query_ann_index(ann_index, dataset, embedding, n)
    #     return display_images(img_paths, sim_scores)
    return img_paths, sim_scores




