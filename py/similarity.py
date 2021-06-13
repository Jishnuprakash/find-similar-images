# Initialisation
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
import model.load_model as m


# Functions
def feature_extraction(file, cuda=False):
    """
    description: extracts features from a given image with resmet18 model.
    ;param file: either a file path or PIL Image (image opened with PIL) of the target picture.
    ;param cuda: bool, optional, GPU or not. The default is False.
    Returns: Array, Features of image
    """
    img = Image.open(file)
    device = torch.device("cuda" if cuda else "cpu")
    model = m.model_pt_res18()
    extraction_layer = model._modules.get('avgpool')
    model = model.to(device)
    model.eval()
    scaler = transforms.Resize((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()
    image = normalize(to_tensor(scaler(img))).unsqueeze(0).to(device)
    layer_output_size = 512
    my_embedding = torch.zeros(1, layer_output_size, 1, 1)
    def copy_data(m, i, o):
        my_embedding.copy_(o.data)
    h = extraction_layer.register_forward_hook(copy_data)
    h_x = model(image)
    h.remove()
    return my_embedding.numpy()[0, :, 0, 0]


def similarity(target, reference):
    """
    description: It calculates cosine similarities between the target image features with the set of references
    ;param target: features (array) of the target image
    ;param reference: feature (array) of the references (expected to be more than 1)
    return: cosine similarity of target features with  * 100

    question: 20210613|Rakesh: why do we multiply the similarity calculated with 100 ?
                               why do we round the match value here ?
    """
    similarities = cosine_similarity(target.reshape((1, -1)), reference.reshape((1, -1)))[0][0]
    similarities = float(similarities * 100)
    similarities = round(similarities, 2)
    return similarities


def show_images(list_files, cols=1, size=11, add_titles=None):
    """
    description: Display a list of images in a single figure with matplotlib.
    :param cols: Default = 1, Number of columns in figure (number of rows is set to np.ceil(n_images/float(cols))).
    :param list_files: List of file paths (images)
    ;param cols
    :param size:
    :param add_titles: List of titles corresponding to each image. Must have the same length as titles.
    return: a plot device
    """
    images = [np.asarray(Image.open(i)) for i in list_files]
    assert ((add_titles is None) or (len(images) == len(add_titles)))
    n_images = len(images)
    if add_titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, add_titles)):
        a = fig.add_subplot(np.ceil(n_images / float(cols)), cols, n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(size, size)
    plt.show()
