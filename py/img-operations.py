import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
import _models.load_model as m

def img2vec(imagePath, cuda = False):
    """
    Image to Features

    Parameters
    ----------
    img : PIL Image, Image opened in PIL.
    cuda : bool, optional, GPU or not. The default is False.

    Returns
    -------
    Array, Features of image

    """
    img = Image.open(imagePath)
    device = torch.device("cuda" if cuda else "cpu")
    model = m.model_pt_res18()
#    model = models.resnet18(pretrained=True)
    extraction_layer = model._modules.get('avgpool')
    model = model.to(device)
    model.eval()
    scaler = transforms.Resize((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()
    image = normalize(to_tensor(scaler(img))).unsqueeze(0).to(device)
    layer_output_size=512
    my_embedding = torch.zeros(1, layer_output_size, 1, 1)
    def copy_data(m, i, o):
        my_embedding.copy_(o.data)
    h = extraction_layer.register_forward_hook(copy_data)
    h_x = model(image)
    h.remove()
    return my_embedding.numpy()[0, :, 0, 0]

def cos_sim(query_vec, match_vec):
    """
    :param query_vec: feature extracted from query image
    :param match_vec: feature extracted from image from db
    :return: cosine similarity of input vectors * 100
    """
    match = cosine_similarity(query_vec.reshape((1, -1)), match_vec.reshape((1, -1)))[0][0]
    match = float(match*100)
    match = round(match,2)
    return match

def show_images(images, titles = None, cols = 1, size = 11):
    """
    Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of imagepaths
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    images = [np.asarray(Image.open(i)) for i in images]
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        # a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        a = fig.add_subplot(np.ceil(n_images/float(cols)), cols, n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    # fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    fig.set_size_inches(size, size)
    # plt.figure()
    plt.show()