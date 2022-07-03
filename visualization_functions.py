import matplotlib.pyplot as plt
import numpy as np
# import torch

def show_images(images, labels=None, img_per_row=8, img_height=1, colorbar=False, 
                clim=False, scale_0_1=False, hist_bins=None, show_axis=False):
    assert type(images) == list or type(images) == np.ndarray, "do not use torch.tensor for hist"

    def scale(x):
        if x.min() < 0:
            return (x - x.min()) / (x.max() - x.min())
        else:
            return x/(x.max() - x.min())
    
    h = images[0].shape[1] // images[0].shape[0]*img_height + 1
    if not labels:
        labels = range(len(images))
        
    n = 1
    if hist_bins: n +=1
        
    fig, axes = plt.subplots(n*len(images)//img_per_row+1*int(len(images)%img_per_row>0), img_per_row, 
                             figsize=(16, n*h*len(images)//img_per_row+1))
    for i, img in enumerate(images):
        
#         if torch.is_tensor(x_tensor):
#             if img.requires_grad: img = img.detach()
#             img = img.numpy()
            
        if scale_0_1: img = scale(img)
        
        if len(images) <= img_per_row and not hist_bins:
            index = i%img_per_row
        else:
            index = (i//img_per_row)*n, i%img_per_row

        axes[index].title.set_text(labels[i])
        im = axes[index].imshow(img)
        if colorbar:
            fig.colorbar(im, ax=axes[index])
            
        if clim:
            m, s = np.mean(img), np.std(img)            
            im.set_clim(m-3*s, m+3*s) 
            
        if not show_axis:
            axes[index].axis('off')

        if hist_bins:
            index_hist = (i//img_per_row)*n+1, i%img_per_row
            h = axes[index_hist].hist(img.flatten(), bins=hist_bins)
        
    plt.show()
    
    
def show_plots(ys, xs=None, labels=None, ys_fit=None, img_per_row=4, subplot_height=3):
    
#     assert type(ys) == np.ndarray, "use numpy array with shape=(n, y)"
#     assert type(labels) == list or type(labels) == type(None), "use list for labels"
    
    if type(labels) == type(None): labels = range(len(ys))
    
    if type(xs) == type(None):
        xs = []
        for y in ys:
            xs.append(np.linspace(0, len(y), len(y)+1))            
#         xs, _ = np.meshgrid(np.linspace(0, 1, ys.shape[1]), np.ones((ys.shape[0])))
        
    fig, axes = plt.subplots(len(ys)//img_per_row+1*int(len(ys)%img_per_row>0), img_per_row, 
                             figsize=(16, subplot_height*len(ys)//img_per_row+1))    
    
    for i in range(len(ys)):
        
        if len(ys) <= img_per_row:
            index = i%img_per_row
        else:
            index = (i//img_per_row), i%img_per_row

        axes[index].title.set_text(labels[i])
        
        im = axes[index].plot(xs[i], ys[i], marker='.')
        
        if type(ys_fit) != type(None):
            im = axes[index].plot(xs[i], ys_fit[i])
                   
    fig.tight_layout()
    plt.show()