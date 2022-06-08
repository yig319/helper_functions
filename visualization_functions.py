import matplotlib.pyplot as plt
import torch

def show_images(images, labels=None, img_per_row=8, img_height=1, colorbar=False, scale_0_1=False,
                show_hist=False, show_axis=False):
    
    def scale(x):
        if x.min() < 0:
            return (x - x.min()) / (x.max() - x.min())
        else:
            return x/(x.max() - x.min())
    
    h = images[0].shape[1] // images[0].shape[0]*img_height + 1
    if not labels:
        labels = range(len(images))
        
    n = 1
    if show_hist: n +=1
        
    fig, axes = plt.subplots(n*len(images)//img_per_row+1*int(len(images)%img_per_row>0), img_per_row, 
                             figsize=(16, n*h*len(images)//img_per_row+1))
    for i, img in enumerate(images):
        
        if torch.is_tensor(x_tensor):
            if img.requires_grad: img = img.detach()
            img = img.numpy()
            
        if scale: img = scale(img)
        
        if len(images) <= img_per_row and not show_hist:
            index = i%img_per_row
        else:
            index = (i//img_per_row)*n, i%img_per_row

        axes[index].title.set_text(labels[i])
        im = axes[index].imshow(img)
        if colorbar:
            fig.colorbar(im, ax=axes[index])
        if not show_axis:
            axes[index].axis('off')

        if show_hist:
            index_hist = (i//img_per_row)*n+1, i%img_per_row
            h = axes[index_hist].hist(img.flatten())

    plt.show()