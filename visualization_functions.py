import matplotlib.pyplot as plt

<<<<<<< HEAD
def show_images(images, labels=None, img_per_row=8, img_height=1, show_axis=False, colorbar=False):
    h = images[0].shape[1] // images[0].shape[0]*img_height + 1
=======
def show_images(images, labels=None, img_per_row=8, colorbar=False):
    h = images[0].shape[1] // images[0].shape[0]*0.5 + 1
>>>>>>> 8f0d497a17d1cb3cda3234db936d76d680b351aa
    if not labels:
        labels = range(len(images))
    fig, axes = plt.subplots(len(images)//img_per_row+1*int(len(images)%img_per_row>0), img_per_row, 
                             figsize=(16, h*len(images)//img_per_row+1))
    for i in range(len(images)):
        if len(images) <= img_per_row:
            axes[i%img_per_row].title.set_text(labels[i])
            im = axes[i%img_per_row].imshow(images[i])
            if colorbar:
                fig.colorbar(im, ax=axes[i%img_per_row])
<<<<<<< HEAD
                
            if not show_axis:
                axes[i//img_per_row, i%img_per_row].axis('off')
    
=======
            axes[i//img_per_row, i%img_per_row].axis('off')

>>>>>>> 8f0d497a17d1cb3cda3234db936d76d680b351aa
        else:
            axes[i//img_per_row, i%img_per_row].title.set_text(labels[i])
            im = axes[i//img_per_row, i%img_per_row].imshow(images[i])
            if colorbar:
                fig.colorbar(im, ax=axes[i//img_per_row, i%img_per_row])
<<<<<<< HEAD
            if not show_axis:
                axes[i//img_per_row, i%img_per_row].axis('off')
=======
            axes[i//img_per_row, i%img_per_row].axis('off')
>>>>>>> 8f0d497a17d1cb3cda3234db936d76d680b351aa
            
    plt.show()