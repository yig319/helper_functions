import torch
import numpy as np

def fft_transform(image):
    '''
    assert r.dtype==torch.float32, 'Suppose to be torch.float32, not '+str(r.dtype)
    '''  
    
    image = torch.tensor(image)
    out = torch.clone(image)
    img_fft = torch.fft.fft2(image, dim=(0,1))
    img_shift = torch.fft.fftshift(img_fft)
    out = np.log(np.abs(img_shift))
    
    # abnormal value
    out[out==-np.inf] = 0
    
    # scale to 0-1
    out =  (out * 1/out.max())
    return out.numpy()