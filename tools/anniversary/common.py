import cv2, numpy as np
H,W=1292,724; ANG=-23.75
M=cv2.getRotationMatrix2D((W/2,H/2),ANG,1.0); Minv=cv2.invertAffineTransform(M)
def fr(i): return cv2.imread(f'frames/{i:04d}.png')
def level(img,interp=cv2.INTER_CUBIC): return cv2.warpAffine(img,M,(W,H),flags=interp,borderMode=cv2.BORDER_REPLICATE)
def unlevel(img,interp=cv2.INTER_LINEAR): return cv2.warpAffine(img,Minv,(W,H),flags=interp)
def inkmap(img):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32) if img.ndim==3 else img.astype(np.float32)
    return np.clip((cv2.GaussianBlur(g,(0,0),12)-g)/90,0,1)
def lowpass(img,mask,sig=8):
    k=(1-mask)[...,None]
    return cv2.GaussianBlur(img*k,(0,0),sig)/np.maximum(cv2.GaussianBlur(k,(0,0),sig),1e-3)[...,None]
