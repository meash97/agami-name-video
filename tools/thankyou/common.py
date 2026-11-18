import cv2, numpy as np, json
H,W=1280,720
ANG,SHEAR=np.load('/home/claude/t4/textlevel.npy')
_R=np.vstack([cv2.getRotationMatrix2D((W/2,H/2),ANG,1.0),[0,0,1]])
_S=np.array([[1,SHEAR,-SHEAR*H/2],[0,1,0],[0,0,1]])
L3=_S@_R                       # frame(ref) -> level
M=L3[:2].astype(np.float32); Minv=cv2.invertAffineTransform(M)   # level -> frame(ref)
CAM=[np.array(c,np.float32) for c in json.load(open('/home/claude/t4/cam.json'))]   # ref -> frame i
def fr(i): return cv2.imread(f'/home/claude/t4/f/{i:04d}.png')
def level(img,interp=cv2.INTER_CUBIC): return cv2.warpAffine(img,M,(W,H),flags=interp,borderMode=cv2.BORDER_REPLICATE)
def unlevel(img,interp=cv2.INTER_LINEAR): return cv2.warpAffine(img,Minv,(W,H),flags=interp)
def to_ref(img,i,interp=cv2.INTER_LINEAR):   # frame i -> ref coords
    return cv2.warpAffine(img,CAM[i-1],(W,H),flags=interp|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REPLICATE)
def from_ref(img,i,interp=cv2.INTER_LINEAR): # ref coords -> frame i
    return cv2.warpAffine(img,CAM[i-1],(W,H),flags=interp,borderMode=cv2.BORDER_REPLICATE)
def inkmap(img,div=60):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32) if img.ndim==3 else img.astype(np.float32)
    return np.clip((cv2.GaussianBlur(g,(0,0),12)-g)/div,0,1)
def lowpass(img,mask,sig=8):
    k=(1-mask)[...,None]
    return cv2.GaussianBlur(img*k,(0,0),sig)/np.maximum(cv2.GaussianBlur(k,(0,0),sig),1e-3)[...,None]
def to_frame(pts): pts=np.asarray(pts,np.float32); return (Minv[:,:2]@pts.T+Minv[:,2:]).T
