import cv2, numpy as np, sys
sys.path.insert(0,'/home/claude/t5'); from common import *
P8=to_ref(fr(241),241); cv2.imwrite('P241.png',P8)
ref=P8.astype(np.float32); hsv=cv2.cvtColor(P8,cv2.COLOR_BGR2HSV)
g=cv2.cvtColor(P8,cv2.COLOR_BGR2GRAY).astype(np.float32)
# ---- old logo mask: medallion + AGAMI + their raised-edge shadows
bgl=cv2.GaussianBlur(g,(0,0),30)
m=((hsv[...,1]>55)|(np.abs(g-bgl)>14)).astype(np.uint8)
yy,xx=np.mgrid[0:H,0:W]; lvy=(M[1,0]*xx+M[1,1]*yy+M[1,2]); roi=((lvy<572)&(yy>110)&(xx<480)).astype(np.uint8)
m*=roi
m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
n,lab,st,_=cv2.connectedComponentsWithStats(m); k=np.zeros_like(m)
for j in range(1,n):
    if st[j,4]>=40: k[lab==j]=1
old=cv2.dilate(k,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(19,19)))
hw=((inkmap(P8,40)>0.25)&(hsv[...,1]<90)).astype(np.uint8)          # dark handwriting (the logo is gold)
n2,l2,st2,_=cv2.connectedComponentsWithStats(hw); prot=np.zeros_like(hw)
for j in range(1,n2):
    yy2,xx2=np.where(l2==j)
    if ((M[1,0]*xx2+M[1,1]*yy2+M[1,2])>566).any(): prot[l2==j]=1
old[cv2.dilate(prot,np.ones((7,7),np.uint8))>0]=0
# ---- paper fill: smooth lighting interpolated from the surroundings + fine paper texture from a clean area
def inkm(img): return cv2.dilate((inkmap(np.clip(img,0,255).astype(np.uint8),40)>0.2).astype(np.uint8),np.ones((7,7),np.uint8)).astype(np.float32)
mask_t=np.maximum(old.astype(np.float32),inkm(ref))
def smooth_fill(img,mask):
    out=None; acc=None
    for sig in (10,25,60,120):
        lp=lowpass(img,mask,sig); sup=cv2.GaussianBlur(1-mask,(0,0),sig)
        w_=np.clip(sup/0.25,0,1)[...,None]
        out=lp if out is None else out*acc+lp*(1-acc)
        acc=w_ if acc is None else np.maximum(acc,w_)
    return out
lowT=smooth_fill(ref,mask_t)
SRC=np.float32([[1,0,300],[0,1,-40]])           # texture from the blank paper right of the logo
src=cv2.warpAffine(ref,SRC,(W,H),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REFLECT)
srcm=inkm(src); lowS=lowpass(src,srcm,6)
hp=(src-lowS)/np.maximum(lowS,1)                 # relative texture
hp=np.clip(hp,-0.08,0.08); hp[cv2.dilate(srcm,np.ones((9,9),np.uint8))>0]=0; hp=cv2.GaussianBlur(hp,(3,3),0.5)
fill=lowT*(1+hp)
fe=np.minimum(cv2.GaussianBlur(old.astype(np.float32),(0,0),2.0),1)[...,None]
clean=ref*(1-fe)+fill*fe
cv2.imwrite('dbg_clean.png',np.clip(clean,0,255).astype(np.uint8)[60:620,0:520])
# ---- new logo: printed onto the paper (multiply with the paper), mapped with the page affine
lg=cv2.imread('logo.png').astype(np.float32); bgc=np.median(lg[5:40,5:200].reshape(-1,3),axis=0)
cont=(np.abs(lg-bgc).max(2)>14).astype(np.float32)
bg_est=np.broadcast_to(bgc,lg.shape)
ratio=np.clip(lg/np.maximum(bg_est,1),0,1.12)
dd=np.abs(lg-bgc).max(2); soft=np.clip((dd-8)/30,0,1)
soft=np.maximum(soft,cv2.dilate((dd>30).astype(np.float32),np.ones((3,3),np.uint8))*0)   # alpha from distance to the paper colour
inside=cv2.morphologyEx((dd>14).astype(np.uint8),cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)))
n_,lab_,st_,_=cv2.connectedComponentsWithStats(inside); big_=1+np.argmax(st_[1:,4])
med=(lab_==big_).astype(np.uint8); cs_,_=cv2.findContours(med,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(med,cs_,-1,1,-1)
soft=np.maximum(soft,cv2.erode(med,np.ones((5,5),np.uint8)).astype(np.float32))   # white diamonds inside the medallion stay artwork
ratio=1-soft[...,None]*(1-ratio)
near=cv2.dilate((soft>0.02).astype(np.float32),np.ones((5,5),np.uint8))
Mm=np.load('M_logo.npy').astype(np.float32)
UP=2
big=cv2.resize(ratio,None,fx=UP,fy=UP,interpolation=cv2.INTER_CUBIC)
Mu=Mm.copy(); Mu[:,:2]/=UP
rat_ref=cv2.warpAffine(big,Mu,(W,H),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_CONSTANT,borderValue=(1,1,1)); rat_ref=cv2.GaussianBlur(rat_ref,(0,0),0.7)
cov=cv2.warpAffine(cv2.resize(near,None,fx=UP,fy=UP),Mu,(W,H),flags=cv2.INTER_LINEAR)
# soft raised-print shadow (light from upper right): darken just below-left of the artwork
cmask=cv2.warpAffine(cv2.resize(soft,None,fx=UP,fy=UP),Mu,(W,H),flags=cv2.INTER_LINEAR)
sh=cv2.GaussianBlur(cv2.warpAffine(cmask,np.float32([[1,0,-2],[0,1,3]]),(W,H)),(0,0),2.2)*(1-cmask)
lit=clean*rat_ref*(1-0.18*sh[...,None])
newlogo_ref=np.clip(lit,0,255)
area=np.maximum(old.astype(np.float32),np.clip(cov*1.0,0,1))
area=cv2.dilate((area>0.02).astype(np.uint8),np.ones((5,5),np.uint8)).astype(np.float32)
np.save('logo_area.npy',area); cv2.imwrite('logo_plate.png',newlogo_ref.astype(np.uint8))
cv2.imwrite('dbg_logo.png',np.hstack([fr(241)[60:620,0:520],newlogo_ref.astype(np.uint8)[60:620,0:520]]))
