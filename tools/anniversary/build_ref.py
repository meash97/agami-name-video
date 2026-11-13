# Clean reference page P: frame 164 with the hand (right edge) replaced by doodle-free paper from frame 241
import cv2, numpy as np
from common import *
a=fr(164).astype(np.float32); b=fr(241).astype(np.float32)
d=np.abs(cv2.GaussianBlur(a,(0,0),1.5)-cv2.GaussianBlur(b,(0,0),1.5)).max(2)
m=(d>18).astype(np.uint8); m[:, :540]=0; m[:400]=0; m[840:]=0
m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
n,lab,st,_=cv2.connectedComponentsWithStats(m)
R=np.zeros_like(m)
for i in range(1,n):
    x,y,w,h,area=st[i]
    if x+w>=W-1 and area>300: R[lab==i]=1          # touches the right edge = hand / pen / its shadow
R[555:610,700:]=1
R=cv2.dilate(R,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17)))
# never let R swallow frame-164 text: protect its ink (except right at the nib)
ink164=(inkmap(a.astype(np.uint8))>0.3).astype(np.uint8)
txt=ink164.copy(); txt[600:, 576:]=0; txt[:600, 640:]=0
R[cv2.dilate(txt,np.ones((5,5),np.uint8))>0]=0
np.save('R164.npy',R)
# doodle-free version of 241 inside R (+margin)
Rbig=cv2.dilate(R,np.ones((41,41),np.uint8))
dood=((inkmap(b.astype(np.uint8))>0.12)&(Rbig>0)).astype(np.uint8)
dood=cv2.dilate(dood,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))).astype(np.float32)
SH=300                                                   # clone blank paper from 300 px lower in the same frame
T=np.float32([[1,0,0],[0,1,-SH]]); src=cv2.warpAffine(b,T,(W,H),borderMode=cv2.BORDER_REFLECT)
inkm=lambda im: cv2.dilate((inkmap(im.astype(np.uint8))>0.12).astype(np.uint8),np.ones((7,7),np.uint8)).astype(np.float32)
lowB=lowpass(b,np.maximum(dood,inkm(b)),10); lowS=lowpass(src,inkm(src),10)
fill=src*np.clip(lowB/np.maximum(lowS,1),0.5,2)
fe=np.minimum(cv2.GaussianBlur(dood,(0,0),1.5),1)[...,None]
bclean=b*(1-fe)+fill*fe
# match 241 lighting to 164 around R
valid=((Rbig>0)&(R==0)).astype(np.float32); valid*= (1-inkm(a))*(1-inkm(bclean))
la=cv2.GaussianBlur(a*valid[...,None],(0,0),25); lb=cv2.GaussianBlur(bclean*valid[...,None],(0,0),25)
gain=np.clip(la/np.maximum(lb,1e-3),0.7,1.4); gain[cv2.GaussianBlur(valid,(0,0),25)<0.02]=1
fR=np.minimum(cv2.GaussianBlur(R.astype(np.float32),(0,0),3),1)[...,None]
P=a*(1-fR)+bclean*gain*fR
P=np.clip(P,0,255).astype(np.uint8)
dot=np.zeros((H,W),np.uint8); cv2.circle(dot,(577,678),7,255,-1); P=cv2.inpaint(P,dot,4,cv2.INPAINT_TELEA)   # pen-touch dot after the last Y
cv2.imwrite('P.png',P)
ov=a.astype(np.uint8).copy(); ov[R>0]=(0,0,255)
cv2.imwrite('dbg_P.png',np.hstack([ov[380:860,440:],np.clip(P,0,255).astype(np.uint8)[380:860,440:]]))
