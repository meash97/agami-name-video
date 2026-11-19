# AGIMI -> AGAMI on the finished page (level space), then back to frame space
import cv2, numpy as np, sys
sys.path.insert(0,'/home/claude/t5'); from common import *
src=cv2.imread('logo_plate.png'); lv=level(src).astype(np.float32); g=cv2.cvtColor(lv.astype(np.uint8),cv2.COLOR_BGR2GRAY).astype(np.float32)
def alpha(x0,x1,y0,y1):
    a=np.zeros((H,W),np.float32); gg=g[y0:y1,x0:x1]; paper=np.percentile(g[y0:y1,max(0,x0-60):x1+60],80); ink=np.percentile(g[880:960,260:470],0.5)
    a[y0:y1,x0:x1]=np.clip((paper-gg)/(paper-ink),0,1); b=(a>0.3).astype(np.uint8); n,lab,st,_=cv2.connectedComponentsWithStats(b); k=np.zeros_like(b)
    for i in range(1,n):
        if st[i,4]>=6: k[lab==i]=1
    return a*cv2.dilate(k,np.ones((3,3),np.uint8))
Y0,Y1=896,956                                   # above the underline
Ibar=alpha(372,398,Y0,Y1); Mi=alpha(390,472,Y0,Y1); Mi[:913,:404]=0; Aa=alpha(262,332,884,957)
erase=cv2.dilate(((Ibar>0.05)|(Mi>0.05)).astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
erase[956:]=0
# paper fill: smooth lighting from around + texture from blank paper higher up
k=erase.astype(np.float32); inkm=cv2.dilate((cv2.GaussianBlur(g,(0,0),12)-g>30).astype(np.uint8),np.ones((5,5),np.uint8)).astype(np.float32)
low=lowpass(lv,np.maximum(k,inkm),10)
tx=cv2.warpAffine(lv,np.float32([[1,0,0],[0,1,-110]]),(W,H),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REFLECT)
lowt=lowpass(tx,cv2.dilate((cv2.GaussianBlur(cv2.cvtColor(tx.astype(np.uint8),cv2.COLOR_BGR2GRAY).astype(np.float32),(0,0),12)-cv2.cvtColor(tx.astype(np.uint8),cv2.COLOR_BGR2GRAY).astype(np.float32)>30).astype(np.uint8),np.ones((5,5),np.uint8)).astype(np.float32),5)
hp=np.clip((tx-lowt)/np.maximum(lowt,1),-0.06,0.06)
fe=np.minimum(cv2.GaussianBlur(k,(0,0),1.2),1)[...,None]
out=lv*(1-fe)+low*(1+hp)*fe
SH=37; SA=0.8
def shift(a,dx,dy=0): return cv2.warpAffine(a,np.float32([[1,0,dx],[0,1,dy]]),(W,H))
# new A: the word's own A, scaled, bottom on the M's bottom
ys,xs=np.where(Aa>0.15); ab=Aa[ys.min():ys.max()+1,xs.min():xs.max()+1]; ab=cv2.resize(ab,None,fx=SA,fy=SA,interpolation=cv2.INTER_AREA)
my=np.where(Mi.max(1)>0.3)[0].max()
Anew=np.zeros((H,W),np.float32); top=my-ab.shape[0]+2; left=368
Anew[top:top+ab.shape[0],left:left+ab.shape[1]]=ab
ink_rgb=np.median(lv[(Mi>0.8)],axis=0)/np.maximum(np.median(low[(Mi>0.8)],axis=0),1)
for lay in (Anew,shift(Mi,SH)):
    al=np.clip(lay,0,1)[...,None]; out=out*(1-al)+out*ink_rgb*al
back=unlevel(out.astype(np.float32)); chg=unlevel(np.maximum(k,np.maximum(Anew,shift(Mi,SH))).astype(np.float32))
chg=cv2.dilate((chg>0.01).astype(np.uint8),np.ones((5,5),np.uint8)).astype(np.float32); chg=np.minimum(cv2.GaussianBlur(chg,(0,0),1.0),1)[...,None]
final=src.astype(np.float32)*(1-chg)+back*chg
cv2.imwrite('final_page.png',np.clip(final,0,255).astype(np.uint8))
cv2.imwrite('dbg_agami.png',cv2.resize(np.vstack([lv[860:975,240:620],level(final.astype(np.uint8))[860:975,240:620]]).astype(np.uint8),None,fx=2,fy=2))
