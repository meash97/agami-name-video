# Builds the level-space alpha of the new "a" and the shifted "mi." for Agimi -> Agami
import cv2, numpy as np
from common import *
gray=cv2.cvtColor(level(fr(241)),cv2.COLOR_BGR2GRAY).astype(np.float32)
def alpha_box(x0,x1,y0,y1,keep_big=False):
    g=gray[y0:y1,x0:x1]; paper=np.percentile(gray[y0:y1,max(0,x0-30):x1+30],80); ink=np.percentile(gray[y0:y1,max(0,x0-60):x1+60],1)
    a=np.clip((paper-g)/(paper-ink),0,1); b=(a>0.35).astype(np.uint8); n,lab,st,_=cv2.connectedComponentsWithStats(b); k=np.zeros_like(b)
    if n>1:
        big=1+np.argmax(st[1:,4])
        for i in range(1,n):
            if (i==big) or (not keep_big and st[i,4]>=4): k[lab==i]=1
    a=a*cv2.dilate(k,np.ones((3,3),np.uint8)); full=np.zeros((H,W),np.float32); full[y0:y1,x0:x1]=a; return full
SW=3.3; UP=8
def stroke(pts_list,box_w,box_h,smooth=True):
    c=np.zeros((int((box_h+8)*UP),int((box_w+8)*UP)),np.uint8)
    for pts in pts_list:
        p=np.array([[(4+x)*UP,(4+y)*UP] for x,y in pts],np.float32)
        if smooth and len(p)>3:
            for _ in range(3):
                q=0.75*p[:-1]+0.25*p[1:]; r=0.25*p[:-1]+0.75*p[1:]; p=np.vstack([p[:1],np.column_stack([q,r]).reshape(-1,2),p[-1:]])
        cv2.polylines(c,[p.astype(np.int32)],False,255,int(SW*UP),cv2.LINE_AA)
    a=cv2.resize(c.astype(np.float32)/255,(box_w+8,box_h+8),interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(a,(3,3),0.6)
def make_a(xl,ytop,ybot):
    w=12; h=ybot-ytop
    bowl=[(9.5,2.2),(6.5,0.2),(3.0,1.2),(0.6,5.0),(0.4,9.5),(2.2,h-1.0),(5.5,h-0.2),(8.6,h-3.5),(10.0,h*0.45)]
    stem=[(10.6,0.0),(10.2,h*0.5),(10.0,h-1.6),(11.2,h),(12.6,h-1.0)]
    a=stroke([bowl,stem],w+2,h)
    full=np.zeros((H,W),np.float32); full[ytop-4:ytop-4+a.shape[0],xl-4:xl-4+a.shape[1]]=a; return full
i1=alpha_box(357,367,798,834); i1[:, :358]=0
m_=alpha_box(368,394,806,836,keep_big=True); i2=alpha_box(395,404,795,834); dot=alpha_box(407,419,818,834)
SHIFT=5
A_=make_a(360,814,830)
def shiftL(a,d): return cv2.warpAffine(a,np.float32([[1,0,d],[0,1,0]]),(W,H))
if __name__=='__main__':
    lv=level(fr(241)).astype(np.float32)
    er=cv2.dilate((np.maximum.reduce([i1,m_,i2,dot])>0.05).astype(np.uint8),np.ones((5,5),np.uint8))
    clean=cv2.inpaint(lv.astype(np.uint8),er,3,cv2.INPAINT_TELEA).astype(np.float32)
    k=np.array([0.32,0.18,0.28],np.float32)
    out=clean.copy()
    for lay in (A_,shiftL(m_,SHIFT),shiftL(i2,SHIFT),shiftL(dot,SHIFT)):
        al=lay[...,None]; out=out*(1-al)+out*k*al
    cv2.imwrite('agami_ctx.png',cv2.resize(np.vstack([lv[785:850,300:440],out[785:850,300:440]]).astype(np.uint8),None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC))
