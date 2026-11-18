import cv2, numpy as np, json
from common import *
rot=cv2.imread('rot.png'); g=cv2.cvtColor(rot,cv2.COLOR_BGR2GRAY).astype(np.float32); bg=cv2.GaussianBlur(g,(0,0),15)
bands={1:(604,662,125,545),2:(662,706,110,580),3:(706,750,220,520),4:(756,800,250,500),5:(800,830,300,420)}
out={};vis=[]
for ln,(y0,y1,x0,x1) in bands.items():
    ink=(bg[y0:y1,x0:x1]-g[y0:y1,x0:x1])>50; col=ink.sum(0); segs=[];ins=False
    for x,c in enumerate(col):
        if c>0 and not ins: s=x;ins=True
        if c==0 and ins: segs.append((s+x0,x+x0));ins=False
    if ins: segs.append((s+x0,x1))
    segs=[s for s in segs if s[1]-s[0]>=2]; out[ln]=segs
    print(ln,len(segs),segs)
    img=cv2.resize(rot[y0:y1,100:600],None,fx=2.6,fy=2.6,interpolation=cv2.INTER_CUBIC)
    for i,(a,b) in enumerate(segs):
        cv2.rectangle(img,(int((a-100)*2.6),0),(int((b-100)*2.6),img.shape[0]-1),(0,0,255),1); cv2.putText(img,str(i),(int((a-100)*2.6)+1,11),cv2.FONT_HERSHEY_SIMPLEX,0.38,(255,0,0),1)
    vis.append(img)
cv2.imwrite('seg.png',np.vstack(vis)); json.dump(out,open('segs.json','w'))
