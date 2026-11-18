import cv2, numpy as np, json, sys
sys.path.insert(0,'/home/claude/t4'); from common import *
rot=cv2.imread('rot.png'); g=cv2.cvtColor(rot,cv2.COLOR_BGR2GRAY).astype(np.float32); bg=cv2.GaussianBlur(g,(0,0),15)
LINES={1:(552,584,'DEARSANTOSHJUCHAAJAR,'),2:(600,632,'THANKYOUFORAKINDVISIT'),3:(648,680,'ANDGIVINGYOUR'),4:(694,726,'VALUABLETIME.'),5:(740,772,'WARMREGARDS,'),6:(788,820,'AGAMI.')}
out={}; vis=[]
for ln,(y0,y1,txt) in LINES.items():
    ink=(bg[y0:y1+6]-g[y0:y1+6])>38; col=ink.sum(0); segs=[];ins=False
    for x,c in enumerate(col):
        if c>0 and not ins: s=x;ins=True
        if c==0 and ins: segs.append((s,x));ins=False
    segs=[s for s in segs if s[1]-s[0]>=2 and s[0]>90]
    print(ln,len(segs),len(txt),txt)
    out[ln]=segs
    img=cv2.resize(rot[y0-4:y1+8,100:620],None,fx=2.4,fy=2.4,interpolation=cv2.INTER_CUBIC)
    for i,(a,b) in enumerate(segs):
        cv2.rectangle(img,(int((a-100)*2.4),0),(int((b-100)*2.4),img.shape[0]-1),(0,0,255),1); cv2.putText(img,str(i),(int((a-100)*2.4)+1,10),cv2.FONT_HERSHEY_SIMPLEX,0.33,(255,0,0),1)
    vis.append(img)
cv2.imwrite('seg.png',np.vstack(vis)); json.dump(out,open('segs.json','w'))
