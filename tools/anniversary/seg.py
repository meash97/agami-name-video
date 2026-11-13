import cv2, numpy as np, json
rot=cv2.imread('rot.png'); g=cv2.cvtColor(rot,cv2.COLOR_BGR2GRAY).astype(np.float32)
bands={1:(493,542),2:(548,600),3:(604,656),4:(658,712),5:(714,768)}
texts={1:'SANTOSHJISUBHASHJICHHAJER,',2:'WISHINGYOUATRULYSPECIAL',3:'COUPLEDAYLOVEWITHLOVEAND',4:'FULLOFCHERISHEDMOMENTS,',5:'HAPPIESTANNIVERSARY'}
XR={1:(150,600),2:(140,600),3:(135,625),4:(145,640),5:(165,545)}
out={};vis=[]
for ln,(y0,y1) in bands.items():
    x0,x1=XR[ln]; band=g[y0:y1,x0:x1]; bg=cv2.GaussianBlur(g,(0,0),15)[y0:y1,x0:x1]
    ink=(bg-band)>55; col=ink.sum(0); segs=[];ins=False
    for x,c in enumerate(col):
        if c>0 and not ins: s=x;ins=True
        if c==0 and ins: segs.append((s+x0,x+x0));ins=False
    if ins: segs.append((s+x0,x1))
    segs=[s for s in segs if s[1]-s[0]>=2]
    print(ln,len(segs),'expected',len(texts[ln]),[ (a,b) for a,b in segs])
    out[ln]=segs
    img=cv2.resize(rot[y0:y1,130:650],None,fx=2.5,fy=2.5,interpolation=cv2.INTER_CUBIC)
    for i,(a,b) in enumerate(segs):
        cv2.rectangle(img,(int((a-130)*2.5),0),(int((b-130)*2.5),img.shape[0]-1),(0,0,255),1); cv2.putText(img,str(i),(int((a-130)*2.5)+1,11),cv2.FONT_HERSHEY_SIMPLEX,0.35,(255,0,0),1)
    vis.append(img)
cv2.imwrite('seg.png',np.vstack(vis)); json.dump(out,open('segs.json','w'))
