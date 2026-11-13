# Clean plate: erase the whole name "USHA" from every frame; record per-frame ink colour.
import cv2, numpy as np, os, json, subprocess
h,w=1292,724; ANG=-20.5
M=cv2.getRotationMatrix2D((w/2,h/2),ANG,1.0); Minv=cv2.invertAffineTransform(M)
rot=cv2.imread('rot.png'); gray=cv2.cvtColor(rot,cv2.COLOR_BGR2GRAY).astype(np.float32)
def ink_alpha(box, xcut=None):
    y0,y1,x0,x1=box
    g=gray[y0:y1,x0:x1]; paper=np.percentile(g,85); ink=np.percentile(g,2)
    a=np.clip((paper-g)/(paper-ink),0,1)
    b=(a>0.45).astype(np.uint8); n,lab,st,_=cv2.connectedComponentsWithStats(b)
    keep=np.zeros_like(b)
    for i in range(1,n):
        if st[i,cv2.CC_STAT_AREA]>=6: keep[lab==i]=1
    keep=cv2.dilate(keep,np.ones((3,3),np.uint8)); a=a*keep
    full=np.zeros((h,w),np.float32); full[y0:y1,x0:x1]=a
    if xcut: full[:, xcut:]=0
    return full
th=np.deg2rad(20.5); SIG=8
def make_job(full, D, src_frame, first_frame):
    erase=cv2.dilate((full>0.05).astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    erase_o=(cv2.warpAffine(erase.astype(np.float32),Minv,(w,h))>0.3).astype(np.uint8)
    blob=cv2.dilate(erase_o,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))).astype(np.float32)
    feather=np.minimum(cv2.GaussianBlur(blob,(0,0),1.5),1)
    ys,xs=np.where(blob>0); Y0,Y1,X0,X1=ys.min()-30,ys.max()+30,xs.min()-30,xs.max()+30
    f1=cv2.imread(f'frames/{src_frame:04d}.png').astype(np.float32)
    T=np.float32([[1,0,D*np.cos(th)],[0,1,-D*np.sin(th)]])
    plate=cv2.warpAffine(f1,T,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)[Y0:Y1,X0:X1]
    return dict(blob=blob,feather=feather,box=(Y0,Y1,X0,X1),plate=plate,first=first_frame,erase_o=erase_o)
usha_full=ink_alpha((583,627,128,218),xcut=219)
inkmask_o=cv2.warpAffine(usha_full,Minv,(w,h))>0.6
jobs=[make_job(usha_full,100,1,1), make_job(ink_alpha((580,622,52,92)),45,1,165), make_job(ink_alpha((574,636,224,277)),170,1,1)]   # name, sparkle, 'JI,'
blob=jobs[0]['blob']; erase_o=jobs[0]['erase_o']
def lowpass(img,mask):
    k=(1-mask)[...,None]
    return cv2.GaussianBlur(img*k,(0,0),SIG)/np.maximum(cv2.GaussianBlur(k,(0,0),SIG),1e-3)[...,None]
def inkm(img):
    gg=cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.dilate((gg<np.median(gg)-35).astype(np.uint8),np.ones((7,7),np.uint8)).astype(np.float32)
for J in jobs: J['lowP']=lowpass(J['plate'],inkm(J['plate']))
ring=(cv2.dilate(blob,np.ones((25,25),np.uint8))>0)&(blob==0)
_g=cv2.cvtColor(cv2.imread('frames/0241.png'),cv2.COLOR_BGR2GRAY); ring&=(_g>np.median(_g[ring])-25)  # paper only (no ink/doodles)
# hand/nib matte for the first frames (the pen is still finishing 'I,' right where the name ends)
REG=dict(x=0,y=540,w=290,h=110); MATTE_UNTIL=22
motion=json.load(open('motion.json'))
refb=cv2.GaussianBlur(cv2.imread('frames/0045.png').astype(np.float32),(0,0),1.2)
cb=np.array([[0,540],[290,540],[290,650],[0,650]],np.float32); cbf=(Minv[:,:2]@cb.T+Minv[:,2:]).T
MX0,MY0=max(0,int(cbf[:,0].min())-30),max(0,int(cbf[:,1].min())-30); MX1,MY1=min(w,int(cbf[:,0].max())+30),min(h,int(cbf[:,1].max())+30)
def hand_matte(f):
    d=np.abs(cv2.GaussianBlur(f,(0,0),1.2)-refb)[MY0:MY1,MX0:MX1].max(2)
    m=(d>26).astype(np.uint8); m=cv2.morphologyEx(m,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    n,lab,st,_=cv2.connectedComponentsWithStats(m); k=np.zeros_like(m)
    for j in range(1,n):
        if st[j,cv2.CC_STAT_AREA]>=250: k[lab==j]=1
    k=cv2.morphologyEx(k,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)))
    k=cv2.dilate(k,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    out=np.zeros((h,w),np.float32); out[MY0:MY1,MX0:MX1]=k; return out
os.makedirs('plate',exist_ok=True); inks=[]
for i in range(1,242):
    im=cv2.imread(f'frames/{i:04d}.png')
    px=im[inkmask_o].reshape(-1,3).astype(np.float32); lum=px.mean(1); inkc=np.median(px[lum<=np.percentile(lum,30)],axis=0)
    pp=im[ring].reshape(-1,3).astype(np.float32); pl=pp.mean(1); paperc=np.median(pp[pl>=np.percentile(pl,40)],axis=0)
    inks.append([round(float(v),3) for v in (inkc/paperc)[::-1]])  # RGB ink/paper ratio
    res=im.copy()
    hm=hand_matte(im.astype(np.float32)) if i<=MATTE_UNTIL else np.zeros((h,w),np.float32); hmd=cv2.dilate(hm,np.ones((9,9),np.uint8)); keepf=1-np.minimum(cv2.GaussianBlur(hm,(0,0),1.0),1)
    for J in jobs:
        if i<J['first']: continue
        Y0,Y1,X0,X1=J['box']; roi=res[Y0:Y1,X0:X1].astype(np.float32)
        lowF=lowpass(roi,np.maximum(np.maximum(J['blob'][Y0:Y1,X0:X1],inkm(roi)),hmd[Y0:Y1,X0:X1]))
        relit=J['plate']*np.clip(lowF/np.maximum(J['lowP'],1),0.4,2.5)
        fe=(J['feather']*keepf)[Y0:Y1,X0:X1][...,None]
        res[Y0:Y1,X0:X1]=np.clip(roi*(1-fe)+relit*fe,0,255).astype(np.uint8)
    mv=motion[i-1]; hm_s=cv2.warpAffine(hm,np.float32([[1,0,-mv[0]],[0,1,-mv[1]]]),(w,h))
    strip=cv2.warpAffine(hm_s,M,(w,h),flags=cv2.INTER_LINEAR)[REG['y']:REG['y']+REG['h'],REG['x']:REG['x']+REG['w']]
    out=np.zeros((h+8+REG['h'],w,3),np.uint8); out[:h]=res; out[h+8:,:REG['w']]=(np.clip(strip,0,1)*255).astype(np.uint8)[...,None]
    cv2.imwrite(f'plate/{i:04d}.png',out)
json.dump(inks,open('inks.json','w'))
import base64
sp=ink_alpha((580,622,52,92)); ys,xs=np.where(sp>0.15); sy0,sy1,sx0,sx1=ys.min()-2,ys.max()+3,xs.min()-2,xs.max()+3
a=sp[sy0:sy1,sx0:sx1]; a3=cv2.resize(a,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC); a3=np.clip(a3*1.3,0,1); a3=cv2.GaussianBlur(a3,(5,5),0.6)
a8=(a3*255).astype(np.uint8); ok,buf=cv2.imencode('.png',np.dstack([np.full_like(a8,255)]*3+[a8]))
json.dump([{'name':'sparkle','src':'data:image/png;base64,'+base64.b64encode(buf).decode(),'x':int(sx0),'y':int(sy0),'w':int(sx1-sx0),'h':int(sy1-sy0),'fromFrame':164}],open('decor.json','w'))
print('sparkle box',sx0,sy0,sx1,sy1)
subprocess.run('ffmpeg -v error -y -framerate 24 -i plate/%04d.png -i input.mp4 -map 0:v -map 1:a -c:v libx264 -preset slow -crf 19 -bf 0 -g 24 -pix_fmt yuv420p -c:a aac -b:a 128k -movflags +faststart -shortest plate.mp4',shell=True,check=True)
print('plate KB',os.path.getsize('plate.mp4')//1024, 'inks', inks[0], inks[-1])
