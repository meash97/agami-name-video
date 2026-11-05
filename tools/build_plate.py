# Clean plate: erase the whole name "USHA" from every frame; record per-frame ink colour.
import cv2, numpy as np, os, json, subprocess
h,w=1292,724; ANG=-20.5
M=cv2.getRotationMatrix2D((w/2,h/2),ANG,1.0); Minv=cv2.invertAffineTransform(M)
rot=cv2.imread('rot.png'); gray=cv2.cvtColor(rot,cv2.COLOR_BGR2GRAY).astype(np.float32)
# ink alpha of USHA in rotated space
y0,y1,x0,x1=583,627,128,218
g=gray[y0:y1,x0:x1]; paper=np.percentile(g,85); ink=np.percentile(g,2)
a=np.clip((paper-g)/(paper-ink),0,1)
b=(a>0.45).astype(np.uint8); n,lab,st,_=cv2.connectedComponentsWithStats(b)
keep=np.zeros_like(b)
for i in range(1,n):
    if st[i,cv2.CC_STAT_AREA]>=6: keep[lab==i]=1
keep=cv2.dilate(keep,np.ones((3,3),np.uint8)); a=a*keep
full=np.zeros((h,w),np.float32); full[y0:y1,x0:x1]=a
erase=cv2.dilate((full>0.05).astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
erase[:, 219:]=0
erase_o=(cv2.warpAffine(erase.astype(np.float32),Minv,(w,h))>0.3).astype(np.uint8)
inkmask_o=cv2.warpAffine(full,Minv,(w,h))>0.6
blob=cv2.dilate(erase_o,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))).astype(np.float32)
feather=np.minimum(cv2.GaussianBlur(blob,(0,0),1.5),1)
ys,xs=np.where(blob>0); Y0,Y1,X0,X1=ys.min()-30,ys.max()+30,xs.min()-30,xs.max()+30
th=np.deg2rad(20.5); D=100
f1=cv2.imread('frames/0001.png').astype(np.float32)
T=np.float32([[1,0,D*np.cos(th)],[0,1,-D*np.sin(th)]])
plate=cv2.warpAffine(f1,T,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)[Y0:Y1,X0:X1]
SIG=8
def lowpass(img,mask):
    k=(1-mask)[...,None]
    return cv2.GaussianBlur(img*k,(0,0),SIG)/np.maximum(cv2.GaussianBlur(k,(0,0),SIG),1e-3)[...,None]
def inkm(img):
    gg=cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.dilate((gg<np.median(gg)-35).astype(np.uint8),np.ones((7,7),np.uint8)).astype(np.float32)
lowP=lowpass(plate,inkm(plate))
ring=(cv2.dilate(blob,np.ones((25,25),np.uint8))>0)&(blob==0)
_g=cv2.cvtColor(cv2.imread('frames/0241.png'),cv2.COLOR_BGR2GRAY); ring&=(_g>np.median(_g[ring])-25)  # paper only (no ink/doodles)
os.makedirs('plate',exist_ok=True); inks=[]
for i in range(1,242):
    im=cv2.imread(f'frames/{i:04d}.png'); imf=im.astype(np.float32); roi=imf[Y0:Y1,X0:X1]
    px=im[inkmask_o].reshape(-1,3).astype(np.float32); lum=px.mean(1); inkc=np.median(px[lum<=np.percentile(lum,30)],axis=0)
    pp=im[ring].reshape(-1,3).astype(np.float32); pl=pp.mean(1); paperc=np.median(pp[pl>=np.percentile(pl,40)],axis=0)
    inks.append([round(float(v),3) for v in (inkc/paperc)[::-1]])  # RGB ink/paper ratio
    lowF=lowpass(roi,np.maximum(blob[Y0:Y1,X0:X1],inkm(roi)))
    relit=plate*np.clip(lowF/np.maximum(lowP,1),0.4,2.5)
    fe=feather[Y0:Y1,X0:X1][...,None]
    res=im.copy(); res[Y0:Y1,X0:X1]=np.clip(roi*(1-fe)+relit*fe,0,255).astype(np.uint8)
    cv2.imwrite(f'plate/{i:04d}.png',res)
json.dump(inks,open('inks.json','w'))
subprocess.run('ffmpeg -v error -y -framerate 24 -i plate/%04d.png -i input.mp4 -map 0:v -map 1:a -c:v libx264 -preset slow -crf 19 -bf 0 -g 24 -pix_fmt yuv420p -c:a aac -b:a 128k -movflags +faststart -shortest plate.mp4',shell=True,check=True)
print('plate KB',os.path.getsize('plate.mp4')//1024, 'inks', inks[0], inks[-1])
