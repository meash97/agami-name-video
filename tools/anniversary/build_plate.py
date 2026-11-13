# Anniversary template: clean plate (name line removed, hand kept on top), hand matte strip,
# hand exit + two hearts ending, per-frame motion and ink colour.
import cv2, numpy as np, json, os, subprocess, pickle
from common import *
P=cv2.imread('P.png'); Pf=P.astype(np.float32); rotP=level(P); gray=cv2.cvtColor(rotP,cv2.COLOR_BGR2GRAY).astype(np.float32)
NF=241; LAST=164                       # frames 1..LAST are real footage; after that the hand leaves and the page holds
REG=dict(x=140,y=484,w=584,h=72)       # name band in level space (also the matte strip size)
# ---- line-1 ink (level space) -> erase mask (frame space)
y0,y1,x0,x1=490,548,150,592
g=gray[y0:y1,x0:x1]; paper=np.percentile(g,85); inkv=np.percentile(g,1)
a=np.clip((paper-g)/(paper-inkv),0,1); b=(a>0.4).astype(np.uint8); n,lab,st,_=cv2.connectedComponentsWithStats(b)
keep=np.zeros_like(b)
for i in range(1,n):
    if st[i,cv2.CC_STAT_AREA]>=4: keep[lab==i]=1
a=a*cv2.dilate(keep,np.ones((3,3),np.uint8))
full=np.zeros((H,W),np.float32); full[y0:y1,x0:x1]=a
erase=cv2.dilate((full>0.05).astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
erase_o=(unlevel(erase.astype(np.float32))>0.3).astype(np.uint8)
blob0=cv2.dilate(erase_o,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))).astype(np.float32)
inkcore=unlevel(full)>0.6
# ---- motion of the handwriting (track "SANTOSH Ji", always visible)
c=np.array([[152,492],[318,492],[318,548],[152,548]],np.float32); cf=(Minv[:,:2]@c.T+Minv[:,2:]).T
bx0,by0,bx1,by1=int(cf[:,0].min()),int(cf[:,1].min()),int(cf[:,0].max())+1,int(cf[:,1].max())+1
refink=inkmap(P); tpl=refink[by0:by1,bx0:bx1].copy()
tmask=np.zeros((H,W),np.uint8); cv2.fillConvexPoly(tmask,cf.astype(np.int32),1); tpl*=tmask[by0:by1,bx0:bx1]
UP=8; tplU=cv2.resize(tpl,None,fx=UP,fy=UP,interpolation=cv2.INTER_CUBIC); pad=10
mot=[]
for i in range(1,LAST+1):
    m=inkmap(fr(i)); s=cv2.resize(m[by0-pad:by1+pad,bx0-pad:bx1+pad],None,fx=UP,fy=UP,interpolation=cv2.INTER_CUBIC)
    r=cv2.matchTemplate(s,tplU,cv2.TM_CCOEFF_NORMED); _,mx,_,loc=cv2.minMaxLoc(r); mot.append((loc[0]/UP-pad,loc[1]/UP-pad,mx))
u=np.array(mot); print('motion dx',u[:,0].min(),u[:,0].max(),'dy',u[:,1].min(),u[:,1].max(),'min score',u[:,2].min().round(3))
motion=[[round(float(np.median(u[max(0,k-1):k+2,0])),3),round(float(np.median(u[max(0,k-1):k+2,1])),3)] for k in range(LAST)]+[[0,0]]*(NF-LAST)
json.dump(motion,open('motion.json','w'))
# ---- paper to put under the erased name: blank paper cloned from 300 px lower on the (level) page, relit per frame
dv=Minv[:,:2]@np.array([0,300],np.float32)            # level-space "down 300" in frame space
T=np.float32([[1,0,-dv[0]],[0,1,-dv[1]]]); tex=cv2.warpAffine(Pf,T,(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
def inkm(img): return cv2.dilate((inkmap(np.clip(img,0,255).astype(np.uint8))>0.15).astype(np.uint8),np.ones((7,7),np.uint8)).astype(np.float32)
ys,xs=np.where(blob0>0); Y0,Y1,X0,X1=max(0,ys.min()-40),ys.max()+40,max(0,xs.min()-40),min(W,xs.max()+40)
texR=tex[Y0:Y1,X0:X1]; lowT=lowpass(texR,inkm(texR))
# ---- hand matte (frame space), computed around the name band
cb=np.array([[REG['x'],REG['y']],[REG['x']+REG['w'],REG['y']],[REG['x']+REG['w'],REG['y']+REG['h']],[REG['x'],REG['y']+REG['h']]],np.float32)
cbf=(Minv[:,:2]@cb.T+Minv[:,2:]).T; MX0,MY0=max(0,int(cbf[:,0].min())-30),max(0,int(cbf[:,1].min())-30); MX1,MY1=min(W,int(cbf[:,0].max())+30),min(H,int(cbf[:,1].max())+30)
Pb=cv2.GaussianBlur(Pf,(0,0),1.2)
def hand_matte(f):
    d=np.abs(cv2.GaussianBlur(f,(0,0),1.2)-Pb)[MY0:MY1,MX0:MX1].max(2)
    m=(d>26).astype(np.uint8); m=cv2.morphologyEx(m,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    n,lab,st,_=cv2.connectedComponentsWithStats(m); k=np.zeros_like(m)
    for j in range(1,n):
        if st[j,cv2.CC_STAT_AREA]>=400: k[lab==j]=1
    k=cv2.morphologyEx(k,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)))
    inv=1-k; ff=np.zeros((inv.shape[0]+2,inv.shape[1]+2),np.uint8); bg=inv.copy()      # fill any holes inside the hand
    for sx,sy in [(0,0),(inv.shape[1]-1,0),(0,inv.shape[0]-1)]:
        if bg[sy,sx]==1: cv2.floodFill(bg,ff,(sx,sy),2)
    k[bg==1]=1
    k=cv2.dilate(k,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    out=np.zeros((H,W),np.float32); out[MY0:MY1,MX0:MX1]=k; return out
# ---- ending: hand slides out to the right, then two hearts draw themselves after ANNIVERSARY
R=np.load('R164.npy').astype(np.float32); Rf=np.minimum(cv2.GaussianBlur(cv2.dilate(R,np.ones((5,5),np.uint8)),(0,0),2),1)
f164=fr(LAST).astype(np.float32)
SLIDE=[10,26,48,76,110,150,196,248]                    # px, eased; hand is fully out after 8 frames
hrt=level(fr(241)); hg=cv2.cvtColor(hrt,cv2.COLOR_BGR2GRAY).astype(np.float32)
hc=(M[:,:2]@np.array([605,640],np.float32)+M[:,2]).astype(int); hx0,hx1,hy0,hy1=hc[0]-34,hc[0]+34,hc[1]-32,hc[1]+32
hgc=hg[hy0:hy1,hx0:hx1]; hp=np.percentile(hgc,85); hi=np.percentile(hgc,1); ha=np.clip((hp-hgc)/(hp-hi),0,1)
hb=(ha>0.4).astype(np.uint8); n,lab,st,_=cv2.connectedComponentsWithStats(hb); big=1+np.argmax(st[1:,cv2.CC_STAT_AREA])
ha=ha*cv2.dilate((lab==big).astype(np.uint8),np.ones((3,3),np.uint8)); ys,xs=np.where(ha>0.15); ha=ha[ys.min():ys.max()+1,xs.min():xs.max()+1]
cv2.imwrite('dbg_heart.png',(255-ha*255).astype(np.uint8)); print('heart size',ha.shape)
def heart_layer(cx,cy,width,rotdeg,flip):
    s=width/ha.shape[1]; UPS=3
    hh=cv2.resize(ha[:, ::-1] if flip else ha,None,fx=s*UPS,fy=s*UPS,interpolation=cv2.INTER_AREA)
    hh=cv2.dilate(hh,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(HD,HD))); hh=np.clip((hh*1.25-0.3)/0.7,0,1); hh=cv2.GaussianBlur(hh,(5,5),0.7)
    big=np.zeros((H*UPS//4,W*UPS//4),np.float32)   # local canvas (quarter page at 3x) centred on the heart
    oy,ox=big.shape[0]//2-hh.shape[0]//2,big.shape[1]//2-hh.shape[1]//2; big[oy:oy+hh.shape[0],ox:ox+hh.shape[1]]=hh
    # level->frame for this heart: rotate by rotdeg about canvas centre, then place at (cx,cy) level, then unlevel
    A=np.array([[1/UPS,0,cx-big.shape[1]/(2*UPS)],[0,1/UPS,cy-big.shape[0]/(2*UPS)],[0,0,1]])
    Rm=np.vstack([cv2.getRotationMatrix2D((big.shape[1]/2,big.shape[0]/2),rotdeg,1.0),[0,0,1]])
    full=np.vstack([Minv,[0,0,1]])@A@Rm
    lay=cv2.warpAffine(big,full[:2],(W,H),flags=cv2.INTER_AREA)
    # sweep angle map (frame space) around the heart centre, starting at the top notch, clockwise
    c=(Minv[:,:2]@np.array([cx,cy])+Minv[:,2]); yy,xx=np.mgrid[0:H,0:W]
    ang=(np.degrees(np.arctan2(xx-c[0],-(yy-c[1])))+ANG+360)%360
    return lay,ang
HD=int(os.environ.get('HD','5'))
H1=heart_layer(569,741,28,-6,False); H2=heart_layer(604,738,22,10,True)
HEARTS=[(H1,176,190),(H2,193,205)]
# ---- per-frame render
os.makedirs('plate',exist_ok=True); inks=[]; strips=[]
ring=(cv2.dilate(blob0,np.ones((25,25),np.uint8))>0)&(blob0==0)&(inkmap(P)<0.1)
last_ink=None
for i in range(1,NF+1):
    if i<=LAST:
        f=fr(i).astype(np.float32); mv=motion[i-1]; hm=hand_matte(f)
        sh=np.float32([[1,0,round(mv[0])],[0,1,round(mv[1])]])
        blob=cv2.warpAffine(blob0,sh,(W,H)); core=cv2.warpAffine(inkcore.astype(np.uint8),sh,(W,H))>0
        vis=core&(hm==0); rg=ring&(cv2.dilate(hm,np.ones((15,15),np.uint8))==0)
        if vis.sum()>300 and rg.sum()>300:
            px=f[vis]; lum=px.mean(1); ic=np.median(px[lum<=np.percentile(lum,30)],axis=0)
            pp=f[rg]; pl=pp.mean(1); pc=np.median(pp[pl>=np.percentile(pl,40)],axis=0); last_ink=[round(float(v),3) for v in (ic/pc)[::-1]]
        inks.append(last_ink)
        roi=f[Y0:Y1,X0:X1]; hmd=cv2.dilate(hm,np.ones((9,9),np.uint8))
        lowF=lowpass(roi,np.maximum(np.maximum(blob[Y0:Y1,X0:X1],inkm(roi)),hmd[Y0:Y1,X0:X1]))
        relit=texR*np.clip(lowF/np.maximum(lowT,1),0.4,2.5)
        fe=np.minimum(cv2.GaussianBlur(blob,(0,0),1.5),1)*(1-np.minimum(cv2.GaussianBlur(hm,(0,0),1.0),1))
        fe=fe[Y0:Y1,X0:X1][...,None]
        res=f.copy(); res[Y0:Y1,X0:X1]=roi*(1-fe)+relit*fe
        # matte strip in level space, pre-shifted by the handwriting motion so it lines up with the moving name
        Ms=M.copy(); # level p -> frame q = Minv(p)+mv  =>  sample hm at q:  use inverse map  frame->level with shift
        Tm=np.float32([[1,0,-mv[0]],[0,1,-mv[1]]]); hm_s=cv2.warpAffine(hm,Tm,(W,H))
        strip=cv2.warpAffine(hm_s,M,(REG['x']+REG['w'],REG['y']+REG['h']),flags=cv2.INTER_LINEAR)[REG['y']:,REG['x']:]
    else:
        k=i-LAST-1; res=None
        base=cleanP
        if k<len(SLIDE):
            d=SLIDE[k]; Ts=np.float32([[1,0,d],[0,1,-0.12*d]])
            hs=cv2.warpAffine(f164,Ts,(W,H)); ms=cv2.warpAffine(Rf,Ts,(W,H))[...,None]
            base=cleanP*(1-ms)+hs*ms
        res=base.copy(); r=np.array(last_ink[::-1],np.float32)
        for (lay,ang),fa,fb in HEARTS:
            if i<fa: continue
            prog=min(1,(i-fa+1)/(fb-fa+1)); reveal=np.clip((prog*380-ang)/20,0,1) if prog<1 else 1
            al=(lay*reveal)[...,None]; res=res*(1-al)+res*r*al
        inks.append(last_ink); strip=np.zeros((REG['h'],REG['w']),np.float32)
    if i==LAST:
        # the page as it holds for the ending: reference page with the name line removed
        hm0=np.zeros((H,W),np.float32); roi=Pf[Y0:Y1,X0:X1]
        lowF=lowpass(roi,np.maximum(blob0[Y0:Y1,X0:X1],inkm(roi))); relit=texR*np.clip(lowF/np.maximum(lowT,1),0.4,2.5)
        fe=np.minimum(cv2.GaussianBlur(blob0,(0,0),1.5),1)[Y0:Y1,X0:X1][...,None]
        cleanP=Pf.copy(); cleanP[Y0:Y1,X0:X1]=roi*(1-fe)+relit*fe
        # f164 with the name erased too, for the sliding hand layer
        f164=res.copy()
        # true hand/pen/shadow mask = where the real frame differs from the clean page (so no text is dragged along)
        dd=np.abs(cv2.GaussianBlur(f164,(0,0),1)-cv2.GaussianBlur(cleanP,(0,0),1)).max(2)
        hmask=cv2.dilate(((dd>10)&(R>0)).astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))).astype(np.float32)
        Rf=np.minimum(cv2.GaussianBlur(hmask,(0,0),1.5),1)
    out=np.zeros((H+8+REG['h'],W,3),np.uint8); out[:H]=np.clip(res,0,255).astype(np.uint8)
    out[H+8:,:REG['w']]=(np.clip(strip,0,1)*255).astype(np.uint8)[...,None]
    cv2.imwrite(f'plate/{i:04d}.png',out)
json.dump(inks,open('inks.json','w')); json.dump(REG,open('reg.json','w'))
subprocess.run('ffmpeg -v error -y -framerate 24 -i plate/%04d.png -i input.mp4 -map 0:v -map 1:a -c:v libx264 -preset slow -crf 19 -bf 0 -g 24 -pix_fmt yuv420p -c:a aac -b:a 128k -movflags +faststart -shortest plate.mp4',shell=True,check=True)
print('plate KB',os.path.getsize('plate.mp4')//1024,'ink',inks[0],inks[-1])
