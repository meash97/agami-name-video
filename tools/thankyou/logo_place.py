import cv2, numpy as np, sys, json
sys.path.insert(0,'/home/claude/t4'); from common import *
A3=np.load('A_logo.npy'); A=A3[:,:2]; C=A3[:,2]
lg=cv2.imread('logo.png').astype(np.float32); LH,LW=lg.shape[:2]
bgc=np.median(lg[5:40,5:200].reshape(-1,3),axis=0)
cont=(np.abs(lg-bgc).max(2)>22).astype(np.uint8)
ys,xs=np.where(cont[:625]); mc=np.array([(xs.min()+xs.max())/2,(ys.min()+ys.max())/2]); mr=(xs.max()-xs.min())/2
print('logo medallion centre',mc,'radius',mr)
def ang_of(v): return np.degrees(np.arctan2(v[1],v[0]))
best=None
for psi in np.linspace(-0.3,0.3,601):
    R=np.array([[np.cos(psi),-np.sin(psi)],[np.sin(psi),np.cos(psi)]])
    a=ang_of(A@R@[1,0])
    if best is None or abs(a+13.4)<best[0]: best=(abs(a+13.4),psi)
psi=best[1]; R=np.array([[np.cos(psi),-np.sin(psi)],[np.sin(psi),np.cos(psi)]]); print('psi deg',np.degrees(psi).round(2))
ref=fr(241); line1=(inkmap(ref)>0.35).astype(np.uint8); line1[:470]=0; line1[700:]=0
L1=np.zeros_like(line1); lv=level(line1.astype(np.float32)); lv[:540]=0; lv[600:]=0; L1=(unlevel(lv)>0.3).astype(np.uint8)
dist=cv2.distanceTransform(1-L1,cv2.DIST_L2,5)
cy,cx=np.where(cont>0); P=np.stack([cx,cy],1).astype(np.float32)[::7]
def mapping(s,o):
    # logo px -> ref px
    Mq=A@R*(s/mr)                       # 2x2
    t=C+A@R@np.asarray(o)-Mq@mc
    return np.hstack([Mq,t[:,None]])
def score(s,o):
    Mm=mapping(s,o); q=(Mm[:,:2]@P.T+Mm[:,2:]).T
    if q[:,1].min()<70 or q[:,0].min()<12 or q[:,0].max()>705: return None
    qi=np.clip(np.round(q).astype(int),[0,0],[W-1,H-1]); d=dist[qi[:,1],qi[:,0]].min()
    return d
sol=None
for s in np.arange(1.0,0.6,-0.01):
    for oy in np.arange(0,-0.8,-0.02):
        for ox in (0,-0.05,0.05,-0.1,0.1):
            d=score(s,(ox,oy))
            if d is not None and d>=16: sol=(s,ox,oy,d); break
        if sol: break
    if sol: break
print('placement s=%.2f ox=%.2f oy=%.2f clearance=%.1f px'%sol)
Mm=mapping(sol[0],(sol[1],sol[2])); np.save('M_logo.npy',Mm); json.dump({'s':sol[0],'ox':sol[1],'oy':sol[2]},open('logo_place.json','w'))
