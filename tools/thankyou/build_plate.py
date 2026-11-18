# Thank-you template: erase the name on line 1 (written live at frames ~51-95), fix "Agimi" -> "Agami",
# keep hand + shadow, carry a hand-matte strip, record motion, ink colour and the name's write-on progress.
import cv2, numpy as np, json, os, subprocess
from common import *
NF=241
Pf=fr(NF).astype(np.float32); rotP=level(fr(NF)); gray=cv2.cvtColor(rotP,cv2.COLOR_BGR2GRAY).astype(np.float32)
def alpha_box(x0,x1,y0,y1,minarea=4):
    g=gray[y0:y1,x0:x1]; paper=np.percentile(gray[y0:y1,max(0,x0-30):x1+30],80); ink=np.percentile(gray[y0:y1,max(0,x0-60):x1+60],1)
    a=np.clip((paper-g)/(paper-ink),0,1); b=(a>0.35).astype(np.uint8); n,lab,st,_=cv2.connectedComponentsWithStats(b); k=np.zeros_like(b)
    for i in range(1,n):
        if st[i,cv2.CC_STAT_AREA]>=minarea: k[lab==i]=1
    a=a*cv2.dilate(k,np.ones((3,3),np.uint8)); full=np.zeros((H,W),np.float32); full[y0:y1,x0:x1]=a; return full
# ---- pieces (level space)
import agami
name=np.maximum(alpha_box(232,552,604,662), agami.alpha_box(366,388,604,680,keep_big=True))   # "SANTOSH Ji CHHAJAR." (+ J descender)
i1,m_,i2,dot,A_,SHIFT=agami.i1,agami.m_,agami.i2,agami.dot,agami.A_,agami.SHIFT
i2s=i2.copy(); i2s[:806]=0; i2d=i2.copy(); i2d[806:]=0
def shiftL(a,d): return cv2.warpAffine(a,np.float32([[1,0,d],[0,1,0]]),(W,H))
# ---- frame-space masks
def F(a): return unlevel(a)
def blob_of(a):
    e=cv2.dilate((a>0.05).astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    return cv2.dilate((F(e.astype(np.float32))>0.3).astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))).astype(np.float32)
# each original piece is erased only from the frame it starts being written (no erasing of blank paper in deep shadow)
BLOBS=[(blob_of(name),49)]
SMALL=[(blob_of(i1),180),(blob_of(m_),183),(blob_of(i2),195),(blob_of(dot),207)]   # Agimi pieces: filled from their surroundings (keeps the pen shadow)
blob0=np.maximum.reduce([b for b,_ in BLOBS+SMALL])
sy_,sx_=np.where(np.maximum.reduce([b for b,_ in SMALL])>0); SY0,SY1,SX0,SX1=sy_.min()-12,sy_.max()+12,sx_.min()-12,sx_.max()+12
# texture source: blank paper 260 px lower on the (level) final page, relit per frame
dv=Minv[:,:2]@np.array([0,260],np.float32); tex=cv2.warpAffine(Pf,np.float32([[1,0,-dv[0]],[0,1,-dv[1]]]),(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
def inkm(img): return cv2.dilate((inkmap(np.clip(img,0,255).astype(np.uint8))>0.15).astype(np.uint8),np.ones((7,7),np.uint8)).astype(np.float32)
ys,xs=np.where(blob0>0); Y0,Y1,X0,X1=max(0,ys.min()-40),min(H,ys.max()+40),max(0,xs.min()-40),min(W,xs.max()+40)
texR=tex[Y0:Y1,X0:X1]; lowT=lowpass(texR,inkm(texR))
# ---- hand matte: colour change (skin, gold nib) or very dark (pen body); shadows only darken, so they stay out
Pn=Pf/(Pf.sum(2,keepdims=True)+1); Ps=Pf.sum(2)+1
FOOT=None   # frame-space footprint of the old ink (set below); thin strokes there are ink unless they don't fit ink-on-paper
def hand_parts(f):
    ch=np.abs(f/(f.sum(2,keepdims=True)+1)-Pn).sum(2); lr=(f.sum(2)+1)/Ps
    md=cv2.morphologyEx((lr<0.30).astype(np.uint8),cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))   # thick dark = pen barrel
    mc=(ch>0.07).astype(np.uint8)
    mcb=cv2.morphologyEx(mc,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))                          # skin, nib body
    big=np.maximum(md,mcb); n,lab,st,_=cv2.connectedComponentsWithStats(big); k=np.zeros_like(big)
    for j in range(1,n):
        if st[j,cv2.CC_STAT_AREA]>=300: k[lab==j]=1
    k=cv2.morphologyEx(k,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    if FOOT is not None: k[(FOOT>0)&(cv2.erode(k,np.ones((5,5),np.uint8))==0)]=0   # closing must not swallow letters at the hand's edge
    k=cv2.dilate(k,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    thin=cv2.morphologyEx(mc,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))&(1-k)&(cv2.dilate(k,np.ones((15,15),np.uint8)))      # thin bits touching the hand (nib tip)
    if FOOT is None: return k.astype(np.float32), thin.astype(np.float32), np.zeros_like(k,np.float32)
    tin=thin&(FOOT>0); tout=thin&(FOOT==0)
    return np.maximum(k,tout).astype(np.float32), tin.astype(np.float32), k.astype(np.float32)
def hand_matte(f): return hand_parts(f)[0]
def refine(hm,tin,f,relit_full,ink_bgr):
    """thin pixels on old ink: hand only if they can't be explained as paper*(shade)*(ink at some alpha)"""
    ys,xs=np.where(tin>0)
    if len(ys)==0: return hm
    q=f[ys,xs]/np.maximum(relit_full[ys,xs],1); k=np.asarray(ink_bgr,np.float32)
    best=np.full(len(ys),9.0)
    for t in np.linspace(0,1,9):
        base=1-t*(1-k); s_=np.clip(q.mean(1)/base.mean(),0.25,1.15); pred=s_[:,None]*base[None,:]
        r=np.abs(q-pred).max(1)/np.maximum(pred.max(1),0.05); best=np.minimum(best,r)
    out=hm.copy(); h=best>0.22; out[ys[h],xs[h]]=1; return out
# ---- motion (track "DEAR"; ~1 px drift)
refink=inkmap(fr(NF)); c=to_frame([[134,606],[222,606],[222,650],[134,650]])
tx0,ty0,tx1,ty1=int(c[:,0].min()),int(c[:,1].min()),int(c[:,0].max())+1,int(c[:,1].max())+1
tU=cv2.resize(refink[ty0:ty1,tx0:tx1],None,fx=8,fy=8,interpolation=cv2.INTER_CUBIC); mot=[]
for i in range(1,NF+1):
    if i<50: mot.append(None); continue
    mm=inkmap(fr(i)); s=cv2.resize(mm[ty0-10:ty1+10,tx0-10:tx1+10],None,fx=8,fy=8,interpolation=cv2.INTER_CUBIC)
    r=cv2.matchTemplate(s,tU,cv2.TM_CCOEFF_NORMED); _,mx,_,loc=cv2.minMaxLoc(r); mot.append((loc[0]/8-10,loc[1]/8-10))
first=next(m for m in mot if m is not None); mot=[m if m is not None else first for m in mot]
u=np.array(mot); motion=[[round(float(np.median(u[max(0,k-2):k+3,0])),3),round(float(np.median(u[max(0,k-2):k+3,1])),3)] for k in range(NF)]
json.dump(motion,open('motion.json','w'))
# ---- per-frame render
REG=dict(x=225,y=596,w=395,h=80)                       # name band (level) == matte strip size
nameF=F(name); core=(nameF>0.6)
FOOT0=cv2.dilate(((nameF>0.08)|(np.maximum.reduce([F(a) for a in (i1,m_,i2,dot)])>0.08)).astype(np.uint8),np.ones((3,3),np.uint8))
ring=(cv2.dilate(blob0,np.ones((25,25),np.uint8))>0)&(blob0==0)&(refink<0.08)
# write-on progress of the name: per level column, first frame at which its ink has (and keeps) arrived
cols=np.arange(REG['x'],REG['x']+REG['w']); nm_cols=name[604:680,REG['x']:REG['x']+REG['w']].sum(0)
seen=np.zeros(len(cols),np.float32); col_frame=np.full(len(cols),NF,np.int32)
os.makedirs('plate',exist_ok=True); inks=[]; last=None; prog=[]
for i in range(1,NF+1):
    f=fr(i).astype(np.float32); mv=motion[i-1]; sh=np.float32([[1,0,mv[0]],[0,1,mv[1]]])
    FOOT=cv2.warpAffine(FOOT0,sh,(W,H))
    hm,tin,hbig=hand_parts(f); hmd=cv2.dilate(hm,np.ones((7,7),np.uint8))
    # ink colour from DEAR (always visible once written), else the final page
    dm=(F(alpha_box(134,222,606,652))>0.6)&(hm==0)
    if i>=50 and dm.sum()>200:
        px=f[dm]; lum=px.mean(1); ic=np.median(px[lum<=np.percentile(lum,30)],axis=0)
        rg=ring&(hmd==0); pp=f[rg]; pl=pp.mean(1); pc=np.median(pp[pl>=np.percentile(pl,40)],axis=0)
        last=[round(float(v),3) for v in (ic/pc)[::-1]]
    inks.append(last)
    # progress of the original name (level-space columns, ignoring hand-covered spots)
    lv=inkmap(level(fr(i)))[604:680,REG['x']:REG['x']+REG['w']]; hl=level(hm)[604:680,REG['x']:REG['x']+REG['w']]>0.5
    have=(np.minimum(lv,name[604:680,REG['x']:REG['x']+REG['w']])*(~hl)).sum(0); tot=(name[604:680,REG['x']:REG['x']+REG['w']]*(~hl)).sum(0)
    done=(tot>3)&(have>0.6*tot); newly=done&(col_frame==NF); col_frame[newly]=i
    # erase originals (never under the hand)
    blob=cv2.warpAffine(np.maximum.reduce([b*(i>=s0) for b,s0 in BLOBS]),sh,(W,H)); roi=f[Y0:Y1,X0:X1]
    msk=np.maximum(np.maximum(blob[Y0:Y1,X0:X1],inkm(roi)),hmd[Y0:Y1,X0:X1])
    lowF=lowpass(roi,msk); sup=cv2.GaussianBlur(1-msk,(0,0),8)
    lowF2=lowpass(roi,msk,20); wgt=np.clip((sup-0.15)/0.35,0,1)[...,None]
    lowF=lowF*wgt+lowF2*(1-wgt)                        # steadier lighting estimate right beside the hand
    relit=texR*np.clip(lowF/np.maximum(lowT,1),0.35,2.5)
    rf=f.copy(); rf[Y0:Y1,X0:X1]=relit
    hm=refine(hm,tin,f,rf,np.array(last if last else [0.24,0.11,0.16],np.float32)[::-1])
    fe=(np.minimum(cv2.GaussianBlur(blob,(0,0),1.5),1)*(1-np.minimum(cv2.GaussianBlur(hm,(0,0),1.0),1)))[Y0:Y1,X0:X1][...,None]
    res=f.copy(); res[Y0:Y1,X0:X1]=roi*(1-fe)+relit*fe
    sm=np.maximum.reduce([b*(i>=s0) for b,s0 in SMALL])
    if sm.any():
        sm=(cv2.warpAffine(sm,sh,(W,H))>0)
        hh_=cv2.dilate(hm,np.ones((5,5),np.uint8))>0          # hand/nib: not a fill source, and restored afterwards
        sub=np.clip(res[SY0:SY1,SX0:SX1],0,255).astype(np.uint8)
        filled=cv2.inpaint(sub,(sm|hh_)[SY0:SY1,SX0:SX1].astype(np.uint8),4,cv2.INPAINT_TELEA).astype(np.float32)
        keep=(hm[SY0:SY1,SX0:SX1]>0)[...,None]
        res[SY0:SY1,SX0:SX1]=np.where(keep,res[SY0:SY1,SX0:SX1],np.where(sm[SY0:SY1,SX0:SX1,None],filled,res[SY0:SY1,SX0:SX1]))
    r=np.array(last if last else [0.25,0.2,0.2],np.float32)[::-1]
    hide=1-np.minimum(cv2.GaussianBlur(hm,(0,0),1.0),1)
    def ramp(a,b): return float(np.clip((i-a+1)/(b-a+1),0,1))
    def wipe(lay_lv,a,b):     # reveal left->right in level x between frames a..b
        t=ramp(a,b)
        if t<=0: return None
        if t>=1: return lay_lv
        xs=np.where(lay_lv.max(0)>0.05)[0]; xc=xs.min()+t*(xs.max()-xs.min()+1)
        g=np.clip((xc-np.arange(W))/2.0+0.5,0,1)[None,:]; return lay_lv*g
    LAY=[]
    if ramp(181,183)>0: LAY.append(A_*ramp(181,183))
    w=wipe(shiftL(m_,SHIFT),184,193);   LAY+= [w] if w is not None else []
    if ramp(196,198)>0: LAY.append(shiftL(i2s,SHIFT)*ramp(196,198))
    if ramp(199,200)>0: LAY.append(shiftL(i2d,SHIFT)*ramp(199,200))
    if ramp(209,210)>0: LAY.append(shiftL(dot,SHIFT)*ramp(209,210))
    for lay in LAY:
        al=(np.clip(F(lay),0,1)*hide)[...,None]; res=res*(1-al)+res*r*al
    # matte strip (level space, pre-shifted by motion)
    hm_s=cv2.warpAffine(hm,np.float32([[1,0,-mv[0]],[0,1,-mv[1]]]),(W,H))
    strip=cv2.warpAffine(hm_s,M,(REG['x']+REG['w'],REG['y']+REG['h']),flags=cv2.INTER_LINEAR)[REG['y']:,REG['x']:]
    out=np.zeros((H+8+REG['h'],W,3),np.uint8); out[:H]=np.clip(res,0,255).astype(np.uint8); out[H+8:,:REG['w']]=(np.clip(strip,0,1)*255).astype(np.uint8)[...,None]
    cv2.imwrite(f'plate/{i:04d}.png',out)
# write-on curve of the original name (level x of the writing frontier), read off the frames
KEY=[(50,236),(53,357),(57,369),(61,397),(65,412),(81,531),(95,531),(97,547)]
fx=np.interp(np.arange(1,NF+1),[k for k,_ in KEY],[x for _,x in KEY],left=236,right=547)
fr_prog=[round(float((x-236)/(547-236)),4) for x in fx]
json.dump(fr_prog,open('progress.json','w')); json.dump(inks,open('inks.json','w')); json.dump(REG,open('reg.json','w'))
subprocess.run('ffmpeg -v error -y -framerate 24 -i plate/%04d.png -i input.mp4 -map 0:v -map 1:a -c:v libx264 -preset slow -crf 19 -bf 0 -g 24 -pix_fmt yuv420p -c:a aac -b:a 128k -movflags +faststart -shortest plate.mp4',shell=True,check=True)
print('plate KB',os.path.getsize('plate.mp4')//1024,'ink',inks[60],inks[-1])
