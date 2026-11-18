# Thank You (Agami, printed) template: new logo in every frame, old name erased as it appears,
# hand kept in front (matte strip), per-frame camera affine, ink colour and reveal curve.
import cv2, numpy as np, json, os, subprocess, sys
sys.path.insert(0,'/home/claude/t4'); from common import *
NF=241
ref=fr(NF).astype(np.float32); rotP=level(fr(NF)); gray=cv2.cvtColor(rotP,cv2.COLOR_BGR2GRAY).astype(np.float32)
logo_plate=cv2.imread('logo_plate.png').astype(np.float32); logo_area=np.load('logo_area.npy')
segs=json.load(open('segs.json'))['1']
# old name pieces (line 1 from "SANTOSH"), with the frame each starts to appear
LETTERS=segs[4:]            # S A N TO S H J U C H AA J A R ,
STARTS=[1,1,1,1,1,1, 1,5,15,25,33, 86,92,98,116]
def alpha_box(x0,x1,y0=546,y1=592):
    g=gray[y0:y1,x0:x1]; paper=np.percentile(gray[y0:y1,max(0,x0-40):x1+40],80); ink=np.percentile(gray[546:826,100:560],0.3)
    a=np.clip((paper-g)/(paper-ink),0,1); b=(a>0.3).astype(np.uint8); n,lab,st,_=cv2.connectedComponentsWithStats(b); k=np.zeros_like(b)
    for i in range(1,n):
        if st[i,4]>=4: k[lab==i]=1
    a=a*cv2.dilate(k,np.ones((3,3),np.uint8)); full=np.zeros((H,W),np.float32); full[y0:y1,x0:x1]=a; return full
def blob_of(a_lv):
    e=cv2.dilate((a_lv>0.04).astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    return cv2.dilate((unlevel(e.astype(np.float32))>0.3).astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))).astype(np.float32)
PIECES=[(blob_of(alpha_box(x0-2,x1+2)),s0) for (x0,x1),s0 in zip(LETTERS,STARTS)]
name_lv=np.maximum.reduce([alpha_box(x0-2,x1+2) for x0,x1 in LETTERS]); nameR=unlevel(name_lv)
FOOT=cv2.dilate((nameR>0.06).astype(np.uint8),np.ones((3,3),np.uint8))
allblob=np.maximum.reduce([b for b,_ in PIECES])
ys,xs=np.where(allblob>0); Y0,Y1,X0,X1=max(0,ys.min()-40),min(H,ys.max()+40),max(0,xs.min()-40),min(W,xs.max()+40)
def inkm(img): return cv2.dilate((inkmap(np.clip(img,0,255).astype(np.uint8),40)>0.2).astype(np.uint8),np.ones((7,7),np.uint8)).astype(np.float32)
# paper texture (relative) from blank paper right of the logo, in ref space
SRC=np.float32([[1,0,0],[0,1,-170]])
src=cv2.warpAffine(ref,SRC,(W,H),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REFLECT)
sm=inkm(src); lowS=lowpass(src,sm,6); hp=np.clip((src-lowS)/np.maximum(lowS,1),-0.08,0.08); hp[cv2.dilate(sm,np.ones((9,9),np.uint8))>0]=0
hpR=hp[Y0:Y1,X0:X1]
# ---- hand matte (computed in ref space on the stabilised frame)
Pn=ref/(ref.sum(2,keepdims=True)+1); Ps=ref.sum(2)+1
def hand_parts(f):
    ch=np.abs(f/(f.sum(2,keepdims=True)+1)-Pn).sum(2); lr=(f.sum(2)+1)/Ps
    md=cv2.morphologyEx((lr<0.32).astype(np.uint8),cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    mc=(ch>0.06).astype(np.uint8)
    mcb=cv2.morphologyEx(mc,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    big=np.maximum(md,mcb); n,lab,st,_=cv2.connectedComponentsWithStats(big); k=np.zeros_like(big)
    for j in range(1,n):
        if st[j,4]>=400: k[lab==j]=1
    k=cv2.morphologyEx(k,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))
    k[(FOOT>0)&(cv2.erode(k,np.ones((5,5),np.uint8))==0)]=0
    k=cv2.dilate(k,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    return k.astype(np.float32)
# logo region gain ring (for exposure changes)
ring_logo=(cv2.dilate(logo_area,np.ones((31,31),np.uint8))>0)&(logo_area==0)&(inkmap(fr(NF))<0.1)
ring_name=(cv2.dilate(allblob,np.ones((25,25),np.uint8))>0)&(allblob==0)&(inkmap(fr(NF))<0.08)
dear=unlevel(alpha_box(130,230))>0.6
REG=dict(x=225,y=538,w=445,h=64)
os.makedirs('plate',exist_ok=True); inks=[]; last=None
modmask=np.maximum(logo_area,allblob)
modF=cv2.dilate((modmask>0).astype(np.uint8),np.ones((5,5),np.uint8)).astype(np.float32)
for i in range(1,NF+1):
    f0=fr(i).astype(np.float32); f=to_ref(f0,i)                       # stabilised to the reference
    hm=hand_parts(f); hmd=cv2.dilate(hm,np.ones((7,7),np.uint8)); keepf=1-np.minimum(cv2.GaussianBlur(hm,(0,0),1.0),1)
    # ink colour from DEAR
    dm=dear&(hm==0)
    if dm.sum()>150:
        px=f[dm]; lum=px.mean(1); ic=np.median(px[lum<=np.percentile(lum,30)],axis=0)
        rg=ring_name&(hmd==0); pp=f[rg]; pl=pp.mean(1); pc=np.median(pp[pl>=np.percentile(pl,40)],axis=0)
        last=[round(float(v),3) for v in (ic/pc)[::-1]]
    inks.append(last)
    res=f.copy()
    # logo: replace with the new logo, matched to this frame's exposure, behind the hand
    rg=ring_logo&(hmd==0)
    gain=np.median(f[rg],axis=0)/np.maximum(np.median(ref[rg],axis=0),1) if rg.sum()>200 else np.ones(3)
    # anything near the hand that doesn't match the clean page here is part of the hand (nails, highlights)
    near=cv2.dilate(hm,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(41,41)))>0
    fb=cv2.GaussianBlur(f,(0,0),1.0); rb=cv2.GaussianBlur(ref*gain,(0,0),1.0)
    k5=np.ones((5,5),np.uint8)
    dif=np.maximum(fb-cv2.dilate(rb,k5),cv2.erode(rb,k5)-fb).max(2)      # tolerant of 1-2 px misalignment
    occ=((dif>16)&near).astype(np.uint8)
    occ=cv2.morphologyEx(occ,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    occ=cv2.dilate(np.maximum(occ,hm.astype(np.uint8)),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))).astype(np.float32)
    keepL=1-np.minimum(cv2.GaussianBlur(occ,(0,0),1.2),1)
    la=(np.minimum(cv2.GaussianBlur(logo_area,(0,0),1.5),1)*keepL)[...,None]
    # local light changes (hand shadow / bounce light) measured around the logo, spread smoothly over it
    valid=((cv2.dilate(logo_area,np.ones((61,61),np.uint8))>0)&(logo_area==0)&(occ==0)&(inkmap(fr(NF))<0.1)).astype(np.float32)
    rat=np.clip(f/np.maximum(ref,1),0.3,2.0)
    num=cv2.GaussianBlur(rat*valid[...,None],(0,0),28); den=cv2.GaussianBlur(valid,(0,0),28)[...,None]
    num2=cv2.GaussianBlur(rat*valid[...,None],(0,0),80); den2=cv2.GaussianBlur(valid,(0,0),80)[...,None]
    w_=np.clip(den/0.15,0,1); lr_=np.where(den>1e-4,num/np.maximum(den,1e-4),1)*w_+np.where(den2>1e-5,num2/np.maximum(den2,1e-5),gain)*(1-w_)
    res=res*(1-la)+(logo_plate*lr_)*la
    # name: erase each original piece once it has started to appear
    blob=np.maximum.reduce([b*(i>=s0) for b,s0 in PIECES])
    roi=res[Y0:Y1,X0:X1]
    msk=np.maximum(np.maximum(blob[Y0:Y1,X0:X1],inkm(roi)),hmd[Y0:Y1,X0:X1])
    lowF=lowpass(roi,msk,8); sup=cv2.GaussianBlur(1-msk,(0,0),8); lowF2=lowpass(roi,msk,20)
    wgt=np.clip((sup-0.15)/0.35,0,1)[...,None]; lowF=lowF*wgt+lowF2*(1-wgt)
    fill=lowF*(1+hpR)
    occN=((dif>16)&near&~(blob>0)).astype(np.uint8)                       # pale skin next to the letters
    occN=np.maximum(occN,((dif>16)&near&(blob>0)&(np.abs(fb-rb).max(2)>40)&(fb.mean(2)>rb.mean(2))).astype(np.uint8))  # brighter than the ink there = finger over it
    occN=cv2.dilate(np.maximum(occN,hm.astype(np.uint8)),np.ones((5,5),np.uint8)).astype(np.float32)
    keepN=1-np.minimum(cv2.GaussianBlur(occN,(0,0),1.2),1)
    fe=(np.minimum(cv2.GaussianBlur(blob,(0,0),1.5),1)*keepN)[Y0:Y1,X0:X1][...,None]
    res[Y0:Y1,X0:X1]=roi*(1-fe)+fill*fe
    # back to this frame's camera, only where something changed
    back=from_ref(res,i); mF=from_ref(modF,i)[...,None]
    out_img=f0*(1-mF)+back*mF
    # matte strip (level space of the reference)
    strip=level(hm,cv2.INTER_LINEAR)[REG['y']:REG['y']+REG['h'],REG['x']:REG['x']+REG['w']]
    if i in (12,24,60): np.save(f'occ_{i}.npy',occ)
    out=np.zeros((H+8+REG['h'],W,3),np.uint8); out[:H]=np.clip(out_img,0,255).astype(np.uint8); out[H+8:,:REG['w']]=(np.clip(strip,0,1)*255).astype(np.uint8)[...,None]
    cv2.imwrite(f'plate/{i:04d}.png',out)
fv=next(v for v in inks if v); inks=[v if v else fv for v in inks]
json.dump(inks,open('inks.json','w')); json.dump(REG,open('reg.json','w'))
# reveal: writing frontier (level x) of the original name, read off the frames
KEY=[(1,398),(7,432),(15,442),(19,475),(27,486),(30,490),(36,515),(42,530),(48,535),(84,541),(90,553),(96,585),(102,593),(108,598),(120,606)]
fx=np.interp(np.arange(1,NF+1),[k for k,_ in KEY],[x for _,x in KEY])
json.dump([round(float((x-239)/(606-239)),4) for x in fx],open('progress.json','w'))
subprocess.run('ffmpeg -v error -y -framerate 24 -i plate/%04d.png -i input.mp4 -map 0:v -map 1:a -c:v libx264 -preset slow -crf 19 -bf 0 -g 24 -pix_fmt yuv420p -c:a aac -b:a 128k -movflags +faststart -shortest plate.mp4',shell=True,check=True)
print('plate KB',os.path.getsize('plate.mp4')//1024,'ink',inks[0],inks[-1])
