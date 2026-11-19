# Thank You (cursive notebook) template.
#  real frames 1..161 : new logo every frame (behind the hand), original name erased, hand matte
#  162..241           : hand slides away, rest of the note (me. / « Warm Regards. / AGAMI / underline / ») writes on, page holds
import cv2, numpy as np, json, os, subprocess, sys
sys.path.insert(0,'/home/claude/t5'); from common import *
NF=241; LAST=161
FIN=cv2.imread('final_page.png').astype(np.float32)          # ref space: new logo + AGAMI fixed + original name
P241=cv2.imread('P241.png').astype(np.float32)                 # ref space, original final frame (lighting reference for the logo)
logo_plate=cv2.imread('logo_plate.png').astype(np.float32); logo_area=np.load('logo_area.npy')
finlv=level(FIN.astype(np.uint8)).astype(np.float32); fg=cv2.cvtColor(finlv.astype(np.uint8),cv2.COLOR_BGR2GRAY).astype(np.float32)
def ink_lv(img_lv): 
    g=cv2.cvtColor(img_lv.astype(np.uint8),cv2.COLOR_BGR2GRAY).astype(np.float32); return np.clip((cv2.GaussianBlur(g,(0,0),12)-g)/45,0,1)
# ---- original name (level): "Santosh Ji Chaajar"
NM=np.zeros((H,W),np.float32); NM[574:664,212:650]=1
nameA=ink_lv(finlv)*NM; nameA[nameA<0.12]=0
b=(nameA>0.3).astype(np.uint8); n,lab,st,_=cv2.connectedComponentsWithStats(b); k=np.zeros_like(b)
for j in range(1,n):
    if st[j,4]>=6: k[lab==j]=1
nameA=nameA*cv2.dilate(k,np.ones((3,3),np.uint8))
name_lv=cv2.dilate((nameA>0.04).astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))).astype(np.float32)
nameR=(unlevel(name_lv)>0.3).astype(np.float32); nameR=cv2.dilate(nameR,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
def inkm(img): return cv2.dilate((inkmap(np.clip(img,0,255).astype(np.uint8),40)>0.2).astype(np.uint8),np.ones((7,7),np.uint8)).astype(np.float32)
def paper_fill(img,mask,tex_dy=-120):
    msk=np.maximum(mask,inkm(img)); low=lowpass(img,msk,8); sup=cv2.GaussianBlur(1-msk,(0,0),8); low2=lowpass(img,msk,22)
    w_=np.clip((sup-0.15)/0.35,0,1)[...,None]; low=low*w_+low2*(1-w_)
    src=cv2.warpAffine(img,np.float32([[1,0,0],[0,1,tex_dy]]),(W,H),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REFLECT)
    sm=inkm(src); ls=lowpass(src,sm,5); hp=np.clip((src-ls)/np.maximum(ls,1),-0.06,0.06); hp[cv2.dilate(sm,np.ones((7,7),np.uint8))>0]=0
    return low*(1+hp)
def fill_into(img,mask):
    fe=np.minimum(cv2.GaussianBlur(mask,(0,0),1.5),1)[...,None]; return img*(1-fe)+paper_fill(img,mask)*fe
BASE=fill_into(FIN,nameR)                                       # finished page without the name
# ---- the text written after the cut (level): rest of "Time." and everything below it
f161=to_ref(fr(LAST),LAST).astype(np.float32); lv161=level(f161.astype(np.uint8)).astype(np.float32)
A=ink_lv(finlv); A161=ink_lv(lv161)
NEW=np.zeros((H,W),np.float32); NEW[742:822,506:640]=1; NEW[826:990,150:640]=1
old161=cv2.dilate((A161>0.2).astype(np.uint8),np.ones((5,5),np.uint8)).astype(np.float32); old161[826:]=0; old161[742:822,514:]=0
newA=A*(1-old161)*NEW; newA[newA<0.1]=0
b=(newA>0.3).astype(np.uint8); n,lab,st,_=cv2.connectedComponentsWithStats(b); k=np.zeros_like(b)
for j in range(1,n):
    if st[j,4]>=8: k[lab==j]=1
newA=newA*cv2.dilate(k,np.ones((3,3),np.uint8))
new_blob=cv2.dilate((newA>0.04).astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))).astype(np.float32)
newR=(unlevel(new_blob)>0.3).astype(np.float32)
BLANK=fill_into(BASE,newR)                                      # page before the closing lines exist
# write-on schedule (level x spans per stroke group, frames)
GROUPS=[('me.',742,822,506,640,162,173),('<<',830,880,150,190,182,184),('Warm',826,888,200,346,185,197),
        ('Regards.',826,900,358,560,199,214),('AGAMI',884,958,256,520,217,230),('line',952,975,150,560,230,236),('>>',940,975,560,640,237,239)]
def written_lv(i):
    out=np.zeros((H,W),np.float32)
    for name,y0,y1,x0,x1,a,bf in GROUPS:
        if i<a: continue
        t=min(1,(i-a+1)/(bf-a+1)); xc=x0+t*(x1-x0+1)
        g=np.clip((xc-np.arange(W))/3.0+0.5,0,1)[None,:]
        m=np.zeros((H,W),np.float32); m[y0:y1,x0:x1]=1
        out=np.maximum(out,m*g)
    return out*np.clip(newA/0.5,0,1)
# ---- hand matte helpers (ref space)
Pn=P241/(P241.sum(2,keepdims=True)+1); Ps=P241.sum(2)+1
FOOT=cv2.dilate((unlevel(nameA)>0.06).astype(np.uint8),np.ones((3,3),np.uint8))
def hand_parts(f):
    ch=np.abs(f/(f.sum(2,keepdims=True)+1)-Pn).sum(2); lr=(f.sum(2)+1)/Ps
    md=cv2.morphologyEx((lr<0.32).astype(np.uint8),cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    mcb=cv2.morphologyEx((ch>0.07).astype(np.uint8),cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    big=np.maximum(md,mcb); n,lab,st,_=cv2.connectedComponentsWithStats(big); k=np.zeros_like(big)
    for j in range(1,n):
        if st[j,4]>=500: k[lab==j]=1
    k=cv2.morphologyEx(k,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))
    k[(FOOT>0)&(cv2.erode(k,np.ones((5,5),np.uint8))==0)]=0
    return cv2.dilate(k,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))).astype(np.float32)
ring_logo=(cv2.dilate(logo_area,np.ones((31,31),np.uint8))>0)&(logo_area==0)&(inkmap(P241.astype(np.uint8))<0.1)
dear=(unlevel((ink_lv(finlv)*(np.pad(np.ones((88,150)),((574,H-662),(40,W-190))))>0.6).astype(np.float32))>0.5)
ring_name=(cv2.dilate(nameR,np.ones((25,25),np.uint8))>0)&(nameR==0)&(inkmap(P241.astype(np.uint8))<0.08)
REG=dict(x=200,y=570,w=500,h=96)
os.makedirs('plate',exist_ok=True); inks=[]; last=None; cams=[]
k5=np.ones((5,5),np.uint8)
def relit_logo(f,occ,gain):
    valid=((cv2.dilate(logo_area,np.ones((61,61),np.uint8))>0)&(logo_area==0)&(occ==0)&(inkmap(P241.astype(np.uint8))<0.1)).astype(np.float32)
    rat=np.clip(f/np.maximum(P241,1),0.3,2.0)
    num=cv2.GaussianBlur(rat*valid[...,None],(0,0),28); den=cv2.GaussianBlur(valid,(0,0),28)[...,None]
    num2=cv2.GaussianBlur(rat*valid[...,None],(0,0),80); den2=cv2.GaussianBlur(valid,(0,0),80)[...,None]
    w_=np.clip(den/0.15,0,1); return np.where(den>1e-4,num/np.maximum(den,1e-4),1)*w_+np.where(den2>1e-5,num2/np.maximum(den2,1e-5),gain)*(1-w_)
# hand layer of frame 161 for the exit: opaque hand/pen + a separate shadow (a darkening factor, no page detail)
hm161=hand_parts(f161)
ch161=np.abs(f161/(f161.sum(2,keepdims=True)+1)-BLANK/(BLANK.sum(2,keepdims=True)+1)).sum(2)
d161=np.abs(cv2.GaussianBlur(f161,(0,0),1.2)-cv2.GaussianBlur(BLANK,(0,0),1.2)).max(2)
txt=(inkmap(np.clip(BLANK,0,255).astype(np.uint8),40)>0.2)
opq=((hm161>0)|((d161>30)&(cv2.dilate(hm161,np.ones((25,25),np.uint8))>0)&~txt)|((ch161>0.1)&(cv2.dilate(hm161,np.ones((25,25),np.uint8))>0))).astype(np.uint8)
opq[:, :420]=0
n,lab,st,_=cv2.connectedComponentsWithStats(opq); OP=np.zeros_like(opq)
for j in range(1,n):
    if st[j,4]>=150: OP[lab==j]=1
OP=cv2.morphologyEx(OP,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
OP[(cv2.dilate(txt.astype(np.uint8),np.ones((3,3),np.uint8))>0)&(ch161<0.12)&(hm161==0)]=0
HLf=np.minimum(cv2.GaussianBlur(OP.astype(np.float32),(0,0),1.0),1); HL=OP
inkb=np.maximum(cv2.dilate(txt.astype(np.uint8),np.ones((5,5),np.uint8)).astype(np.float32),cv2.dilate(OP,np.ones((9,9),np.uint8)).astype(np.float32))
shade=np.clip(lowpass(f161,inkb,6).mean(2)/np.maximum(lowpass(BLANK,inkb,6).mean(2),1),0.45,1.0)
far=cv2.dilate(OP,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(161,161)))==0
shade[far]=1; shade[:, :380]=np.maximum(shade[:, :380],1)
SHADE161=cv2.GaussianBlur(shade,(0,0),3)
HAND161=f161.copy()
SLIDE=[(14,-10),(34,-24),(62,-42),(98,-66),(142,-94),(196,-128),(262,-170),(340,-220),(430,-278)]
for i in range(1,NF+1):
    if i<=LAST:
        f0=fr(i).astype(np.float32); f=to_ref(f0,i)
        hm=hand_parts(f); hmd=cv2.dilate(hm,np.ones((7,7),np.uint8))
        rg=ring_logo&(hmd==0)
        gain=np.median(f[rg],axis=0)/np.maximum(np.median(P241[rg],axis=0),1) if rg.sum()>200 else np.ones(3)
        near=cv2.dilate(hm,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(41,41)))>0
        fb=cv2.GaussianBlur(f,(0,0),1.0); rb=cv2.GaussianBlur(P241*gain,(0,0),1.0)
        dif=np.maximum(fb-cv2.dilate(rb,k5),cv2.erode(rb,k5)-fb).max(2)
        occ=((dif>16)&near).astype(np.uint8); occ=cv2.morphologyEx(occ,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
        occ=cv2.dilate(np.maximum(occ,hm.astype(np.uint8)),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))).astype(np.float32)
        keepL=1-np.minimum(cv2.GaussianBlur(occ,(0,0),1.2),1)
        res=f.copy()
        la=(np.minimum(cv2.GaussianBlur(logo_area,(0,0),1.5),1)*keepL)[...,None]
        res=res*(1-la)+(logo_plate*relit_logo(f,occ,gain))*la
        # erase the original name (all of its footprint; blank paper where not yet written)
        keepN=1-np.minimum(cv2.GaussianBlur(cv2.dilate(hm,k5),(0,0),1.2),1)
        fe=(np.minimum(cv2.GaussianBlur(nameR,(0,0),1.5),1)*keepN)[...,None]
        res=res*(1-fe)+paper_fill(res,np.maximum(nameR,hmd))*fe
        dm=dear&(hm==0)
        if dm.sum()>150:
            px=f[dm]; lum=px.mean(1); ic=np.median(px[lum<=np.percentile(lum,30)],axis=0)
            rgn=ring_name&(hmd==0); pp=f[rgn]; pl=pp.mean(1); pc=np.median(pp[pl>=np.percentile(pl,40)],axis=0)
            last=[round(float(v),3) for v in (ic/pc)[::-1]]
        mod=cv2.dilate(((logo_area>0)|(nameR>0)).astype(np.uint8),k5).astype(np.float32)
        back=from_ref(res,i); mF=from_ref(mod,i)[...,None]; out_img=f0*(1-mF)+back*mF
        cams.append(CAM[i].tolist())
    else:
        k=i-LAST-1
        page=BLANK.copy(); wr=unlevel(written_lv(i))[...,None]; page=page*(1-wr)+BASE*wr
        hm=np.zeros((H,W),np.float32)
        if k<len(SLIDE):
            dx,dy=SLIDE[k]; T_=np.float32([[1,0,dx],[0,1,dy]])
            sh_=cv2.warpAffine(SHADE161,T_,(W,H),borderValue=1.0)[...,None]; page=page*sh_
            hs=cv2.warpAffine(HAND161,T_,(W,H)); ms=cv2.warpAffine(HLf,T_,(W,H))[...,None]
            page=page*(1-ms)+hs*ms; hm=(cv2.warpAffine(HL.astype(np.float32),T_,(W,H))>0.5).astype(np.float32)
        out_img=from_ref(page,LAST); cams.append(CAM[LAST].tolist())
    inks.append(last)
    strip=level(hm,cv2.INTER_LINEAR)[REG['y']:REG['y']+REG['h'],REG['x']:REG['x']+REG['w']]
    out=np.zeros((H+8+REG['h'],W,3),np.uint8); out[:H]=np.clip(out_img,0,255).astype(np.uint8); out[H+8:,:REG['w']]=(np.clip(strip,0,1)*255).astype(np.uint8)[...,None]
    cv2.imwrite(f'plate/{i:04d}.png',out)
fv=next(v for v in inks if v); inks=[v if v else fv for v in inks]
json.dump(inks,open('inks.json','w')); json.dump(REG,open('reg.json','w')); json.dump(cams,open('cams_out.json','w'))
np.save('nameA_lv.npy',nameA)
subprocess.run('ffmpeg -v error -y -framerate 24 -i plate/%04d.png -i input.mp4 -map 0:v -map 1:a -c:v libx264 -preset slow -crf 19 -bf 0 -g 24 -pix_fmt yuv420p -c:a aac -b:a 128k -movflags +faststart -shortest plate.mp4',shell=True,check=True)
print('plate KB',os.path.getsize('plate.mp4')//1024,'ink',inks[5],inks[-1])
