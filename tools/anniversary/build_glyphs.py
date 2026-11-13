import cv2, numpy as np, json, base64, os
from common import *
rot=level(cv2.imread('P.png')); cv2.imwrite('rotP.png',rot); gray=cv2.cvtColor(rot,cv2.COLOR_BGR2GRAY).astype(np.float32)
SC=3; PAD=3; VIS=0.5
BOOST=float(os.environ.get('BOOST','1.25')); THR=float(os.environ.get('THR','0.3'))
bands={1:(493,544),2:(548,600),3:(604,656),4:(658,712),5:(714,768)}
segs={int(k):v for k,v in json.load(open('segs.json')).items()}
labels={1:['S','A','N','TO','S','H','J','I','S','U','BH','A','S','H','J','I','C','H','H','AJ','E','R',','],
        2:['W','I','S','H','I','N','G','Y','o','U','A','TR','U','LY','S','P','E','C','I','A','L'],
        3:['C','o','U','P','LE','D','Ay','Lo','V','e','W','I','th','L','o','Ve','A','N','D'],
        4:['F','U','L','L','o','F','C','H','E','R','I','S','H','E','d','M','o','M','e','N','TS',','],
        5:['H','AP','P','I','eS','T','A','N','N','I','V','E','RS','A','RY']}
USE=set('ABCDEFGHIJLMNOPRSTUVWY,')|{'o'}        # 'o' is scaled up to stand in for O; e/d/y/t/h lowercase forms skipped
DESC=set('JY,')
def alpha_of(x0,x1,y0,y1):
    g=gray[y0:y1,x0:x1]; paper=np.percentile(gray[y0:y1,max(0,x0-30):x1+30],80); ink=np.percentile(gray[y0:y1,max(0,x0-60):x1+60],1)
    return np.clip((paper-g)/(paper-ink),0,1)
def clean(a,c):
    b=(a>0.4).astype(np.uint8); n,lab,st,_=cv2.connectedComponentsWithStats(b)
    if n<2: return None
    big=1+np.argmax(st[1:,cv2.CC_STAT_AREA]); bx,by,bw,bh,_=st[big]; keep=np.zeros_like(b)
    for i in range(1,n):
        x,y,w,h,area=st[i]
        if i==big: keep[lab==i]=1; continue
        if area<4: continue
        if c=='I' and y+h<=by+2 and abs((x+w/2)-(bx+bw/2))<6: keep[lab==i]=1      # the dot of the i
        elif c!='I' and area>=12 and y+h>by+3 and y<by+bh: keep[lab==i]=1
    keep=cv2.dilate(keep,np.ones((3,3),np.uint8)); a=a*keep
    ys,xs=np.where(a>0.15)
    return a[ys.min():ys.max()+1,xs.min():xs.max()+1], ys.min()
raw={}
for ln,(y0,y1) in bands.items():
    ex=[]
    for (x0,x1),lab in zip(segs[ln],labels[ln]):
        x0-=1; x1+=1
        a=alpha_of(x0,x1,y0,y1)
        parts=[]
        if len(lab)==1: parts=[(lab,a)]
        else:   # split a merged pair: by connected components if separable, else at the emptiest column
            b=(a>0.4).astype(np.uint8); n,cl,st,cen=cv2.connectedComponentsWithStats(b)
            comps=[i for i in range(1,n) if st[i,cv2.CC_STAT_AREA]>=20]
            if len(comps)>=2:
                comps.sort(key=lambda i:cen[i][0]); L=np.isin(cl,[comps[0]]); Rr=np.isin(cl,comps[1:])
                # assign leftover small bits by x
                mL=cv2.dilate(L.astype(np.uint8),np.ones((3,3),np.uint8)); mR=cv2.dilate(Rr.astype(np.uint8),np.ones((3,3),np.uint8))
                parts=[(lab[0],a*mL),(lab[1],a*mR)]
            else:
                col=(a>0.4).sum(0).astype(float); w=len(col); lo,hi=int(w*0.3),int(w*0.7); cut=lo+int(np.argmin(col[lo:hi]))
                aL=a.copy(); aL[:,cut:]=0; aR=a.copy(); aR[:,:cut]=0; parts=[(lab[0],aL),(lab[1],aR)]
        for c,pa in parts:
            r=clean(pa,'I' if c=='I' else c)
            if r is None: continue
            g,ytop=r; vy,vx=np.where(g>VIS)
            body=g>VIS
            ex.append(dict(c=c,a=g,vt=int(vy.min()),vb=int(vy.max()),vl=int(vx.min()),vr=int(vx.max()),y0=y0+ytop,ln=ln))
    for k,e in enumerate(ex):
        e['desc']=0
        if e['c'] in DESC or e['c']=='y':
            nb=[ex[j]['vb']+ex[j]['y0'] for j in (k-1,k+1,k-2,k+2) if 0<=j<len(ex) and ex[j]['c'] not in DESC and ex[j]['c']!='y']
            e['desc']=max(0,(e['vb']+e['y0'])-float(np.median(nb[:2])))
        if e['c'] in USE: raw.setdefault(e['c'].upper() if e['c']=='o' else e['c'],[]).append(e)
# heights of line-1 letters: small caps (non-initial) vs initials
l1=[e for c in raw for e in raw[c] if e['ln']==1 and c not in DESC and c!='I']
hs=sorted(e['vb']-e['vt']+1 for e in l1); print('line-1 letter heights',hs)
TARGET=float(np.median(hs)); print('TARGET',TARGET)
def finish(e,c):
    a=e['a']; above=e['vb']-e['desc']-e['vt']+1
    tgt=TARGET*(1.08 if c=='I' else 1)
    s=tgt/above if c!=',' else 1.0
    a=cv2.resize(a,None,fx=s,fy=s,interpolation=cv2.INTER_AREA if s<1 else cv2.INTER_CUBIC); a=np.clip(a,0,1)
    a3=cv2.resize(a,(a.shape[1]*SC,a.shape[0]*SC),interpolation=cv2.INTER_CUBIC)
    thr=THR+(0.22 if c=='M' else 0)   # the note's own M is drawn heavy; thin it to sit with the rest
    a3=np.clip((a3*BOOST-thr)/(1-thr),0,1); a3=cv2.GaussianBlur(a3,(5,5),0.6)
    a3=cv2.copyMakeBorder(a3,PAD,PAD,PAD,PAD,cv2.BORDER_CONSTANT,0)
    return dict(a=a3, base=round(PAD+(e['vb']-e['desc']+1)*s*SC,1), vl=round(PAD+e['vl']*s*SC,1), vr=round(PAD+(e['vr']+1)*s*SC,1), ln=e['ln'], s=round(s,2))
glyphs={}
for c,lst in raw.items():
    lst.sort(key=lambda e:(e['ln']!=1, abs((e['vb']-e['desc']-e['vt']+1)-TARGET)))   # line-1 samples first, then closest in size
    glyphs[c]=[finish(e,c) for e in lst]
# measure stroke width for synthetic letters (K Q X Z)
sw=2.0; rng=np.random.default_rng(11)
SYN={'K':([[(0.05,0.0),(0.03,1.0)],[(0.92,0.02),(0.08,0.55)],[(0.3,0.42),(0.97,1.0)]],0.62),
     'Q':([[(0.5,0.0),(0.18,0.1),(0.02,0.42),(0.08,0.8),(0.35,1.0),(0.68,0.98),(0.93,0.75),(0.98,0.38),(0.82,0.08),(0.5,0.0)],[(0.55,0.72),(1.02,1.08)]],0.78),
     'X':([[(0.02,0.0),(0.98,1.0)],[(0.95,0.0),(0.05,1.0)]],0.7),
     'Z':([[(0.03,0.02),(0.95,0.0),(0.05,1.0),(0.98,0.98)]],0.68)}
UP=8
for c,(strokes,aspect) in SYN.items():
    vs=[]
    for v in range(2):
        hh=TARGET; ww=TARGET*aspect; pad=4
        canvas=np.zeros((int((hh+2*pad)*UP),int((ww+2*pad)*UP)),np.uint8)
        for st in strokes:
            pts=np.array([[(pad+(x+rng.normal(0,0.015))*ww)*UP,(pad+(y+rng.normal(0,0.015))*hh)*UP] for x,y in st],np.float32)
            if len(pts)>4:
                for _ in range(3):
                    q=0.75*pts[:-1]+0.25*pts[1:]; r=0.25*pts[:-1]+0.75*pts[1:]; pts=np.vstack([pts[:1],np.column_stack([q,r]).reshape(-1,2),pts[-1:]])
            cv2.polylines(canvas,[pts.astype(np.int32)],False,255,int(sw*UP),cv2.LINE_AA)
        a=cv2.GaussianBlur(cv2.resize(canvas.astype(np.float32)/255,(int(ww+2*pad),int(hh+2*pad)),interpolation=cv2.INTER_AREA),(3,3),0.5)
        ys,xs=np.where(a>0.15); a=a[ys.min():ys.max()+1,xs.min():xs.max()+1]; vy,vx=np.where(a>VIS)
        vb=vy.max() if c!='Q' else int(vy.min()+TARGET-1)
        vs.append(finish(dict(a=a,vt=vy.min(),vb=vb,vl=vx.min(),vr=vx.max(),desc=0,ln=0),c))
    glyphs[c]=vs
print({c:len(v) for c,v in sorted(glyphs.items())})
# preview: every variant of every letter on a baseline
rows=[]
for c in sorted(glyphs):
    canvas=np.zeros((50*SC,900),np.float32); X=40; BASE=34*SC
    for k,g in enumerate(glyphs[c]):
        a=g['a']; h,w=a.shape; top=int(round(BASE-g['base'])); left=int(round(X-g['vl']))
        if top<0 or left+w>900: continue
        hh=min(h,canvas.shape[0]-top); canvas[top:top+hh,left:left+w]=np.maximum(canvas[top:top+hh,left:left+w],a[:hh])
        lab8=np.zeros((16,40),np.uint8); cv2.putText(lab8,f"{k}:{g['ln']}",(0,12),cv2.FONT_HERSHEY_SIMPLEX,0.4,150,1); canvas[0:16,left:left+40]=np.maximum(canvas[0:16,left:left+40],lab8[:, :max(0,min(40,900-left))].astype(np.float32)/255) if left+40<=900 else canvas[0:16,left:left+40]; X+=int(g['vr']-g['vl'])+26
    cv2.line(canvas,(0,BASE),(900,BASE),0.2,1); rows.append(canvas)
half=(len(rows)+1)//2; A=np.vstack(rows[:half]); B=np.vstack(rows[half:]); B=np.pad(B,((0,A.shape[0]-B.shape[0]),(0,0)))
cv2.imwrite('alphabet.png',(255-np.hstack([A,B])*255).astype(np.uint8))
import pickle; pickle.dump(glyphs,open('glyphs_all.pkl','wb')); json.dump({'TARGET':TARGET},open('gmeta.json','w'))
