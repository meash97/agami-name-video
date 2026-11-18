import cv2, numpy as np, json, os, pickle
from common import *
rot=level(fr(241)); cv2.imwrite('rotP.png',rot); gray=cv2.cvtColor(rot,cv2.COLOR_BGR2GRAY).astype(np.float32)
SC=3; PAD=3; VIS=0.5
BOOST=float(os.environ.get('BOOST','1.15')); THR=float(os.environ.get('THR','0.3'))
L={(606,676):[('D',136,158),('E',160,178),('R',197,217),('S',236,252),('A',252,268),('N',269,286),('S',322,337),('H',337,357),('J',369,385),('i',386,397),('C',412,431),('H',431,448),('H',448,465),('A',464,482),('R',512,531)],
   (606,656):[('.',537,549)],
   (662,706):[('T',121,138),('A',157,177),('N',178,199),('K',199,218),('F',291,306),('O',306,322),('R',323,342),('V',380,398),('U',459,476),('S',476,491)],
   (706,752):[('L',265,282),('U',283,298),('B',316,329),('L',330,344),('T',379,396),('I',396,408),('M',409,428),('E',430,447)],
   (756,804):[('W',261,287),('M',322,341),('E',371,386),('D',437,453),('S',455,468)],
   (756,820):[('G',388,404)]}
DESC=set('JyG')
def extract(x0,x1,y0,y1,c):
    x0-=1;x1+=1
    g=gray[y0:y1,x0:x1]; paper=np.percentile(gray[y0:y1,max(0,x0-30):x1+30],80); ink=np.percentile(gray[y0:y1,max(0,x0-60):x1+60],1)
    a=np.clip((paper-g)/(paper-ink),0,1)
    b=(a>0.4).astype(np.uint8); n,lab,st,_=cv2.connectedComponentsWithStats(b)
    big=1+np.argmax(st[1:,cv2.CC_STAT_AREA]); bx,by,bw,bh,_=st[big]; keep=np.zeros_like(b); Wd=b.shape[1]
    for i in range(1,n):
        x,y,w,h,area=st[i]
        if i==big: keep[lab==i]=1; continue
        if area<4: continue
        if c in 'iI.' and y+h<=by+3: keep[lab==i]=1; continue            # dot of i
        if x==0 or x+w==Wd or y==0 or y+h==b.shape[0]: continue               # pieces of neighbouring letters
        if y+h<by+3 or y>by+bh: continue
        keep[lab==i]=1
    a=a*cv2.dilate(keep,np.ones((3,3),np.uint8))
    ys,xs=np.where(a>0.15); a=a[ys.min():ys.max()+1,xs.min():xs.max()+1]; y0=y0+ys.min()
    vy,vx=np.where(a>VIS); body=a>VIS
    if c=='i':
        rows=np.where(body.sum(1)>0)[0]; gp=np.where(np.diff(rows)>1)[0]
        if len(gp): body[:rows[gp[-1]+1]]=False
    by_,bx_=np.where(body)
    return dict(c=c,a=a,vt=int(by_.min()),vb=int(by_.max()),vl=int(vx.min()),vr=int(vx.max()),y0=y0)
raw={}
for (y0,y1),items in L.items():
    ex=[extract(x0,x1,y0,y1,c) for c,x0,x1 in items]
    for k,e in enumerate(ex):
        e['desc']=0
        if e['c'] in DESC:
            nb=[ex[j]['vb']+ex[j]['y0'] for j in (k-1,k+1,k-2,k+2) if 0<=j<len(ex) and ex[j]['c'] not in DESC and ex[j]['c'] not in '.i']
            e['desc']=max(0,(e['vb']+e['y0'])-float(np.median(nb[:2])))
        raw.setdefault(e['c'],[]).append(e)
line1=[e for c in 'SANTOHCR' for e in raw[c] if e['y0']<660 and e['c']!='.']
TARGET=float(np.median([e['vb']-e['vt']+1 for e in line1])); print('TARGET cap height',TARGET)
BASE1=float(np.median([e['vb']+e['y0'] for e in line1])); print('line-1 baseline (level y)',BASE1)
def finish(e,c,s=None):
    a=e['a']; above=e['vb']-e['desc']-e['vt']+1
    if s is None: s=1.0 if c in '.i' else TARGET/above
    a=np.clip(cv2.resize(a,None,fx=s,fy=s,interpolation=cv2.INTER_AREA if s<1 else cv2.INTER_CUBIC),0,1)
    a3=cv2.resize(a,(a.shape[1]*SC,a.shape[0]*SC),interpolation=cv2.INTER_CUBIC)
    a3=np.clip((a3*BOOST-THR)/(1-THR),0,1); a3=cv2.GaussianBlur(a3,(5,5),0.6)
    a3=cv2.copyMakeBorder(a3,PAD,PAD,PAD,PAD,cv2.BORDER_CONSTANT,0)
    return dict(a=a3,base=round(PAD+(e['vb']-e['desc']+1)*s*SC,1),vl=round(PAD+e['vl']*s*SC,1),vr=round(PAD+(e['vr']+1)*s*SC,1),y0=e.get('y0',0))
glyphs={}
for c,lst in raw.items():
    key='Y' if c=='y' else c
    lst.sort(key=lambda e:(e['y0']>660, abs((e['vb']-e['desc']-e['vt']+1)-TARGET)))   # name-line samples first
    glyphs.setdefault(key,[]).extend(finish(e,c) for e in lst)
# measure the pen's stroke width on the name line for the drawn letters
ss=[];
for e in line1:
    b=(e['a']>VIS).astype(np.uint8); cs,_=cv2.findContours(b,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE); ss.append(2*b.sum()/sum(cv2.arcLength(c,True) for c in cs))
sw=float(np.median(ss)); print('stroke width',round(sw,2))
rng=np.random.default_rng(5)
SYN={'Y':([[(0.0,0.0),(0.5,0.52)],[(1.0,0.0),(0.5,0.52),(0.48,1.0)]],0.72),
     'P':([[(0.06,1.0),(0.04,0.0)],[(0.04,0.02),(0.55,0.0),(0.85,0.12),(0.88,0.32),(0.6,0.48),(0.06,0.5)]],0.6),
     'Q':([[(0.5,0.0),(0.18,0.1),(0.02,0.42),(0.08,0.8),(0.35,1.0),(0.68,0.98),(0.93,0.75),(0.98,0.38),(0.82,0.08),(0.5,0.0)],[(0.55,0.72),(1.02,1.08)]],0.78),
     'X':([[(0.02,0.0),(0.98,1.0)],[(0.95,0.0),(0.05,1.0)]],0.68),
     'Z':([[(0.03,0.02),(0.95,0.0),(0.05,1.0),(0.98,0.98)]],0.66)}
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
        vs.append(finish(dict(a=a,vt=vy.min(),vb=vb,vl=vx.min(),vr=vx.max(),desc=0),c))
    glyphs[c]=vs
print({c:len(v) for c,v in sorted(glyphs.items())})
rows=[]
for c in sorted(glyphs):
    canvas=np.zeros((52*SC,1000),np.float32); X=10; BASE=36*SC
    for k,g in enumerate(glyphs[c][:6]):
        a=g['a']; h,w=a.shape; top=int(round(BASE-g['base'])); left=int(round(X-g['vl']))
        if top<0 or left<0 or left+w>1000: continue
        hh=min(h,canvas.shape[0]-top); canvas[top:top+hh,left:left+w]=np.maximum(canvas[top:top+hh,left:left+w],a[:hh]); X+=int(g['vr']-g['vl'])+30
    cv2.line(canvas,(0,BASE),(1000,BASE),0.2,1); rows.append(canvas)
half=(len(rows)+1)//2; A=np.vstack(rows[:half]); B=np.vstack(rows[half:]); B=np.pad(B,((0,A.shape[0]-B.shape[0]),(0,0)))
cv2.imwrite('alphabet.png',(255-np.hstack([A,B])*255).astype(np.uint8))
pickle.dump(glyphs,open('glyphs_all.pkl','wb')); json.dump({'TARGET':TARGET,'BASE1':BASE1,'sw':sw},open('gmeta.json','w'))
