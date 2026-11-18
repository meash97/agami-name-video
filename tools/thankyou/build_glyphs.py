import cv2, numpy as np, json, os, pickle, sys
sys.path.insert(0,'/home/claude/t4'); from common import *
rot=cv2.imread('rot.png'); gray=cv2.cvtColor(rot,cv2.COLOR_BGR2GRAY).astype(np.float32)
SC=3; PAD=3; VIS=0.5
BOOST=float(os.environ.get('BOOST','1.35')); THR=float(os.environ.get('THR','0.17'))
segs={int(k):v for k,v in json.load(open('segs.json')).items()}
BAND={1:(546,590),2:(594,638),3:(642,686),4:(688,732),5:(734,778),6:(782,826)}
LAB={1:['D','E','A','R','S','A','N','TO','S','H','J','U','C','H',None,'J','A','R',','],
     2:['T','H','A','N','K','YO','U','FO','R','A','K','I','N','D','V','I','S','I','T'],
     3:['A','N','D','G','I','V','I','N','G','YO','U','R'],
     4:['VA','L','UA','B','L','E','T','I','M','E','.'],
     5:['WA','R','M','R','E','G','A','R','D','S',','],
     6:['A','G','A','M','I','.']}
def alpha(x0,x1,y0,y1):
    g=gray[y0:y1,x0:x1]; paper=np.percentile(gray[y0:y1,max(0,x0-40):x1+40],80); ink=np.percentile(gray[546:826,100:560],0.3)
    return np.clip((paper-g)/(paper-ink),0,1)
def clean(a,c,only_big=False):
    b=(a>0.35).astype(np.uint8); n,lab,st,_=cv2.connectedComponentsWithStats(b)
    if n<2: return None
    big=1+np.argmax(st[1:,4]); keep=np.zeros_like(b); bx,by,bw,bh,_=st[big]
    for i in range(1,n):
        x,y,w,h,area=st[i]
        if i==big or (not only_big and area>=15 and y<by+bh and y+h>by): keep[lab==i]=1
    a=a*cv2.dilate(keep,np.ones((3,3),np.uint8)); ys,xs=np.where(a>0.12)
    return a[ys.min():ys.max()+1,xs.min():xs.max()+1], ys.min()
raw={}; bases={}
for ln,ss in segs.items():
    y0,y1=BAND[ln]; items=[]
    for (x0,x1),lab in zip(ss,LAB[ln]):
        if lab is None: continue
        a=alpha(x0-1,x1+1,y0,y1)
        if len(lab)==2:
            col=(a>0.35).sum(0).astype(float); w=len(col)
            lo,hi=(int(w*0.35),int(w*0.62)) if lab!='WA' else (int(w*0.56),int(w*0.66))
            cut=lo+int(np.argmin(col[lo:hi]))
            aL=a.copy(); aL[:,cut:]=0; aR=a.copy(); aR[:,:cut]=0
            SKIP={('TO','T'),('TO','O'),('VA','V'),('VA','A'),('UA','U'),('UA','A'),('WA','A')}
            for c,pa in ((lab[0],aL),(lab[1],aR)):
                if (lab,c) in SKIP: continue
                items.append((c,pa,1))
        else: items.append((lab,a,0))
    out=[]
    for c,a,split in items:
        r=clean(a,c,only_big=bool(split))
        if r is None: continue
        g,yt=r; vy,vx=np.where(g>VIS)
        if len(vy)==0: print('skip faint',c,ln); continue
        out.append(dict(c=c,a=g,vt=int(vy.min()),vb=int(vy.max()),vl=int(vx.min()),vr=int(vx.max()),y0=y0+yt,ln=ln))
    base=np.median([e['vb']+e['y0'] for e in out if e['c'] not in ',.J'])
    bases[ln]=float(base)
    for e in out:
        e['desc']=max(0,(e['vb']+e['y0'])-base) if e['c'] in ',J' else 0
        raw.setdefault(e['c'],[]).append(e)
caps=[e['vb']-e['vt']+1 for c in 'DEARSNTHUCKIBLMG' for e in raw.get(c,[])]
TARGET=float(np.median(caps)); print('cap height',TARGET,'baselines',bases)
def finish(e,c,s=None):
    a=e['a']; above=e['vb']-e['desc']-e['vt']+1
    if s is None: s=1.0 if c in ',.' else TARGET/above
    a=np.clip(cv2.resize(a,None,fx=s,fy=s,interpolation=cv2.INTER_AREA if s<1 else cv2.INTER_CUBIC),0,1)
    a3=cv2.resize(a,(a.shape[1]*SC,a.shape[0]*SC),interpolation=cv2.INTER_CUBIC)
    a3=np.clip((a3*BOOST-THR)/(1-THR),0,1); a3=cv2.GaussianBlur(a3,(5,5),0.8)
    a3=cv2.copyMakeBorder(a3,PAD,PAD,PAD,PAD,cv2.BORDER_CONSTANT,0)
    return dict(a=a3,base=round(PAD+(e['vb']-e['desc']+1)*s*SC,1),vl=round(PAD+e['vl']*s*SC,1),vr=round(PAD+(e['vr']+1)*s*SC,1),ln=e.get('ln',0))
glyphs={}
for c,lst in raw.items():
    lst.sort(key=lambda e:(e['ln']!=1, abs((e['vb']-e['desc']-e['vt']+1)-TARGET)))
    glyphs[c]=[finish(e,c) for e in lst]
# stroke width
ss=[]
for c in 'HNTIL':
    for e in raw.get(c,[]):
        b=(e['a']>VIS).astype(np.uint8); cs,_=cv2.findContours(b,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE); ss.append(2*b.sum()/sum(cv2.arcLength(q,True) for q in cs))
sw=float(np.median(ss)); print('stroke',round(sw,2))
# P = R without its leg
def make_P(e):
    a=e['a'].copy(); h,w=a.shape; mid=int(h*0.55)
    lower=a[mid:]; col=(lower>0.35).sum(0); stem_r=int(np.argmax(col[:w//2]>0))+int(sw*1.6)+1
    a[mid:,stem_r:]=0; ys,xs=np.where(a>0.12); a=a[:ys.max()+1,:xs.max()+1]; vy,vx=np.where(a>VIS)
    return dict(a=a,vt=vy.min(),vb=e['vb'],vl=vx.min(),vr=vx.max(),desc=0)
glyphs['P']=[finish(make_P(e),'P') for e in raw['R'][:2]]
# Q = O + tail
def make_Q(e):
    a=e['a']; h,w=a.shape; pad=6; A=np.zeros((h+pad,w+pad),np.float32); A[:h,:w]=a
    c=np.zeros(((h+pad)*8,(w+pad)*8),np.uint8); cv2.line(c,(int(w*0.55*8),int(h*0.68*8)),(int((w+2)*8),int((h+3)*8)),255,int(sw*8),cv2.LINE_AA)
    t=cv2.GaussianBlur(cv2.resize(c.astype(np.float32)/255,(w+pad,h+pad),interpolation=cv2.INTER_AREA),(3,3),0.6)
    A=np.maximum(A,t); ys,xs=np.where(A>0.12); A=A[:ys.max()+1,:xs.max()+1]; vy,vx=np.where(A>VIS)
    return dict(a=A,vt=vy.min(),vb=e['vb'],vl=vx.min(),vr=vx.max(),desc=0)
glyphs['Q']=[finish(make_Q(e),'Q') for e in raw['O'][:2]]
UP=8
def synth(strokes,aspect):
    hh=TARGET; ww=TARGET*aspect; pad=4
    c=np.zeros((int((hh+2*pad)*UP),int((ww+2*pad)*UP)),np.uint8)
    for st in strokes:
        p=np.array([[(pad+x*ww)*UP,(pad+y*hh)*UP] for x,y in st],np.int32); cv2.polylines(c,[p],False,255,int(sw*UP),cv2.LINE_AA)
    a=cv2.GaussianBlur(cv2.resize(c.astype(np.float32)/255,(int(ww+2*pad),int(hh+2*pad)),interpolation=cv2.INTER_AREA),(3,3),0.9)
    ys,xs=np.where(a>0.12); a=a[ys.min():ys.max()+1,xs.min():xs.max()+1]; vy,vx=np.where(a>VIS)
    return finish(dict(a=a,vt=vy.min(),vb=vy.max(),vl=vx.min(),vr=vx.max(),desc=0),'X')
glyphs['X']=[synth([[(0.02,0),(0.98,1)],[(0.98,0),(0.02,1)]],0.72)]
glyphs['Z']=[synth([[(0.06,0.02),(0.94,0.02),(0.06,0.98),(0.96,0.98)]],0.66)]
print({c:len(v) for c,v in sorted(glyphs.items())})
rows=[]
for c in sorted(glyphs):
    canvas=np.zeros((46*SC,900),np.float32); X=10; BASE=34*SC
    for g in glyphs[c][:6]:
        a=g['a']; h,w=a.shape; top=int(round(BASE-g['base'])); left=int(round(X-g['vl']))
        if top<0 or left<0 or left+w>900: continue
        hh=min(h,canvas.shape[0]-top); canvas[top:top+hh,left:left+w]=np.maximum(canvas[top:top+hh,left:left+w],a[:hh]); X+=int(g['vr']-g['vl'])+24
    cv2.line(canvas,(0,BASE),(900,BASE),0.2,1); rows.append(canvas)
half=(len(rows)+1)//2; A=np.vstack(rows[:half]); B=np.vstack(rows[half:]); B=np.pad(B,((0,A.shape[0]-B.shape[0]),(0,0)))
cv2.imwrite('alphabet.png',(255-np.hstack([A,B])*255).astype(np.uint8))
pickle.dump(glyphs,open('glyphs_all.pkl','wb')); json.dump({'TARGET':TARGET,'bases':bases,'sw':sw},open('gmeta.json','w'))
