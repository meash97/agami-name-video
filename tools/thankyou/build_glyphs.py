import cv2, numpy as np, json, os, pickle, sys
sys.path.insert(0,'/home/claude/t5'); from common import *
rot=cv2.imread('rot241.png'); gray=cv2.cvtColor(rot,cv2.COLOR_BGR2GRAY).astype(np.float32)
SC=3; PAD=3; VIS=0.5
BOOST=float(os.environ.get('BOOST','1.2')); THR=float(os.environ.get('THR','0.25'))
BAND={1:(572,660),2:(650,737),3:(738,820),4:(826,888),5:(900,956)}
SPECIAL={('g',4):(826,905)}
CUTS={1:[('D',42,102),('e',102,124),('a',124,155),('r',155,188),('S',216,250),('a',250,273),('n',273,297),('t',297,314),('o',314,336),('s',336,357),('h',357,390),
         ('J',395,434),('i',433,447),('C',474,498),('h',498,524),('a',524,544),('a',544,563),('j',563,584),('a',584,605),('r',605,640)],
      2:[('a',97,120),('n',120,147),('k',147,190),('y',220,252),('o',252,277),('u',277,301),('f',330,352),('o',352,375),('r',376,404),('a',426,453),
         ('K',476,506),('D',541,578),('V',587,618),('S',631,653)],
      3:[('a',58,80),('n',80,108),('d',108,142),('Y',170,192),('o',192,221),('u',221,246),('r',246,271),('V',299,319),('a',319,339),('l',339,350),('u',350,371),
         ('a',371,391),('b',391,414),('l',414,429),('e',429,461),('T',484,502),('i',503,512),('m',512,546),('e',546,571)],
      4:[('W',208,252),('m',304,344),('R',362,392),('e',392,410),('g',410,440),('a',440,460),('r',460,480),('d',480,503),('s',503,526)],
      5:[('A',264,331),('G',331,371),('M',391,442)]}
def alpha(x0,x1,y0,y1):
    g=gray[y0:y1,x0:x1]; paper=np.percentile(gray[y0:y1,max(0,x0-50):x1+50],80); ink=np.percentile(gray[580:960,60:640],0.5)
    return np.clip((paper-g)/(paper-ink),0,1)
def clean(a):
    b=(a>0.35).astype(np.uint8); n,lab,st,_=cv2.connectedComponentsWithStats(b)
    if n<2: return None
    big=1+np.argmax(st[1:,4]); bx,by,bw,bh,_=st[big]; keep=np.zeros_like(b); Wd=b.shape[1]; Hd=b.shape[0]
    for i in range(1,n):
        x,y,w,h,area=st[i]
        if i==big: keep[lab==i]=1; continue
        if area<5: continue
        if y==0 or y+h==Hd: continue                                            # pieces of other lines
        if x==0 or x+w==Wd:
            if area<40: continue
        keep[lab==i]=1
    a=a*cv2.dilate(keep,np.ones((3,3),np.uint8)); ys,xs=np.where(a>0.12)
    return a[ys.min():ys.max()+1,xs.min():xs.max()+1], ys.min(), xs.min()
def refine(ln,items):
    y0,y1=BAND[ln]; g=gray[y0:y1]; paper=np.percentile(g,80); ink=np.clip((paper-g)/60,0,1).sum(0)
    out=[list(t) for t in items]
    for k in range(len(out)-1):
        if out[k][2]==out[k+1][1] and ln!=4:
            x=out[k][2]; lo,hi=x-7,x+8; c=lo+int(np.argmin(ink[lo:hi])); out[k][2]=c; out[k+1][1]=c
    return [tuple(t) for t in out]
CUTS={ln:refine(ln,it) for ln,it in CUTS.items()}
raw=[]
for ln,items in CUTS.items():
  for _ in [0]:
    for c,x0,x1 in items:
        y0,y1=SPECIAL.get((c,ln),BAND[ln])
        r=clean(alpha(x0-1,x1+1,y0,y1))
        if r is None: continue
        a,yt,xt=r; vy,vx=np.where(a>VIS)
        raw.append(dict(c=c,a=a,top=y0+yt,vt=int(vy.min()),vb=int(vy.max()),vl=int(vx.min()),vr=int(vx.max()),ln=ln))
DROP={16,17,27,43,44,47,48,56,57,61}
raw=[e for k,e in enumerate(raw) if k not in DROP]
# per-line baseline and x-height from round lowercase
X_LET='aenorsum'
base={}; xh={}
for ln in (1,2,3,4):
    es=[e for e in raw if e['ln']==ln and e['c'] in X_LET]
    base[ln]=float(np.median([e['top']+e['vb'] for e in es])); xh[ln]=float(np.median([e['vb']-e['vt']+1 for e in es]))
print('baselines',base,'x-heights',xh)
CAP=float(np.median([e['vb']-e['vt']+1 for e in raw if e['ln']==1 and e['c'] in 'SC']))
print('cap height (line 1)',CAP)
def finish(a,base_row,vl,vr,s):
    a=np.clip(cv2.resize(a,None,fx=s,fy=s,interpolation=cv2.INTER_AREA if s<1 else cv2.INTER_CUBIC),0,1)
    a3=cv2.resize(a,(a.shape[1]*SC,a.shape[0]*SC),interpolation=cv2.INTER_CUBIC)
    a3=np.clip((a3*BOOST-THR)/(1-THR),0,1); a3=cv2.GaussianBlur(a3,(5,5),0.8)
    a3=cv2.copyMakeBorder(a3,PAD,PAD,PAD,PAD,cv2.BORDER_CONSTANT,0)
    return dict(a=a3,base=round(PAD+base_row*s*SC,1),vl=round(PAD+vl*s*SC,1),vr=round(PAD+(vr+1)*s*SC,1))
glyphs={}
def add(key,g,front=False):
    glyphs.setdefault(key,[]); (glyphs[key].insert(0,g) if front else glyphs[key].append(g))
for e in sorted(raw,key=lambda e:e['ln']!=1):
    if e['ln']==5:
        s=CAP*1.05/(e['vb']-e['vt']+1); brow=e['vb']+1
    else:
        s=xh[1]/xh[e['ln']]; brow=base[e['ln']]-e['top']+1
    add(e['c'],finish(e['a'],brow,e['vl'],e['vr'],s))
# capitals the note doesn't have: its own lowercase, enlarged to cap height (b, f, h, l, k already reach it)
def from_lower(lc,key,target):
    e=next(x for x in raw if x['c']==lc and x['ln']==1) if any(x['c']==lc and x['ln']==1 for x in raw) else next(x for x in raw if x['c']==lc)
    body=e['vb']-e['vt']+1 if lc not in 'bfhlkdgjy' else None
    if lc in 'bhlkd': s=CAP/(base[e['ln']]-e['top']-e['vt']+1)
    elif lc=='f': s=CAP/(base[e['ln']]-e['top']-e['vt']+1)
    else: s=target/body
    brow=base[e['ln']]-e['top']+1
    add(key,finish(e['a'],brow,e['vl'],e['vr'],s))
for lc,K in [('b','B'),('f','F'),('h','H')]: from_lower(lc,K,CAP*0.92)
# lowercase the note doesn't have: c, v, w from the capitals shrunk to x-height
def from_upper(UC,key):
    e=next(x for x in raw if x['c']==UC)
    if e['ln']==5: return
    s=xh[1]/(e['vb']-e['vt']+1)*1.08; brow=e['vb']+1
    add(key,finish(e['a'],brow,e['vl'],e['vr'],s))

glyphs['I']=[g for g in glyphs.get('I',[])]
# stroke width (line 1 lowercase)
ss=[]
for e in raw:
    if e['ln']==1 and e['c'] in 'anr':
        b=(e['a']>VIS).astype(np.uint8); cs,_=cv2.findContours(b,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE); ss.append(2*b.sum()/sum(cv2.arcLength(q,True) for q in cs))
sw=float(np.median(ss)); print('stroke',round(sw,2))
UP=8
def synth(strokes,w,h,top_off,desc=0):
    pad=5; c=np.zeros((int((h+desc+2*pad)*UP),int((w+2*pad)*UP)),np.uint8)
    for st in strokes:
        p=np.array([[(pad+x*w)*UP,(pad+y*h)*UP] for x,y in st],np.float32)
        if len(p)>3:
            for _ in range(3):
                q=0.75*p[:-1]+0.25*p[1:]; r=0.25*p[:-1]+0.75*p[1:]; p=np.vstack([p[:1],np.column_stack([q,r]).reshape(-1,2),p[-1:]])
        cv2.polylines(c,[p.astype(np.int32)],False,255,int(sw*UP),cv2.LINE_AA)
    a=cv2.GaussianBlur(cv2.resize(c.astype(np.float32)/255,(int(w+2*pad),int(h+desc+2*pad)),interpolation=cv2.INTER_AREA),(3,3),0.9)
    sh=np.float32([[1,-0.18,0.18*(h+2*pad)],[0,1,0]]); a=cv2.warpAffine(a,sh,(a.shape[1]+8,a.shape[0]))      # the note's forward slant
    ys,xs=np.where(a>0.12); y0=ys.min(); a=a[ys.min():ys.max()+1,xs.min():xs.max()+1]; vy,vx=np.where(a>VIS)
    brow=(pad+h)-y0
    return finish(a,brow,vx.min(),vx.max(),1.0)
XH=xh[1]
glyphs['p']=[synth([[(0.12,0.0),(0.1,1.6)],[(0.12,0.25),(0.45,0.0),(0.85,0.15),(0.9,0.6),(0.55,1.0),(0.12,0.85)]],XH*0.8,XH,0,XH*0.7)]
glyphs['q']=[synth([[(0.85,0.3),(0.5,0.0),(0.12,0.2),(0.1,0.7),(0.45,1.0),(0.85,0.75)],[(0.88,0.05),(0.86,1.6),(1.05,1.35)]],XH*0.8,XH,0,XH*0.7)]
glyphs['l']=[synth([[(0.1,1.0),(0.55,0.0),(0.95,-0.85),(0.7,-1.0),(0.45,-0.5),(0.3,0.5),(0.35,1.0),(0.8,0.95)]],XH*0.55,XH,0)]
glyphs['j']=[synth([[(0.75,0.05),(0.62,1.25),(0.35,1.65),(0.05,1.55),(0.1,1.3)],[(0.9,-0.45),(0.92,-0.35)]],XH*0.6,XH,0,XH*0.75)]
glyphs['g']=[synth([[(0.85,0.25),(0.5,0.0),(0.12,0.2),(0.1,0.75),(0.45,1.0),(0.85,0.7)],[(0.9,0.05),(0.85,1.25),(0.55,1.65),(0.2,1.55),(0.25,1.3)]],XH*0.8,XH,0,XH*0.75)]
glyphs['x']=[synth([[(0.0,0.0),(1.0,1.0)],[(1.0,0.0),(0.0,1.0)]],XH*0.85,XH,0)]
glyphs['z']=[synth([[(0.05,0.03),(0.95,0.0),(0.05,1.0),(0.98,0.97)]],XH*0.85,XH,0)]
glyphs['I']=[synth([[(0.55,0.0),(0.45,1.0)],[(0.0,0.02),(1.0,0.0)],[(0.0,1.0),(1.0,0.98)]],CAP*0.38,CAP,0)]
glyphs['c']=[synth([[(0.9,0.2),(0.55,0.0),(0.15,0.2),(0.05,0.65),(0.35,1.0),(0.8,0.95)]],XH*0.75,XH,0)]
glyphs['v']=[synth([[(0.0,0.05),(0.45,1.0),(0.95,0.0)]],XH*0.85,XH,0)]
glyphs['w']=[synth([[(0.0,0.05),(0.22,1.0),(0.5,0.25),(0.75,1.0),(1.0,0.0)]],XH*1.2,XH,0)]
glyphs['L']=[synth([[(0.12,0.0),(0.06,1.0),(0.95,0.97)]],CAP*0.5,CAP,0)]
glyphs['E']=[synth([[(0.95,0.02),(0.12,0.0),(0.06,1.0),(0.95,0.98)],[(0.09,0.5),(0.75,0.48)]],CAP*0.55,CAP,0)]
glyphs['N']=[synth([[(0.04,1.0),(0.08,0.0),(0.9,1.0),(0.96,0.0)]],CAP*0.7,CAP,0)]
glyphs['O']=[synth([[(0.5,0.0),(0.15,0.12),(0.02,0.45),(0.12,0.85),(0.45,1.0),(0.8,0.9),(0.97,0.55),(0.88,0.15),(0.5,0.0)]],CAP*0.78,CAP,0)]
glyphs['U']=[synth([[(0.05,0.0),(0.06,0.7),(0.3,1.0),(0.7,1.0),(0.94,0.7),(0.95,0.0)]],CAP*0.7,CAP,0)]
glyphs['P']=[synth([[(0.12,1.0),(0.1,0.0)],[(0.1,0.03),(0.6,0.0),(0.92,0.18),(0.9,0.4),(0.55,0.52),(0.12,0.5)]],CAP*0.6,CAP,0)]
glyphs['Q']=[synth([[(0.5,0.0),(0.15,0.12),(0.02,0.45),(0.12,0.85),(0.45,1.0),(0.8,0.9),(0.97,0.55),(0.88,0.15),(0.5,0.0)],[(0.55,0.75),(1.0,1.1)]],CAP*0.8,CAP,0)]
glyphs['X']=[synth([[(0.0,0.0),(1.0,1.0)],[(1.0,0.0),(0.0,1.0)]],CAP*0.7,CAP,0)]
glyphs['Z']=[synth([[(0.05,0.03),(0.95,0.0),(0.05,1.0),(0.98,0.97)]],CAP*0.7,CAP,0)]
print({c:len(v) for c,v in sorted(glyphs.items())})
# preview: every letter on a baseline
def sheet(keys,fname):
    rows=[]
    for c in keys:
        canvas=np.zeros((74*SC//2,1100),np.float32); X=10; BASE=int(48*SC/2)
        for g in glyphs[c][:6]:
            a=cv2.resize(g['a'],None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA); h,w=a.shape; top=int(round(BASE-g['base']/2)); left=int(round(X-g['vl']/2))
            if top<0 or left<0 or left+w>1100: continue
            hh=min(h,canvas.shape[0]-top); canvas[top:top+hh,left:left+w]=np.maximum(canvas[top:top+hh,left:left+w],a[:hh]); X+=int((g['vr']-g['vl'])/2)+20
        cv2.line(canvas,(0,BASE),(1100,BASE),0.2,1); rows.append(canvas)
    half=(len(rows)+1)//2; A=np.vstack(rows[:half]); B=np.vstack(rows[half:]); B=np.pad(B,((0,A.shape[0]-B.shape[0]),(0,0)))
    cv2.imwrite(fname,(255-np.hstack([A,B])*255).astype(np.uint8))
sheet('ABCDEFGHIJKLMNOPQRSTUVWXYZ','alpha_up.png'); sheet('abcdefghijklmnopqrstuvwxyz','alpha_lo.png')
pickle.dump(glyphs,open('glyphs_all.pkl','wb')); json.dump({'base':base,'xh':xh,'CAP':CAP,'sw':sw},open('gmeta.json','w'))
