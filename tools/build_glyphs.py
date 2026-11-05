import cv2, numpy as np, json, base64
rot=cv2.imread('rot.png'); gray=cv2.cvtColor(rot,cv2.COLOR_BGR2GRAY).astype(np.float32)
SC=3
bands={1:(574,632),2:(634,688),3:(694,752),4:(754,803)}
L={1:[('U',133,152),('S',157,172),('H',176,192),('A',197,214),('J',234,254),('W',285,308),('I',312,317),('S',322,335),('H',338,352),('I',356,361),('N',366,380),('G',382,398),('Y',418,434),('O',436,451),('U',453,469),('A',491,508)],
   2:[('H',150,168),('A',171,190),('P',193,209),('P',212,228),('I',230,237),('E',241,257),('S',259,273),('T',275,294),('B',315,336),('I',340,345),('R',350,367),('H',385,401),('D',406,423)],
   3:[('W',141,170),('I',174,179),('S',184,198),('H',203,219),('Y',242,260),('O',262,278),('U',281,298),('A',320,339),('L',341,355),('B',448,468),('E',471,486),('H',507,522),('A',524,542),('P',544,559),('P',560,575),('Y',576,591)],
   4:[('E',171,187),('A',188,206),('H',238,253),('Y',255,272),('A',289,308),('N',311,328),('D',331,349),('K',375,396),('E',397,415),('E',417,435),('P',437,452),('S',476,491),('H',495,512),('I',516,521),('N',527,545),('N',549,566),('N',581,598),('G',599,618)]}
def extract(x0,x1,y0,y1,c='?'):
    x0-=1;x1+=1
    g=gray[y0:y1,x0:x1]
    paper=np.percentile(gray[y0:y1,max(0,x0-30):x1+30],80); ink=np.percentile(g,2)
    a=np.clip((paper-g)/(paper-ink),0,1)
    b=(a>0.45).astype(np.uint8)
    n,lab,st,_=cv2.connectedComponentsWithStats(b)
    big=1+np.argmax(st[1:,cv2.CC_STAT_AREA]); bx,by,bw,bh,_=st[big]
    keep=np.zeros_like(b); W=b.shape[1]
    for i in range(1,n):
        x,y,ww,hh,area=st[i]
        if i==big: keep[lab==i]=1; continue
        if area<5 or x==0 or x+ww==W: continue
        if y+hh<by-14 or y>by+bh+4: continue   # stray doodles
        if c!='I' and (y+hh<by+4 or area<25): continue   # dots from neighbouring i
        keep[lab==i]=1
    keep=cv2.dilate(keep,np.ones((3,3),np.uint8))
    a=a*keep
    ys,xs=np.where(a>0.15)
    return a[ys.min():ys.max()+1,xs.min():xs.max()+1], y0+ys.max()+1
raw={}
for ln,items in L.items():
    y0,y1=bands[ln]
    ex=[(c,)+extract(x0,x1,y0,y1,c) for c,x0,x1 in items]
    plain=[b for c,a,b in ex if c not in 'JYGI']
    base=int(np.median(plain)); caph=np.median([a.shape[0] for c,a,b in ex if c not in 'JYGIQ'])
    for k,(c,a,b) in enumerate(ex):
        if c in 'JY':
            nb=[ex[j][2] for j in (k-1,k+1,k-2,k+2) if 0<=j<len(ex) and ex[j][0] not in 'JYG']
            dy=b-np.median(nb[:2])
        else: dy=0
        raw.setdefault(c,[]).append((a,dy,caph,ln))
TARGET=29.0
# stroke width estimate from glyph H
A0=raw['H'][0][0]; sw=float((A0>0.5).sum())/ (A0.shape[0]*2+A0.shape[1]) 
print('stroke width ~',round(sw,2))
glyphs={}
raw['I']=[raw['I'][i] for i in (0,1,2,4)]; raw['N']=[raw['N'][i] for i in (0,1,3)]
def finish(a):  # -> 3x, boosted, softened
    a3=cv2.resize(a,(a.shape[1]*SC,a.shape[0]*SC),interpolation=cv2.INTER_CUBIC)
    a3=np.clip(a3*1.6,0,1); a3=cv2.GaussianBlur(a3,(5,5),0.8)
    a3=cv2.copyMakeBorder(a3,3,3,3,3,cv2.BORDER_CONSTANT,0)
    return a3
for c,lst in raw.items():
    for a,dy,caph,ln in lst:
        s=TARGET/caph
        a=cv2.resize(a,None,fx=s,fy=s,interpolation=cv2.INTER_AREA if s<1 else cv2.INTER_LINEAR)
        glyphs.setdefault(c,[]).append({'a':finish(a),'dy':dy*s,'w':a.shape[1],'h':a.shape[0]})
# synthetic letters, normalized 0..1 coords (x right, y down), drawn in handwriting proportions
rng=np.random.default_rng(7)
SYN={'C':([[(0.95,0.18),(0.75,0.03),(0.45,0.0),(0.18,0.12),(0.03,0.4),(0.05,0.7),(0.22,0.93),(0.5,1.0),(0.78,0.95),(0.97,0.82)]],0.62),
     'F':([[(0.05,0.0),(0.02,1.0)],[(0.05,0.02),(0.95,0.0)],[(0.04,0.48),(0.75,0.46)]],0.55),
     'M':([[(0.0,1.0),(0.06,0.0),(0.5,0.62),(0.92,0.0),(1.0,1.0)]],0.95),
     'Q':([[(0.5,0.0),(0.18,0.1),(0.02,0.42),(0.08,0.8),(0.35,1.0),(0.68,0.98),(0.93,0.75),(0.98,0.38),(0.82,0.08),(0.5,0.0)],[(0.55,0.72),(1.02,1.08)]],0.72),
     'V':([[(0.0,0.0),(0.5,1.0),(1.0,0.0)]],0.72),
     'X':([[(0.02,0.0),(0.98,1.0)],[(0.95,0.0),(0.05,1.0)]],0.7),
     'Z':([[(0.03,0.02),(0.95,0.0),(0.05,1.0),(0.98,0.98)]],0.68)}
UP=8
for c,(strokes,aspect) in SYN.items():
    variants=[]
    for v in range(2):
        hh=TARGET; ww=TARGET*aspect; pad=4
        canvas=np.zeros((int((hh+2*pad)*UP),int((ww+2*pad)*UP)),np.uint8)
        for st in strokes:
            pts=[]
            for (x,y) in st:
                jx,jy=rng.normal(0,0.012,2)
                pts.append([ (pad+(x+jx)*ww)*UP, (pad+(y+jy)*hh)*UP ])
            pts=np.array(pts,np.float32)
            # smooth curves (C,Q) by chaikin
            if len(pts)>4:
                for _ in range(3):
                    q=0.75*pts[:-1]+0.25*pts[1:]; r=0.25*pts[:-1]+0.75*pts[1:]
                    pts=np.vstack([pts[:1],np.column_stack([q,r]).reshape(-1,2),pts[-1:]])
            cv2.polylines(canvas,[pts.astype(np.int32)],False,255,int(sw*UP*1.05),cv2.LINE_AA)
        a=cv2.resize(canvas.astype(np.float32)/255,(int(ww+2*pad),int(hh+2*pad)),interpolation=cv2.INTER_AREA)
        a=cv2.GaussianBlur(a,(3,3),0.5)
        ys,xs=np.where(a>0.15); a=a[ys.min():ys.max()+1,xs.min():xs.max()+1]
        variants.append({'a':finish(a),'dy':0.0 if c!='Q' else 2.0,'w':a.shape[1],'h':a.shape[0]})
    glyphs[c]=variants
print({c:len(v) for c,v in sorted(glyphs.items())})
# preview sheet
rows=[]
for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
    vs=glyphs[c]; cell=np.zeros((260,130*len(vs)),np.float32)
    for k,g in enumerate(vs):
        a=g['a']; hh,ww=a.shape; y=170+int(g['dy']*SC)-hh+3
        cell[y:y+hh,k*130+5:k*130+5+min(ww,120)]=a[:,:120]
    cv2.line(cell,(0,173),(cell.shape[1],173),0.15,1)
    rows.append(cell)
W=max(r.shape[1] for r in rows)
rows=[np.pad(r,((0,0),(0,W-r.shape[1]))) for r in rows]
sheet=np.vstack([np.hstack([rows[i],rows[i+13]]) if i+13<26 else rows[i] for i in range(13)])
cv2.imwrite('alphabet.png',(255-sheet*255).astype(np.uint8))
# export
out={}
for c,vs in glyphs.items():
    out[c]=[]
    for g in vs:
        a=(g['a']*255).astype(np.uint8)
        rgba=np.dstack([np.full_like(a,255)]*3+[a])
        ok,buf=cv2.imencode('.png',rgba)
        out[c].append({'src':'data:image/png;base64,'+base64.b64encode(buf).decode(),'w':g['a'].shape[1],'h':g['a'].shape[0],'dy':round(g['dy']*SC,1)})
json.dump(out,open('glyphs.json','w'))
print('json KB',len(json.dumps(out))//1024)
