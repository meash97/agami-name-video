import json, base64, cv2
glyphs=json.load(open('glyphs.json')); inks=json.load(open('inks.json')); motion=json.load(open('motion.json')); decor=json.load(open('decor.json'))
th=cv2.imread('plate/0241.png')[:1292]; th=cv2.resize(th,(181,323),interpolation=cv2.INTER_AREA)
ok,buf=cv2.imencode('.jpg',th,[cv2.IMWRITE_JPEG_QUALITY,80])
tpl=[{
  'id':'bday-notebook-1','title':'Birthday · Notebook','fileTag':'Birthday',
  'video':'bday1','glyphs':'notebook-hand','thumb':'data:image/jpeg;base64,'+base64.b64encode(buf).decode(),
  'w':724,'h':1292,'fps':24,'angle':20.5,'sc':3,'pad':3,
  'anchor':{'right':272,'left':12,'baseline':618},'append':',',
  'gap':3.5,'space':10,'minGap':1.5,'minSpace':6,'minCondense':0.8,'suffix':'','maxLen':22,'help':'Type the name as it should appear — add JI yourself if you want it. Letters A–Z and spaces.',
  'region':{'x':0,'y':540,'w':290,'h':110},'matte':{'sx':0,'sy':1300,'w':290,'h':110},
  'zoom':{'x':20,'y':560,'w':360,'h':200},
  'previewTime':9.95,'inks':inks,'motion':motion,'decor':decor}]
vid='data:video/mp4;base64,'+base64.b64encode(open('plate.mp4','rb').read()).decode()
# ---- anniversary template
import pickle, numpy as np
AD='/home/claude/anniv/'
ag=pickle.load(open(AD+'glyphs_all.pkl','rb')); pick={'A':[0,1,2,3],'S':[0,1,2,4]}
aglyphs={}
for c,vs in ag.items():
    aglyphs[c]=[]
    for k in pick.get(c,range(min(4,len(vs)))):
        g=vs[k]; al=(g['a']*255).astype(np.uint8); ok,b2=cv2.imencode('.png',np.dstack([np.full_like(al,255)]*3+[al]))
        aglyphs[c].append({'src':'data:image/png;base64,'+base64.b64encode(b2).decode(),'w':al.shape[1],'h':al.shape[0],'base':g['base'],'vl':g['vl'],'vr':g['vr']})
reg=json.load(open(AD+'reg.json'))
ath=cv2.resize(cv2.imread(AD+'plate/0241.png')[:1292],(181,323),interpolation=cv2.INTER_AREA); ok,ab=cv2.imencode('.jpg',ath,[cv2.IMWRITE_JPEG_QUALITY,80])
tpl.append({
  'id':'anniv-sketchbook-1','title':'Anniversary · Sketchbook','fileTag':'Anniversary',
  'video':'anniv1','glyphs':'sketchbook-hand','thumb':'data:image/jpeg;base64,'+base64.b64encode(ab).decode(),
  'w':724,'h':1292,'fps':24,'angle':23.75,'sc':3,'pad':3,
  'align':'left','append':',','firstScale':1.2,
  'anchor':{'left':158,'right':716,'baseline':528},
  'gap':2,'space':14,'minGap':1,'minSpace':8,'minCondense':0.8,'suffix':'','maxLen':40,
  'label':'Names on the note','help':'Type the whole first line, e.g. both names with JI and the surname. Letters A–Z and spaces.',
  'region':{'x':reg['x'],'y':reg['y'],'w':reg['w'],'h':reg['h']},
  'matte':{'sx':0,'sy':1300,'w':reg['w'],'h':reg['h']},
  'zoom':{'x':80,'y':350,'w':600,'h':330},
  'previewTime':9.95,'inks':json.load(open(AD+'inks.json')),'motion':json.load(open(AD+'motion.json'))})
avid='data:video/mp4;base64,'+base64.b64encode(open(AD+'plate.mp4','rb').read()).decode()
s=open('app_template.html').read()
s=s.replace('/*TEMPLATES*/[]',json.dumps(tpl)).replace('/*GLYPHS*/{}',json.dumps({'notebook-hand':glyphs,'sketchbook-hand':aglyphs})).replace('/*VIDEOS*/{}',json.dumps({'bday1':vid,'anniv1':avid}))
open('/home/claude/AGAMI-Name-Video-Maker.html','w').write(s)
import hashlib
ver=hashlib.sha1(s.encode()).hexdigest()[:10]
open('/home/claude/vid/sw.js','w').write('''const CACHE = 'agami-name-video-%s';
const ASSETS = ['./', './index.html', './manifest.webmanifest', './icon-192.png', './icon-512.png', './icon-maskable-512.png', './apple-touch-icon.png'];
self.addEventListener('install', e => { e.waitUntil(caches.open(CACHE).then(c => c.addAll(ASSETS))); });
self.addEventListener('activate', e => { e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k)))).then(() => self.clients.claim())); });
self.addEventListener('message', e => { if (e.data === 'skipWaiting') self.skipWaiting(); });
self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET' || new URL(e.request.url).origin !== location.origin) return;
  e.respondWith(caches.match(e.request, {ignoreSearch: true}).then(r => r || fetch(e.request).then(res => { const copy = res.clone(); caches.open(CACHE).then(c => c.put(e.request, copy)); return res; })));
});
''' % ver)
print('sw version',ver)
import os; print('HTML MB',round(os.path.getsize('/home/claude/AGAMI-Name-Video-Maker.html')/1048576,2))
