import json, base64, cv2
glyphs=json.load(open('glyphs.json')); inks=json.load(open('inks.json')); motion=json.load(open('motion.json')); decor=json.load(open('decor.json'))
th=cv2.imread('plate/0241.png'); th=cv2.resize(th,(181,323),interpolation=cv2.INTER_AREA)
ok,buf=cv2.imencode('.jpg',th,[cv2.IMWRITE_JPEG_QUALITY,80])
tpl=[{
  'id':'bday-notebook-1','title':'Birthday · Notebook','fileTag':'Birthday',
  'video':'bday1','glyphs':'notebook-hand','thumb':'data:image/jpeg;base64,'+base64.b64encode(buf).decode(),
  'w':724,'h':1292,'fps':24,'angle':20.5,'sc':3,'pad':3,
  'anchor':{'right':217,'left':12,'baseline':618},
  'gap':3.5,'space':10,'minGap':1.5,'minSpace':6,'minCondense':0.8,'suffix':'JI,','maxLen':16,
  'region':{'x':0,'y':540,'w':240,'h':110},
  'zoom':{'x':20,'y':560,'w':360,'h':200},
  'previewTime':9.95,'inks':inks,'motion':motion,'decor':decor}]
vid='data:video/mp4;base64,'+base64.b64encode(open('plate.mp4','rb').read()).decode()
s=open('app_template.html').read()
s=s.replace('/*TEMPLATES*/[]',json.dumps(tpl)).replace('/*GLYPHS*/{}',json.dumps({'notebook-hand':glyphs})).replace('/*VIDEOS*/{}',json.dumps({'bday1':vid}))
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
