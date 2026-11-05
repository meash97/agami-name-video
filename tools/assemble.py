import json, base64, cv2
glyphs=json.load(open('glyphs.json')); inks=json.load(open('inks.json')); motion=json.load(open('motion.json')); decor=json.load(open('decor.json'))
th=cv2.imread('plate/0241.png'); th=cv2.resize(th,(181,323),interpolation=cv2.INTER_AREA)
ok,buf=cv2.imencode('.jpg',th,[cv2.IMWRITE_JPEG_QUALITY,80])
tpl=[{
  'id':'bday-notebook-1','title':'Birthday · Notebook','fileTag':'Birthday',
  'video':'bday1','glyphs':'notebook-hand','thumb':'data:image/jpeg;base64,'+base64.b64encode(buf).decode(),
  'w':724,'h':1292,'fps':24,'angle':20.5,'sc':3,'pad':3,
  'anchor':{'right':217,'left':12,'baseline':619},
  'gap':5,'space':11,'minGap':2,'minSpace':6,'minCondense':0.8,'suffix':'JI,','maxLen':16,
  'region':{'x':0,'y':540,'w':240,'h':110},
  'zoom':{'x':20,'y':560,'w':360,'h':200},
  'previewTime':9.95,'inks':inks,'motion':motion,'decor':decor}]
vid='data:video/mp4;base64,'+base64.b64encode(open('plate.mp4','rb').read()).decode()
s=open('app_template.html').read()
s=s.replace('/*TEMPLATES*/[]',json.dumps(tpl)).replace('/*GLYPHS*/{}',json.dumps({'notebook-hand':glyphs})).replace('/*VIDEOS*/{}',json.dumps({'bday1':vid}))
open('/home/claude/AGAMI-Name-Video-Maker.html','w').write(s)
import os; print('HTML MB',round(os.path.getsize('/home/claude/AGAMI-Name-Video-Maker.html')/1048576,2))
