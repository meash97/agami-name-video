import subprocess, sys, cv2, numpy as np, os
sys.path.insert(0,'/home/claude/t3'); os.chdir('/home/claude/t3'); from common import *; os.chdir('/home/claude/vid')
def measure(img, box):
    g=cv2.cvtColor(level(img),cv2.COLOR_BGR2GRAY)[box[1]:box[3],box[0]:box[2]].astype(np.float32)
    paper=np.percentile(g,85); b=(g<paper-70).astype(np.uint8)
    cs,_=cv2.findContours(b,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE); per=sum(cv2.arcLength(c,True) for c in cs)
    return 2*b.sum()/per, float(np.mean(g[b>0]))
for combo in sys.argv[1:]:
    boost,thr=combo.split(',')
    env=dict(os.environ,BOOST=boost,THR=thr,TPL='thanks-gold-1')
    subprocess.run('cd /home/claude/t3 && python3 build_glyphs.py >/dev/null && cd /home/claude/vid && python3 assemble.py >/dev/null && rm -f app_out.* && timeout 150 python3 test_app.py "SANTOSH JI CHHAJAR" make >/dev/null 2>&1 && ffmpeg -v error -y -ss 9.95 -i app_out.mp4 -frames:v 1 calib_last.png',shell=True,env=env,check=True)
    out=cv2.imread('calib_last.png'); n=measure(out,(236,606,540,646)); d=measure(out,(134,606,222,646)); l2=measure(out,(120,662,580,700))
    print(f'BOOST {boost} THR {thr}: name {n[0]:.2f}/{n[1]:.1f} | DEAR {d[0]:.2f}/{d[1]:.1f} | line2 {l2[0]:.2f}/{l2[1]:.1f}',flush=True)
