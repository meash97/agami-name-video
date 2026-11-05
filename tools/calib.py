import subprocess, sys, cv2, numpy as np, os
h,w=1292,724; M=cv2.getRotationMatrix2D((w/2,h/2),-20.5,1.0)
def measure(img, box):
    r=cv2.warpAffine(img,M,(w,h),flags=cv2.INTER_CUBIC)
    g=cv2.cvtColor(r,cv2.COLOR_BGR2GRAY)[box[1]:box[3],box[0]:box[2]].astype(np.float32)
    paper=np.percentile(g,85); b=(g<paper-70).astype(np.uint8)
    cs,_=cv2.findContours(b,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE); per=sum(cv2.arcLength(c,True) for c in cs)
    return 2*b.sum()/per, float(np.mean(g[b>0]))
for combo in sys.argv[1:]:
    boost,gamma,thr=combo.split(',')
    env=dict(os.environ,BOOST=boost,GAMMA=gamma,THR=thr)
    subprocess.run('python3 build_glyphs.py >/dev/null && python3 assemble.py >/dev/null && rm -f app_out.* && timeout 120 python3 test_app.py SAURABH make >/dev/null 2>&1 && ffmpeg -v error -y -ss 9.95 -i app_out.mp4 -frames:v 1 calib_last.png',shell=True,env=env,check=True)
    out=cv2.imread('calib_last.png')
    n=measure(out,(60,583,218,627)); wi=measure(out,(283,576,400,627)); ha=measure(out,(148,638,296,686))
    print(f'BOOST {boost} GAMMA {gamma} THR {thr}: name stroke {n[0]:.2f} tone {n[1]:.1f} | WISHING {wi[0]:.2f}/{wi[1]:.1f} | HAPPIEST {ha[0]:.2f}/{ha[1]:.1f}',flush=True)
