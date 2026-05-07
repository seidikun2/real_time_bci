# -----------------------------------------
# EEG simulado + bursts disparados por LSL
# -----------------------------------------

import os, time, math, threading
import numpy as np
from collections import deque
from pylsl import StreamInfo, StreamOutlet, resolve_byprop, StreamInlet

# ================= CONFIG =================
LOG_DIR          = r"C:\Users\User\Desktop\Dados"
SIGNAL_CSV_NAME  = "sim_signal.csv"
MARKER_CSV_NAME  = "sim_markers.csv"

# LSL
MARKER_NAME      = "GrazMI_Markers"
MARKER_TYPE      = "Markers"
SIGNAL_NAME      = "Cognionics Wireless EEG"
SIGNAL_TYPE      = "EEG"

# Sinal
FS               = 250.0
CHANNELS         = 16
CHUNK            = 10
NOISE_STD        = 0.5

# Burst
BURST_FREQ       = 10.0
BURST_AMP        = 1.5
BURST_DUR        = 3.5
TAPER_FRAC       = 0.2

# Perfis hemisféricos
PROFILE_LEFT_MI  = (1.0, 0.3)   # mão direita -> ESQ maior
PROFILE_RIGHT_MI = (0.3, 1.0)   # mão esquerda -> DIR maior

# Códigos
LEFT_MI          = 3
RIGHT_MI         = 4
ATTEMPT          = 5

CODE_MAP = {1:"BASELINE",2:"ATTENTION",3:"LEFT_MI_STIM",4:"RIGHT_MI_STIM",5:"ATTEMPT",6:"REST"}

# ================ Funções =================

def make_outlet():
    info = StreamInfo(SIGNAL_NAME, SIGNAL_TYPE, CHANNELS, FS, 'float32', 'simEEG')
    ch   = info.desc().append_child("channels")
    for i in range(CHANNELS):
        ch.append_child("channel").append_child_value("label", f"CH{i+1}")
    return StreamOutlet(info)

def find_markers():
    while True:
        s = resolve_byprop('name', MARKER_NAME, timeout=1.0) \
            or resolve_byprop('type', MARKER_TYPE, timeout=1.0)
        if s:
            return StreamInlet(s[0], recover=True)
        print("Aguardando marcadores...")

def burst():
    n  = int(BURST_DUR * FS)
    t  = np.arange(n)/FS
    w  = BURST_AMP*np.sin(2*math.pi*BURST_FREQ*t)
    k  = int(TAPER_FRAC*n)
    if k>0:
        ramp   = 0.5-0.5*np.cos(np.linspace(0,math.pi,k))
        w[:k] *= ramp
        w[-k:]*= ramp[::-1]
    return w.astype(np.float32)

def weights(l,r):
    mid = CHANNELS//2
    v   = np.empty(CHANNELS,np.float32)
    v[:mid] = l
    v[mid:] = r
    return v

# Evento sintetizado
class Event:
    def __init__(self,wave,w):
        self.wave = wave
        self.w    = w
        self.i    = 0
    def add(self,x):
        k = min(len(self.wave)-self.i, x.shape[0])
        if k>0:
            x[:k] += np.outer(self.wave[self.i:self.i+k], self.w)
        self.i += k
    def done(self):
        return self.i>=len(self.wave)

# =============== Logs =====================
def open_logs():
    os.makedirs(LOG_DIR,exist_ok=True)
    sig = open(os.path.join(LOG_DIR,SIGNAL_CSV_NAME),"w")
    mrk = open(os.path.join(LOG_DIR,MARKER_CSV_NAME),"w")
    sig.write("t," + ",".join(f"ch{i+1}" for i in range(CHANNELS)) + "\n")
    mrk.write("t,code,label\n")
    return sig,mrk

# ============= Threads =====================
def marker_thread(inlet,q,mrk):
    last = None
    wave = burst()
    while True:
        s,t = inlet.pull_chunk(timeout=0.2)
        if not t:
            continue
        for v,ts in zip(s,t):
            try:
                c = int(v[0])
            except:
                continue
            if c==LEFT_MI:
                last = "L"
            if c==RIGHT_MI:
                last = "R"
            if c==ATTEMPT and last:
                q.append(Event(wave, weights(*PROFILE_RIGHT_MI if last=="L" else PROFILE_LEFT_MI)))
            mrk.write(f"{time.time():.3f},{c},{CODE_MAP.get(c,'UNK')}\n")

def stream(outlet,q,sig):
    rng = np.random.default_rng()
    ev  = []
    dt  = CHUNK/FS
    while True:
        x = rng.normal(0,NOISE_STD,(CHUNK,CHANNELS)).astype(np.float32)
        while q:
            ev.append(q.popleft())
        keep = []
        for e in ev:
            e.add(x)
            if not e.done():
                keep.append(e)
        ev = keep
        outlet.push_chunk(x.tolist())
        t0 = time.time()
        for i in range(CHUNK):
            sig.write(f"{t0+i/FS:.3f}," + ",".join(f"{v:.4f}" for v in x[i]) + "\n")
        time.sleep(dt)

def main():
    outlet    = make_outlet()
    inlet     = find_markers()
    sig, mrk  = open_logs()
    q         = deque()
    threading.Thread(target=marker_thread,args=(inlet,q,mrk),daemon=True).start()
    try:
        stream(outlet,q,sig)
    except KeyboardInterrupt:
        sig.close()
        mrk.close()

if __name__=="__main__":
    main()
