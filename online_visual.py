# viz_pca_lsl.py
import os
import time
import pickle
import threading
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # janela flutuante, amigável ao Spyder
import matplotlib.pyplot as plt
from pylsl import resolve_byprop, StreamInlet

# ============ CONFIG ============
# Prefixo dos artefatos salvos no treino (sem extensão)
SESS_PREFIX = r"C:\Users\seidi\Desktop\Data\TEST\RIEMANN_LDA_screening_2_classes_mi_TEST_Sess0000_Run1"

# Nomes dos streams LSL criados pelo decoder em tempo real
PCA_STREAM_NAME   = "GrazMI_PCA"          # (pca1, pca2)
STATE_STREAM_NAME = "GrazMI_OutputState"  # int32: -1/0/+1

# Rótulos mostrados na tela
STATE_LABEL = {
    -1: "Mão Esquerda",
     0: "Ambos",
     1: "Mão Direita",
}
# ===============================


# ---------- Utils LSL ----------
def resolve_inlet_by_name(name: str, timeout: float = 2.0, retry: float = 1.0) -> StreamInlet:
    print(f"[viz] procurando LSL stream name='{name}' ...")
    while True:
        streams = resolve_byprop('name', name, timeout=timeout)
        if streams:
            si = streams[0]
            print(f"[viz] conectado a '{si.name()}' (type={si.type()}, ch={si.channel_count()}, fs={si.nominal_srate():.2f})")
            return StreamInlet(si, recover=True)
        print("[viz] não encontrado; tentando novamente ...")
        time.sleep(retry)

# ---------- Carregamento de artefatos (range e KDE) ----------
def load_range(prefix: str):
    """
    Tenta carregar o range PCA salvo no treino.
    Primeiro *_range_pca.pkl (novo); se falhar, tenta *_range_red_dim.pkl (compatibilidade).
    """
    cand1 = prefix + "_range_pca.pkl"
    cand2 = prefix + "_range_red_dim.pkl"   # compat
    for p in (cand1, cand2):
        if os.path.exists(p):
            with open(p, "rb") as f:
                vals = pickle.load(f)
            if isinstance(vals, (tuple, list)) and len(vals) == 4:
                print(f"[viz] range carregado de: {p}")
                return vals  # (x_min, x_max, y_min, y_max)
    print("[viz] range não encontrado — usando autoescala temporária.")
    return None

def load_kde(prefix: str):
    """
    Tenta carregar *_kde.pkl. Suporta chaves 'kde0','kde1',... ou nomes custom.
    Se não existir, retorna None.
    """
    p = prefix + "_kde.pkl"
    if not os.path.exists(p):
        print("[viz] KDE não encontrado (opcional).")
        return None
    try:
        with open(p, "rb") as f:
            d = pickle.load(f)
        print(f"[viz] KDE carregado: {p}")
        return d
    except Exception as e:
        print(f"[viz] falha ao ler KDE: {e}")
        return None

def draw_kde_background(ax, kde_dict, xrange, yrange):
    """
    Desenha contornos KDE se disponível.
    Aceita chaves como 'kde0','kde1',... ou nomes tipo 'kde_left_hand', etc.
    """
    if kde_dict is None:
        return
    x_min, x_max, y_min, y_max = xrange[0], xrange[1], yrange[0], yrange[1]
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    # selecione até 5 mapas de cor para classes
    cmaps = ["Greys", "Blues", "Reds", "Oranges", "Purples"]

    # ordena chaves para ter reprodutibilidade
    keys = sorted(list(kde_dict.keys()))
    for i, k in enumerate(keys):
        try:
            kde = kde_dict[k]
            Z = kde.evaluate(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            ax.contourf(xx, yy, Z,
                        alpha=0.35,
                        levels=np.linspace(Z.min(), Z.max(), 10),
                        cmap=cmaps[i % len(cmaps)])
        except Exception:
            # ignora chaves incompatíveis
            continue

# ---------- Threads de leitura LSL ----------
class Shared:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.state = 0
        self.lock = threading.Lock()

def pca_listener(shared: Shared):
    inlet = resolve_inlet_by_name(PCA_STREAM_NAME)
    while True:
        try:
            samp, ts = inlet.pull_sample(timeout=0.2)
            if samp is None:
                continue
            with shared.lock:
                # samp = [pca1, pca2]
                shared.x = float(samp[0])
                shared.y = float(samp[1])
        except KeyboardInterrupt:
            break
        except Exception:
            time.sleep(0.05)

def state_listener(shared: Shared):
    inlet = resolve_inlet_by_name(STATE_STREAM_NAME)
    while True:
        try:
            samp, ts = inlet.pull_sample(timeout=0.2)
            if samp is None:
                continue
            st = int(samp[0])
            if st < -1: st = -1
            if st > +1: st = +1
            with shared.lock:
                shared.state = st
        except KeyboardInterrupt:
            break
        except Exception:
            time.sleep(0.05)

# ---------- Plot em tempo real ----------
def make_figure(xrange=None, yrange=None):
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_axis_off()

    # tenta forçar janela em primeiro plano
    try:
        mng = plt.get_current_fig_manager()
        mng.window.attributes('-topmost', True)
        # libera o "sempre no topo" depois de trazer à frente
        fig.canvas.flush_events()
        mng.window.attributes('-topmost', False)
    except Exception:
        pass

    # limites
    if xrange and yrange:
        ax.set_xlim(*xrange)
        ax.set_ylim(*yrange)
    else:
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)

    # ponto do PCA
    scat = ax.scatter([0], [0], s=200, c="k")

    # rótulo grande no centro
    txt = ax.text(0.5, 0.5, "Aguardando...", transform=ax.transAxes,
                  ha="center", va="center", fontsize=28)

    return fig, ax, scat, txt

def main():
    # Carrega range e KDE (opcional)
    rng = load_range(SESS_PREFIX)
    kde = load_kde(SESS_PREFIX)

    if rng is None:
        xrange = (-1, 1)
        yrange = (-1, 1)
    else:
        x_min, x_max, y_min, y_max = rng
        # pequena margem
        dx = (x_max - x_min) * 0.05 or 0.5
        dy = (y_max - y_min) * 0.05 or 0.5
        xrange = (x_min - dx, x_max + dx)
        yrange = (y_min - dy, y_max + dy)

    fig, ax, scat, txt = make_figure(xrange, yrange)

    # desenha background KDE se disponível
    draw_kde_background(ax, kde, xrange, yrange)

    # Compartilhado entre threads
    shared = Shared()

    # Threads para LSL
    t1 = threading.Thread(target=pca_listener, args=(shared,), daemon=True)
    t2 = threading.Thread(target=state_listener, args=(shared,), daemon=True)
    t1.start()
    t2.start()

    # Loop de atualização do plot
    try:
        while True:
            with shared.lock:
                x, y, st = shared.x, shared.y, shared.state
            scat.set_offsets(np.c_[[x], [y]])
            txt.set_text(STATE_LABEL.get(st, "Ambos"))

            # Atualiza rápido, mantendo a janela fluída
            plt.pause(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()
