"""Dia 6 - Ablacao da segmentacao: quanto cada correcao do Dia 6 aproxima a
fatia de inferencia do recorte que a CNN viu no treino.

Roda LOCAL (Mac, .venv), sem Colab e sem GPU. As duas primeiras partes nem
precisam da CNN.

Contexto: o Dia 5 fechou com acuracia por caractere de 0,051 e a causa
"nao identificada". Foram achados dois bugs empilhados em segmentar():

  Bug 1 - corte_superior=0.35 era aplicado mesmo quando a entrada ja era so
          a faixa dos caracteres (recortar_via_caixas), decapitando o topo
          de cada caractere antes da segmentacao comecar.
  Bug 2 - o recorte de cada fatia era fonte[:, x:x+w] -- justo em X, altura
          INTEIRA da faixa -- enquanto o treino da CNN recorta justo nos
          DOIS eixos (04_cnn_caracteres.py: img[y1:y2, x1:x2]).

Este script mede o efeito de cada um separadamente, de duas formas:

  Parte A (sem modelo) - compara a fatia produzida com o "alvo": o recorte
      no estilo do treino, feito da imagem original pela caixa anotada.
      Metrica: fracao de pixels iguais entre fatia e alvo (32x32 binarias).
  Parte B (com a CNN)  - acuracia por caractere de verdade, variante a
      variante.

Uso:
    .venv/bin/python notebooks/05b_ablacao_segmentacao.py
"""

import glob
import os
import sys

import cv2
import numpy as np
import pandas as pd
import yaml

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from src.preprocessamento import preparar, segmentar  # noqa: E402
from src.validacao import CLASSES  # noqa: E402

DADOS_CHARS = f"{RAIZ}/dados/caracteres"
# O .keras salvo pela VM (Keras 3.13) nao abre no Keras 3.10 local (Python
# 3.9 nao aceita keras>=3.11). `cnn_chars_compat.keras` e o MESMO modelo
# com a chave `quantization_config` removida do config -- mesmos 359.588
# parametros, mesmos pesos.
CAMINHO_CNN = f"{RAIZ}/modelos/cnn_chars_compat.keras"
if not os.path.exists(CAMINHO_CNN):
    CAMINHO_CNN = f"{RAIZ}/modelos/cnn_chars.keras"
SAIDA_TAB = f"{RAIZ}/resultados/tabelas"
SAIDA_FIG = f"{RAIZ}/resultados/figuras"

cfg = yaml.safe_load(open(f"{DADOS_CHARS}/data.yaml"))
nomes = cfg["names"] if isinstance(cfg["names"], list) else list(cfg["names"].values())


# ---------------------------------------------------------------
# Reconstrucao das placas de teste (mesma logica do Dia 5, pra ser
# comparavel numero a numero com o 05_pipeline_final.py)
# ---------------------------------------------------------------
def montar_placas(split, n_esperado=7):
    """Reconstroi o texto de cada placa ordenando as caixas de caractere por X."""
    registros = []
    descartadas = dict(sem_rotulo=0, tamanho_diferente_de_7=0)
    for caminho in sorted(glob.glob(f"{DADOS_CHARS}/{split}/images/*")):
        rot = caminho.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        if not os.path.exists(rot):
            descartadas["sem_rotulo"] += 1
            continue
        caracteres = []
        for linha in open(rot):
            partes = linha.split()
            if len(partes) < 5:
                continue
            classe = str(nomes[int(partes[0])]).upper()
            if classe not in CLASSES:          # ignora EUR / "-"
                continue
            xc, yc, bw, bh = map(float, partes[1:5])
            caracteres.append((xc, yc, bw, bh, classe))
        if len(caracteres) != n_esperado:
            descartadas["tamanho_diferente_de_7"] += 1
            continue
        caracteres.sort(key=lambda t: t[0])
        registros.append(dict(arquivo=os.path.basename(caminho),
                              real="".join(c[-1] for c in caracteres),
                              caixas=caracteres))
    return registros, descartadas


def recortar_via_caixas(img, caixas, margem=0.12):
    """Estima o angulo real pela reta dos centros dos caracteres, desrotaciona
    e recorta so a regiao da placa. Copia fiel do Dia 5, pra comparabilidade."""
    h_img, w_img = img.shape[:2]
    xs = np.array([c[0] * w_img for c in caixas])
    ys = np.array([c[1] * h_img for c in caixas])
    inclinacao, _ = np.polyfit(xs, ys, 1)
    angulo = np.degrees(np.arctan(inclinacao))

    matriz = cv2.getRotationMatrix2D((w_img / 2, h_img / 2), angulo, 1.0)
    girada = cv2.warpAffine(img, matriz, (w_img, h_img), flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE)

    pts = np.stack([xs, ys, np.ones_like(xs)], axis=1) @ matriz.T
    larguras = np.array([c[2] * w_img for c in caixas])
    alturas = np.array([c[3] * h_img for c in caixas])
    x_min = (pts[:, 0] - larguras / 2).min()
    x_max = (pts[:, 0] + larguras / 2).max()
    y_min = (pts[:, 1] - alturas / 2).min()
    y_max = (pts[:, 1] + alturas / 2).max()

    mx, my = (x_max - x_min) * margem, (y_max - y_min) * margem
    x1, y1 = max(0, int(x_min - mx)), max(0, int(y_min - my))
    x2, y2 = min(w_img, int(x_max + mx)), min(h_img, int(y_max + my))
    return girada[y1:y2, x1:x2]


def recorte_estilo_treino(img, caixa, saida=(32, 32)):
    """Reproduz EXATAMENTE o que 04_cnn_caracteres.py faz pra gerar um exemplo
    de treino: recorte justo nos dois eixos pela caixa anotada, CLAHE, Otsu
    local, resize 32x32. E o "alvo" com que a fatia de inferencia e comparada."""
    h, w = img.shape[:2]
    xc, yc, bw, bh = caixa[:4]
    x1, y1 = max(0, int((xc - bw / 2) * w)), max(0, int((yc - bh / 2) * h))
    x2, y2 = min(w, int((xc + bw / 2) * w)), min(h, int((yc + bh / 2) * h))
    rec = img[y1:y2, x1:x2]
    if rec.size == 0 or rec.shape[0] < 8 or rec.shape[1] < 5:
        return None
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cinza = clahe.apply(cv2.cvtColor(rec, cv2.COLOR_BGR2GRAY))
    binaria = cv2.threshold(cinza, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return cv2.resize(binaria, saida, interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------
# As variantes da ablacao
# ---------------------------------------------------------------
# `apertar` False reproduz o comportamento do Dia 5 (altura inteira da faixa).
VARIANTES = [
    ("Dia 5 (baseline)",          0.35, False, 0.0),
    ("so corte_superior=0",       0.0,  False, 0.0),
    ("so aperto vertical",        0.35, True,  0.0),
    ("corte=0 + aperto",          0.0,  True,  0.0),
    ("corte=0 + aperto + folga",  0.0,  True,  0.10),
]


def segmentar_variante(binaria, realcada, corte, apertar, folga):
    """Chama segmentar() na variante pedida. Quando `apertar` e False,
    reproduz o bug do Dia 5 recortando a altura inteira da faixa."""
    if apertar:
        return segmentar(binaria, cinza=realcada, corte_superior=corte,
                         margem_vertical=folga)
    # --- reproducao fiel do codigo do Dia 5, so pra servir de baseline ---
    from src.preprocessamento import (_componentes_de_caracteres,
                                      _juntar_ou_dividir, projecao_vertical)
    faixa, perfil = projecao_vertical(binaria, corte)
    caixas = _componentes_de_caracteres(faixa)
    if not caixas:
        lf = faixa.shape[1] / 7
        caixas = [(round(i * lf), 0, round((i + 1) * lf) - round(i * lf),
                   faixa.shape[0]) for i in range(7)]
    caixas = _juntar_ou_dividir(caixas, 7, perfil, faixa.shape[0])
    y0 = int(realcada.shape[0] * corte)
    fonte = realcada[y0:, :]
    fatias = []
    for x, _, w, _ in caixas:
        pedaco = fonte[:, x:x + w]                      # <<< o bug: so em X
        if pedaco.size == 0:
            fatias.append(np.zeros((32, 32), dtype=np.uint8))
            continue
        pedaco = cv2.threshold(pedaco, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        fatias.append(cv2.resize(pedaco, (32, 32), interpolation=cv2.INTER_AREA))
    return fatias


def main():
    registros, descartadas = montar_placas("test")
    print(f"Placas de teste reconstruidas: {len(registros)}  (descartadas: {descartadas})")
    if not registros:
        print("ERRO: nenhuma placa reconstruida — confira dados/caracteres/test/")
        return

    # ---------- Parte A: sem modelo ----------
    # Pra cada variante, o quanto a fatia se parece com o recorte de treino.
    print("\n=== Parte A — semelhanca com o recorte estilo treino (sem modelo) ===")
    linhas_a = []
    for nome, corte, apertar, folga in VARIANTES:
        acordos, tintas = [], []
        for r in registros:
            img = cv2.imread(f"{DADOS_CHARS}/test/images/{r['arquivo']}")
            if img is None:
                continue
            alvos = [recorte_estilo_treino(img, c) for c in r["caixas"]]
            recorte = recortar_via_caixas(img, r["caixas"])
            _, _, realcada, binaria = preparar(recorte)
            fatias = segmentar_variante(binaria, realcada, corte, apertar, folga)
            for alvo, fatia in zip(alvos, fatias):
                if alvo is None:
                    continue
                # fracao de pixels iguais entre as duas binarias 32x32
                acordos.append(float(((alvo > 127) == (fatia > 127)).mean()))
                tintas.append(float((fatia > 127).mean()))
        linhas_a.append(dict(variante=nome, corte_superior=corte,
                             aperto_vertical=apertar, folga=folga,
                             acordo_com_treino=round(np.mean(acordos), 4),
                             fracao_tinta_media=round(np.mean(tintas), 4),
                             n_caracteres=len(acordos)))
        print(f"  {nome:28s} acordo={linhas_a[-1]['acordo_com_treino']:.4f}  "
              f"tinta={linhas_a[-1]['fracao_tinta_media']:.4f}")

    df_a = pd.DataFrame(linhas_a)
    os.makedirs(SAIDA_TAB, exist_ok=True)
    df_a.to_csv(f"{SAIDA_TAB}/ablacao_semelhanca_treino.csv", index=False)

    # referencia: quanto os proprios alvos "concordam" consigo (1.0) e qual a
    # fracao de tinta tipica de um exemplo de treino -- serve de alvo pra coluna
    tintas_alvo = []
    for r in registros:
        img = cv2.imread(f"{DADOS_CHARS}/test/images/{r['arquivo']}")
        if img is None:
            continue
        for c in r["caixas"]:
            a = recorte_estilo_treino(img, c)
            if a is not None:
                tintas_alvo.append(float((a > 127).mean()))
    print(f"\n  referencia (exemplos de treino): fracao de tinta = {np.mean(tintas_alvo):.4f}")

    # ---------- Parte B: com a CNN ----------
    if not os.path.exists(CAMINHO_CNN):
        print(f"\n[Parte B pulada] modelo nao encontrado em {CAMINHO_CNN}")
        print("Baixe do Drive e rode de novo pra ter a acuracia por caractere.")
        return

    print("\n=== Parte B — acuracia por caractere com a CNN ===")
    from tensorflow import keras
    from src.metricas import acuracia_caractere, acuracia_placa
    cnn = keras.models.load_model(CAMINHO_CNN)

    linhas_b = []
    for nome, corte, apertar, folga in VARIANTES:
        reais, previstas = [], []
        for r in registros:
            img = cv2.imread(f"{DADOS_CHARS}/test/images/{r['arquivo']}")
            if img is None:
                continue
            recorte = recortar_via_caixas(img, r["caixas"])
            _, _, realcada, binaria = preparar(recorte)
            fatias = segmentar_variante(binaria, realcada, corte, apertar, folga)
            lote = np.stack(fatias).astype("float32")[..., None]
            idx = cnn.predict(lote, verbose=0).argmax(axis=1)
            reais.append(r["real"])
            previstas.append("".join(CLASSES[k] for k in idx))
        # SEM mascara brasileira: o gabarito aqui e europeu (classe EUR no
        # data.yaml), entao aplicar LLLDDDD/LLLDLDD so degradaria. O Dia 5
        # aplicou indevidamente.
        acc_c = acuracia_caractere(reais, previstas)
        acc_p = acuracia_placa(reais, previstas)
        linhas_b.append(dict(variante=nome, corte_superior=corte,
                             aperto_vertical=apertar, folga=folga,
                             acc_caractere=round(acc_c, 4),
                             acc_placa=round(acc_p, 4)))
        print(f"  {nome:28s} acc_caractere={acc_c:.4f}  acc_placa={acc_p:.4f}")

    pd.DataFrame(linhas_b).to_csv(f"{SAIDA_TAB}/ablacao_segmentacao.csv", index=False)
    print(f"\nTabelas salvas em {SAIDA_TAB}/")

    # ---------- figura: a prova visual do bug do corte_superior ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(SAIDA_FIG, exist_ok=True)
    mostrar = registros[:3]
    fig, eixos = plt.subplots(len(mostrar) * 3, 8, figsize=(12, len(mostrar) * 5))
    for li, r in enumerate(mostrar):
        img = cv2.imread(f"{DADOS_CHARS}/test/images/{r['arquivo']}")
        alvos = [recorte_estilo_treino(img, c) for c in r["caixas"]]
        recorte = recortar_via_caixas(img, r["caixas"])
        _, _, realcada, binaria = preparar(recorte)
        faixas = [
            ("alvo: recorte do treino", alvos),
            ("Dia 5: corte_superior=0.35", segmentar_variante(binaria, realcada, 0.35, False, 0.0)),
            ("Dia 6: corte=0 + folga 10%", segmentar_variante(binaria, realcada, 0.0, True, 0.10)),
        ]
        for vi, (nome, imgs) in enumerate(faixas):
            linha = li * 3 + vi
            eixos[linha, 0].text(0.5, 0.5, f"{r['real']}\n{nome}", fontsize=7,
                                 ha="center", va="center")
            eixos[linha, 0].axis("off")
            for k in range(7):
                ax = eixos[linha, k + 1]
                im = imgs[k] if k < len(imgs) and imgs[k] is not None else np.zeros((32, 32), np.uint8)
                ax.imshow(im, cmap="gray", vmin=0, vmax=255)
                ax.set_title(r["real"][k], fontsize=7)
                ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{SAIDA_FIG}/segmentacao_antes_depois.png", dpi=140, bbox_inches="tight")
    print(f"Figura salva em {SAIDA_FIG}/segmentacao_antes_depois.png")


if __name__ == "__main__":
    main()
