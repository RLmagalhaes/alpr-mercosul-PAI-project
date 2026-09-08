"""Dia 6 - Monta o conjunto de teste Mercosul com gabarito humano.

POR QUE: o dataset de deteccao (trafficbr) tem a CAIXA da placa anotada, mas
nao o TEXTO. O dataset de caracteres (project-swcsj) tem o texto, mas e de
placas EUROPEIAS -- serve pra medir a segmentacao, nunca o sistema no alvo real
do projeto. A meta "acuracia por placa > 0,80" no formato Mercosul so pode ser
medida com um gabarito humano. Sao ~25 placas digitadas a mao, uma vez.

Duas partes:
  PARTE 1 (este script, modo "gerar") - recorta as placas mais legiveis do
      split de teste, amplia, e gera uma figura numerada + um CSV com a coluna
      `texto` vazia pro Raphael preencher.
  PARTE 2 (modo "medir", depois de preenchido) - roda o LeitorDePlacas de ponta
      a ponta nessas fotos e calcula acuracia por caractere e por placa.

Uso:
    .venv/bin/python notebooks/06_gabarito_mercosul.py gerar
    # (preencher a coluna `texto` em resultados/tabelas/gabarito_mercosul.csv)
    .venv/bin/python notebooks/06_gabarito_mercosul.py medir
"""

import glob
import os
import sys

import cv2
import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from src.preprocessamento import recortar  # noqa: E402

BASE = f"{RAIZ}/dados/deteccao/vehicle-plates"
SAIDA_TAB = f"{RAIZ}/resultados/tabelas"
SAIDA_FIG = f"{RAIZ}/resultados/figuras"
CSV = f"{SAIDA_TAB}/gabarito_mercosul.csv"

# O Dia 2 registrou que 76% dos recortes tem altura<40px ou largura<100px e
# saem borrados. Um gabarito duvidoso e pior que gabarito nenhum, entao so
# entram os que passam desse corte.
# O split de TESTE tem so 3 caixas com >= 100x40px depois de filtrar as
# anotacoes erradas -- pouco demais pra um conjunto de avaliacao. O split de
# VALID tem 43. Entao o gabarito usa os dois, com a origem registrada no CSV:
# as de `test` sao held-out puro; as de `valid` foram usadas pra SELECIONAR o
# best.pt do detector (viés otimista leve na parte de deteccao), mas sao
# ineditas pro reconhecimento, que e o que este conjunto mede de fato -- a CNN
# foi treinada noutro dataset (project-swcsj).
SPLITS = ["test", "valid"]
ALTURA_MIN, LARGURA_MIN = 32, 90
N_CANDIDATAS = 30          # folga sobre as ~25, pra descartar ilegiveis na hora

# O dataset trafficbr tem ANOTACOES ERRADAS: as 12 maiores caixas rotuladas
# "plate" tem razao largura/altura entre 0,71 e 1,70 e cobrem 45-63% da imagem
# -- sao carros inteiros, nao placas. Ordenar por area sem filtrar seleciona
# justamente esses erros. Uma placa real neste dataset tem razao mediana 2,20
# (p10=0,99, p90=2,78) e ocupa 0,69% da imagem na mediana.
# (Isso provavelmente tambem explica os piores casos do Dia 2, com IoU 0,01-0,03:
#  erro de anotacao, nao erro do detector.)
RAZAO_MIN, RAZAO_MAX = 1.8, 5.0
FRACAO_MAX = 0.25          # caixa que cobre mais de 25% da foto nao e placa


def candidatas():
    """Recorta as placas anotadas e devolve as mais legiveis, ordenadas por
    area (maior = mais nitida). Prioriza o split `test` (held-out puro)."""
    achadas = []
    for split in SPLITS:
      for caminho in sorted(glob.glob(f"{BASE}/{split}/images/*")):
        rot = caminho.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        if not os.path.exists(rot):
            continue
        img = cv2.imread(caminho)
        if img is None:
            continue
        h, w = img.shape[:2]
        for linha in open(rot):
            partes = linha.split()
            if len(partes) < 5:
                continue
            xc, yc, bw, bh = map(float, partes[1:5])
            if not (RAZAO_MIN <= (bw * w) / max(bh * h, 1e-9) <= RAZAO_MAX):
                continue                      # formato incompativel com placa
            if bw * bh > FRACAO_MAX:
                continue                      # grande demais pra ser placa
            caixa = ((xc - bw / 2) * w, (yc - bh / 2) * h,
                     (xc + bw / 2) * w, (yc + bh / 2) * h)
            rec = recortar(img, caixa)
            if rec.size == 0:
                continue
            if rec.shape[0] < ALTURA_MIN or rec.shape[1] < LARGURA_MIN:
                continue
            achadas.append(dict(split=split, arquivo=os.path.basename(caminho),
                                caixa=[round(v, 1) for v in caixa],
                                area=rec.shape[0] * rec.shape[1], recorte=rec))
    # test primeiro (held-out puro), depois valid; dentro de cada um, maior area
    achadas.sort(key=lambda d: (SPLITS.index(d["split"]), -d["area"]))
    return achadas


def gerar():
    todas = candidatas()
    por_split = {sp: sum(1 for d in todas if d["split"] == sp) for sp in SPLITS}
    print(f"Placas legiveis (>= {LARGURA_MIN}x{ALTURA_MIN}px): {len(todas)}  {por_split}")
    escolhidas = todas[:N_CANDIDATAS]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(SAIDA_FIG, exist_ok=True)
    n = len(escolhidas)
    fig, eixos = plt.subplots(n, 1, figsize=(9, 1.5 * n))
    if n == 1:
        eixos = [eixos]
    for i, (ax, d) in enumerate(zip(eixos, escolhidas), start=1):
        rec = d["recorte"]
        # amplia 4x pra facilitar a leitura a olho
        grande = cv2.resize(rec, (rec.shape[1] * 4, rec.shape[0] * 4),
                            interpolation=cv2.INTER_CUBIC)
        ax.imshow(cv2.cvtColor(grande, cv2.COLOR_BGR2RGB))
        ax.set_title(f"[{i:02d}]  {d['split']}  {d['arquivo'][:38]}  ({rec.shape[1]}x{rec.shape[0]}px)",
                     fontsize=8, loc="left")
        ax.axis("off")
    plt.tight_layout()
    caminho_fig = f"{SAIDA_FIG}/gabarito_para_rotular.png"
    plt.savefig(caminho_fig, dpi=140, bbox_inches="tight")

    os.makedirs(SAIDA_TAB, exist_ok=True)
    pd.DataFrame([dict(n=i, split=d["split"], arquivo=d["arquivo"], caixa=str(d["caixa"]),
                       largura=d["recorte"].shape[1], altura=d["recorte"].shape[0],
                       texto="")
                  for i, d in enumerate(escolhidas, start=1)]).to_csv(CSV, index=False)

    print(f"\nFigura : {caminho_fig}")
    print(f"CSV    : {CSV}")
    print("\nPreencha a coluna `texto` do CSV com a placa de cada linha (7 caracteres,")
    print("sem espaco nem hifen). Linha ilegivel: deixe em branco, que o script pula.")


def medir():
    if not os.path.exists(CSV):
        print(f"ERRO: {CSV} nao existe. Rode antes: python {sys.argv[0]} gerar")
        return
    df = pd.read_csv(CSV, dtype={"texto": str}).fillna({"texto": ""})
    df["texto"] = df["texto"].str.strip().str.upper()
    preenchidas = df[df["texto"] != ""]
    print(f"Linhas com gabarito preenchido: {len(preenchidas)} de {len(df)}")
    if preenchidas.empty:
        print("Nada pra medir ainda.")
        return
    ruins = preenchidas[preenchidas["texto"].str.len() != 7]
    if len(ruins):
        print(f"AVISO: {len(ruins)} linha(s) com tamanho != 7, serao ignoradas:")
        print(ruins[["n", "arquivo", "texto"]].to_string(index=False))
        preenchidas = preenchidas[preenchidas["texto"].str.len() == 7]

    from src.metricas import acuracia_caractere, acuracia_placa, erros_por_posicao
    from src.pipeline import LeitorDePlacas

    cnn = f"{RAIZ}/modelos/cnn_chars_compat.keras"
    if not os.path.exists(cnn):
        cnn = f"{RAIZ}/modelos/cnn_chars.keras"
    leitor = LeitorDePlacas(f"{RAIZ}/modelos/detector_best.pt", cnn)

    linhas = []
    for _, r in preenchidas.iterrows():
        res = leitor.ler(f"{BASE}/{r['split']}/images/{r['arquivo']}")
        linhas.append(dict(
            split=r["split"], arquivo=r["arquivo"], real=r["texto"],
            bruto=res.get("placa_sem_regra", "") or "",
            final=res.get("placa_com_regra", "") or "",
            layout=res.get("layout", ""), status=res.get("status", ""),
            conf_minima=res.get("conf_minima", 0.0)))
    res_df = pd.DataFrame(linhas)
    res_df.to_csv(f"{SAIDA_TAB}/predicoes_mercosul.csv", index=False)

    reais = res_df["real"].tolist()
    metricas = []
    for nome, col in [("CNN sozinha", "bruto"), ("CNN + regra do formato", "final")]:
        prev = res_df[col].tolist()
        metricas.append(dict(versao=nome,
                             acc_caractere=round(acuracia_caractere(reais, prev), 4),
                             acc_placa=round(acuracia_placa(reais, prev), 4)))
    m = pd.DataFrame(metricas)
    m.to_csv(f"{SAIDA_TAB}/metricas_mercosul.csv", index=False)
    print("\n=== Mercosul, gabarito humano ===")
    print(m.to_string(index=False))
    print("\nErros por posicao (1-7):",
          erros_por_posicao(reais, res_df["final"].tolist()))
    print(f"\nDetalhe por placa: {SAIDA_TAB}/predicoes_mercosul.csv")


if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "gerar"
    (gerar if modo == "gerar" else medir)()
