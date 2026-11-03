"""Gera as figuras do README a partir dos CSVs já medidos, sem rodar os modelos.

Uso:
    .venv/bin/python scripts/leitor/gerar_figuras.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAB = f"{RAIZ}/resultados/tabelas"
FIG = f"{RAIZ}/resultados/figuras"

COR = "#2a78d6"
TEXTO = "#0b0b0b"
TEXTO_SECUNDARIO = "#52514e"


def limpar_eixos(ax):
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(length=0, colors=TEXTO_SECUNDARIO)


def salvar(fig, nome):
    fig.savefig(f"{FIG}/{nome}", dpi=140, bbox_inches="tight")
    print(f"-> {FIG}/{nome}")


def comparacao_rotas():
    """Acurácia por caractere das três rotas medidas na entrega, com a regra."""
    df = pd.read_csv(f"{TAB}/comparacao_pipelines.csv")
    df = df[df["pipeline"].str.contains("com regra")].iloc[::-1]
    nomes = {"A": "Segmentação clássica + CNN",
             "B": "YOLO só nas caixas + mesma CNN",
             "C": "YOLO caixa + classe (rota final)"}
    rotulos = [nomes[p[0]] for p in df["pipeline"]]

    fig, ax = plt.subplots(figsize=(7, 2.4))
    barras = ax.barh(rotulos, df["acc_caractere"], color=COR, height=0.55)
    for barra, valor in zip(barras, df["acc_caractere"]):
        ax.text(valor + 0.012, barra.get_y() + barra.get_height() / 2,
                f"{valor:.3f}".replace(".", ","), va="center", color=TEXTO)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title("Acurácia por caractere nas 30 placas (meta: 0,95)",
                 loc="left", color=TEXTO)
    limpar_eixos(ax)
    salvar(fig, "comparacao_rotas.png")


def erros_posicao():
    """Em quantas das 30 placas cada posição sai errada na rota final."""
    df = pd.read_csv(f"{TAB}/erros_por_posicao.csv")
    linha = df[df["regra"] == "com regra"].iloc[0]
    colunas = [c for c in df.columns if c.startswith("pos")]
    valores = [int(linha[c]) for c in colunas]
    posicoes = [str(i + 1) for i in range(len(colunas))]

    fig, ax = plt.subplots(figsize=(7, 2.6))
    barras = ax.bar(posicoes, valores, color=COR, width=0.6)
    for barra, valor in zip(barras, valores):
        ax.text(barra.get_x() + barra.get_width() / 2, valor + 0.2, str(valor),
                ha="center", color=TEXTO)
    ax.set_ylim(0, max(valores) * 1.15)
    ax.set_yticks([])
    ax.set_xlabel("Posição do caractere na placa", color=TEXTO_SECUNDARIO)
    ax.set_title("Erros por posição na rota final, em 30 placas",
                 loc="left", color=TEXTO)
    limpar_eixos(ax)
    salvar(fig, "erros_por_posicao.png")


if __name__ == "__main__":
    comparacao_rotas()
    erros_posicao()
