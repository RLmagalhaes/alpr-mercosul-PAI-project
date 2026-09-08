"""Dia 6 - Compara os tres pipelines sobre o MESMO conjunto de 30 placas
brasileiras com gabarito humano (resultados/tabelas/gabarito_mercosul.csv).

  A) classico   - preparar + segmentar (componentes conectados) + CNN do Dia 4
  B) yolo+cnn   - YOLO de caracteres so pras CAIXAS, classificacao pela CNN
  C) yolo puro  - o YOLO de caracteres da a caixa E a classe

B isola o efeito de trocar SO a segmentacao (mesma CNN dos dois lados);
C mostra o que se ganha trocando tambem o classificador.

Roda LOCAL (.venv), sem Colab.

Uso:
    .venv/bin/python notebooks/07_comparar_pipelines.py
"""

import os
import sys

import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from src.metricas import acuracia_caractere, acuracia_placa, erros_por_posicao  # noqa: E402

BASE = f"{RAIZ}/dados/deteccao/vehicle-plates"
TAB = f"{RAIZ}/resultados/tabelas"
DETECTOR = f"{RAIZ}/modelos/detector_best.pt"
CHARS = f"{RAIZ}/modelos/chars_best.pt"
CNN = (f"{RAIZ}/modelos/cnn_chars_compat.keras"
       if os.path.exists(f"{RAIZ}/modelos/cnn_chars_compat.keras")
       else f"{RAIZ}/modelos/cnn_chars.keras")


def main():
    gab = pd.read_csv(f"{TAB}/gabarito_mercosul.csv", dtype=str).fillna("")
    gab["texto"] = gab["texto"].str.strip().str.upper()
    gab = gab[gab["texto"].str.len() == 7]
    print(f"Placas com gabarito: {len(gab)}")

    from src.leitor_yolo import LeitorDePlacasYOLO
    from src.pipeline import LeitorDePlacas

    classico = LeitorDePlacas(DETECTOR, CNN)
    novo = LeitorDePlacasYOLO(DETECTOR, CHARS, CNN)

    linhas = []
    for _, r in gab.iterrows():
        caminho = f"{BASE}/{r['split']}/images/{r['arquivo']}"
        a = classico.ler(caminho)
        b = novo.ler(caminho, modo="cnn")
        c = novo.ler(caminho, modo="yolo")
        linhas.append(dict(
            n=r["n"], real=r["texto"],
            classico_bruto=a.get("placa_sem_regra", "") or "",
            classico=a.get("placa_com_regra", "") or "",
            yolo_cnn_bruto=b.get("placa_sem_regra", "") or "",
            yolo_cnn=b.get("placa_com_regra", "") or "",
            yolo_bruto=c.get("placa_sem_regra", "") or "",
            yolo=c.get("placa_com_regra", "") or "",
            n_chars_yolo=b.get("n_caracteres", 0),
            layout=b.get("layout", ""),
            conf_min_yolo=c.get("conf_minima", 0.0)))
    det = pd.DataFrame(linhas)
    det.to_csv(f"{TAB}/comparacao_pipelines_detalhe.csv", index=False)

    reais = det["real"].tolist()
    resumo = []
    for nome, col in [
        ("A) classico (segmentar) — sem regra", "classico_bruto"),
        ("A) classico (segmentar) — com regra", "classico"),
        ("B) YOLO caixas + CNN — sem regra", "yolo_cnn_bruto"),
        ("B) YOLO caixas + CNN — com regra", "yolo_cnn"),
        ("C) YOLO caixa+classe — sem regra", "yolo_bruto"),
        ("C) YOLO caixa+classe — com regra", "yolo"),
    ]:
        prev = det[col].tolist()
        resumo.append(dict(pipeline=nome,
                           acc_caractere=round(acuracia_caractere(reais, prev), 4),
                           acc_placa=round(acuracia_placa(reais, prev), 4)))
    res = pd.DataFrame(resumo)
    res.to_csv(f"{TAB}/comparacao_pipelines.csv", index=False)

    print("\n=== 30 placas brasileiras, gabarito humano ===")
    print(res.to_string(index=False))

    # quantas placas produzem exatamente 7 caracteres -- a metrica em que a
    # segmentacao classica trava (ela SEMPRE devolve 7, mas errados)
    print(f"\nYOLO achou exatamente 7 caracteres em "
          f"{(det.n_chars_yolo == 7).sum()} de {len(det)} placas")
    print("distribuicao:", det.n_chars_yolo.value_counts().sort_index().to_dict())
    print("\nErros por posicao:")
    for nome, col in [("A) classico", "classico"), ("B) yolo+cnn", "yolo_cnn"),
                      ("C) yolo puro", "yolo")]:
        print(f"  {nome:14s} {erros_por_posicao(reais, det[col].tolist())}")
    print(f"\nDetalhe: {TAB}/comparacao_pipelines_detalhe.csv")


if __name__ == "__main__":
    main()
