"""Avalia a rota final sobre as 30 placas brasileiras com gabarito humano
(resultados/tabelas/gabarito_mercosul.csv).

Roda LOCAL (.venv), sem Colab.

Uso:
    .venv/bin/python scripts/leitor/avaliar.py
"""

import os
import sys

import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RAIZ)

from src.metricas import acuracia_caractere, acuracia_placa, erros_por_posicao  # noqa: E402

BASE = f"{RAIZ}/dados/deteccao/vehicle-plates"
TAB = f"{RAIZ}/resultados/tabelas"
DETECTOR = f"{RAIZ}/modelos/detector_best.pt"
CHARS = f"{RAIZ}/modelos/chars_best.pt"


def main():
    gab = pd.read_csv(f"{TAB}/gabarito_mercosul.csv", dtype=str).fillna("")
    gab["texto"] = gab["texto"].str.strip().str.upper()
    gab = gab[gab["texto"].str.len() == 7]
    print(f"Placas com gabarito: {len(gab)}")

    from src.leitor_yolo import LeitorDePlacasYOLO

    leitor = LeitorDePlacasYOLO(DETECTOR, CHARS)

    linhas = []
    for _, r in gab.iterrows():
        res = leitor.ler(f"{BASE}/{r['split']}/images/{r['arquivo']}")
        linhas.append(dict(
            n=r["n"], real=r["texto"],
            sem_regra=res.get("placa_sem_regra", "") or "",
            com_regra=res.get("placa_com_regra", "") or "",
            n_caracteres=res.get("n_caracteres", 0),
            layout=res.get("layout", ""),
            conf_minima=res.get("conf_minima", 0.0)))
    det = pd.DataFrame(linhas)
    det.to_csv(f"{TAB}/leitor_final_detalhe.csv", index=False)

    reais = det["real"].tolist()
    resumo, erros = [], []
    for regra, col in [("sem regra", "sem_regra"), ("com regra", "com_regra")]:
        prev = det[col].tolist()
        resumo.append(dict(regra=regra,
                           acc_caractere=round(acuracia_caractere(reais, prev), 4),
                           acc_placa=round(acuracia_placa(reais, prev), 4)))
        e = erros_por_posicao(reais, prev)
        erros.append(dict(regra=regra, **{f"pos{i+1}": v for i, v in enumerate(e)}))
    res = pd.DataFrame(resumo)
    res.to_csv(f"{TAB}/leitor_final.csv", index=False)
    pd.DataFrame(erros).to_csv(f"{TAB}/erros_por_posicao.csv", index=False)

    print("\n=== 30 placas brasileiras, gabarito humano ===")
    print(res.to_string(index=False))
    print(f"\nYOLO achou exatamente 7 caracteres em "
          f"{(det.n_caracteres == 7).sum()} de {len(det)} placas")
    print("Erros por posicao, com regra:", erros[1])
    print(f"\nDetalhe: {TAB}/leitor_final_detalhe.csv")


if __name__ == "__main__":
    main()
