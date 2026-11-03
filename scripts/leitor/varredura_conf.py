"""Mede o efeito do limiar `conf` do detector de caracteres sobre quantas placas
saem completas (7 chars) e sobre a acuracia. Justifica CONF_CARACTERE = 0,05.

Roda LOCAL (.venv), sem Colab.

Uso:
    .venv/bin/python scripts/leitor/varredura_conf.py
"""

import os
import sys

import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RAIZ)

from src.metricas import acuracia_caractere, acuracia_placa  # noqa: E402

BASE = f"{RAIZ}/dados/deteccao/vehicle-plates"
TAB = f"{RAIZ}/resultados/tabelas"
DETECTOR = f"{RAIZ}/modelos/detector_best.pt"
CHARS = f"{RAIZ}/modelos/chars_best.pt"

CONFS = [0.05, 0.10, 0.25, 0.50]
N_CARACTERES = 7


def main():
    gab = pd.read_csv(f"{TAB}/gabarito_mercosul.csv", dtype=str).fillna("")
    gab["texto"] = gab["texto"].str.strip().str.upper()
    gab = gab[gab["texto"].str.len() == N_CARACTERES]
    print(f"Placas com gabarito: {len(gab)}")

    from src.leitor_yolo import LeitorDePlacasYOLO
    leitor = LeitorDePlacasYOLO(DETECTOR, CHARS)

    linhas = []
    for conf in CONFS:
        reais, previstos, completas = [], [], 0
        for _, r in gab.iterrows():
            caminho = f"{BASE}/{r['split']}/images/{r['arquivo']}"
            res = leitor.ler(caminho, conf_char=conf)
            bruto = res.get("placa_sem_regra", "") or ""
            if len(bruto) == N_CARACTERES:
                completas += 1
            reais.append(r["texto"])
            previstos.append(res.get("placa_com_regra", "") or "")
        linhas.append(dict(
            conf=conf,
            placas_com_7_chars=completas,
            n_placas=len(gab),
            acc_caractere=round(acuracia_caractere(reais, previstos), 4),
            acc_placa=round(acuracia_placa(reais, previstos), 4)))
        print(f"  conf={conf:.2f}  completas={completas}/{len(gab)}  "
              f"acc_char={linhas[-1]['acc_caractere']:.4f}  "
              f"acc_placa={linhas[-1]['acc_placa']:.4f}")

    pd.DataFrame(linhas).to_csv(
        f"{TAB}/varredura_conf_caracteres.csv", index=False)
    print(f"-> {TAB}/varredura_conf_caracteres.csv")


if __name__ == "__main__":
    main()
