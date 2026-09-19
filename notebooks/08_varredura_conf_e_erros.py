"""Dia 6 - Gera em CSV dois numeros que ate agora so existiam como texto no
DIARIO, pra que o relatorio possa cita-los com rastreabilidade:

  1) varredura_conf_caracteres.csv  - efeito do limiar `conf` do detector de
     caracteres sobre quantas placas saem completas (7 chars) e sobre a
     acuracia. Justifica a escolha de CONF_CARACTERE = 0,05.

  2) erros_por_posicao_pipelines.csv - quantos erros cada uma das 7 posicoes
     acumula, nos tres pipelines, com e sem a regra de formato.

Roda LOCAL (.venv), sem Colab.

Uso:
    .venv/bin/python notebooks/08_varredura_conf_e_erros.py
"""

import os
import sys

import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from src.metricas import acuracia_caractere, acuracia_placa  # noqa: E402

BASE = f"{RAIZ}/dados/deteccao/vehicle-plates"
TAB = f"{RAIZ}/resultados/tabelas"
DETECTOR = f"{RAIZ}/modelos/detector_best.pt"
CHARS = f"{RAIZ}/modelos/chars_best.pt"
CNN = (f"{RAIZ}/modelos/cnn_chars_compat.keras"
       if os.path.exists(f"{RAIZ}/modelos/cnn_chars_compat.keras")
       else f"{RAIZ}/modelos/cnn_chars.keras")

CONFS = [0.05, 0.10, 0.25, 0.50]
N_CARACTERES = 7


def erros_por_posicao(reais, previstos, n=N_CARACTERES):
    """Conta, pra cada uma das n posicoes, em quantas placas ela saiu errada.
    Predicao curta demais conta como erro em todas as posicoes que faltam."""
    erros = [0] * n
    for real, prev in zip(reais, previstos):
        for i in range(n):
            if i >= len(prev) or prev[i] != real[i]:
                erros[i] += 1
    return erros


def main():
    gab = pd.read_csv(f"{TAB}/gabarito_mercosul.csv", dtype=str).fillna("")
    gab["texto"] = gab["texto"].str.strip().str.upper()
    gab = gab[gab["texto"].str.len() == N_CARACTERES]
    print(f"Placas com gabarito: {len(gab)}")

    # ---------- 1) varredura do limiar de confianca ----------
    from src.leitor_yolo import LeitorDePlacasYOLO
    leitor = LeitorDePlacasYOLO(DETECTOR, CHARS, CNN)

    linhas = []
    for conf in CONFS:
        reais, previstos, completas = [], [], 0
        for _, r in gab.iterrows():
            caminho = f"{BASE}/{r['split']}/images/{r['arquivo']}"
            res = leitor.ler(caminho, conf_char=conf, modo="yolo")
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

    # ---------- 2) erros por posicao, nos 3 pipelines ----------
    det = pd.read_csv(f"{TAB}/comparacao_pipelines_detalhe.csv", dtype=str).fillna("")
    reais = det["real"].tolist()

    colunas = [
        ("A) classico (segmentar)", "sem regra", "classico_bruto"),
        ("A) classico (segmentar)", "com regra", "classico"),
        ("B) YOLO caixas + CNN", "sem regra", "yolo_cnn_bruto"),
        ("B) YOLO caixas + CNN", "com regra", "yolo_cnn"),
        ("C) YOLO caixa+classe", "sem regra", "yolo_bruto"),
        ("C) YOLO caixa+classe", "com regra", "yolo"),
    ]
    saida = []
    for nome, regra, col in colunas:
        e = erros_por_posicao(reais, det[col].tolist())
        saida.append(dict(pipeline=nome, regra=regra,
                          **{f"pos{i+1}": e[i] for i in range(N_CARACTERES)},
                          total_erros=sum(e),
                          acc_caractere=round(1 - sum(e) / (len(reais) * N_CARACTERES), 4)))
    df = pd.DataFrame(saida)
    df.to_csv(f"{TAB}/erros_por_posicao_pipelines.csv", index=False)
    print(f"\n-> {TAB}/erros_por_posicao_pipelines.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
