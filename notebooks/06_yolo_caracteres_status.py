"""Dia 6 - Mostra o andamento do treino do YOLO de caracteres.

Uso (o Raphael roda no terminal dele):
    colab exec -s dia6 -f notebooks/06_yolo_caracteres_status.py --timeout 120
"""

import os

CSV = "/content/modelos/chars/results.csv"
LOG = "/content/log_chars.txt"

if os.path.exists(CSV):
    linhas = open(CSV).read().strip().splitlines()
    cab = [c.strip() for c in linhas[0].split(",")]
    print(f"Epocas concluidas: {len(linhas) - 1}")
    # so as colunas que importam pro relatorio
    quero = ["epoch", "metrics/mAP50(B)", "metrics/mAP50-95(B)",
             "metrics/precision(B)", "metrics/recall(B)"]
    idx = [cab.index(c) for c in quero if c in cab]
    print(" | ".join(cab[i] for i in idx))
    for linha in linhas[-5:]:
        v = [c.strip() for c in linha.split(",")]
        print(" | ".join(v[i] for i in idx))
else:
    print("results.csv ainda nao existe (treino na primeira epoca).")

if os.path.exists(LOG):
    print("\n--- ultimas linhas do log ---")
    print("\n".join(open(LOG).read().strip().splitlines()[-12:]))

pesos = "/content/modelos/chars/weights/best.pt"
if os.path.exists(pesos):
    print(f"\nbest.pt disponivel ({os.path.getsize(pesos)/1e6:.1f} MB)")
    print("Para trazer pro Mac: colab download modelos/chars/weights/best.pt")
