"""Roda o leitor nas placas escolhidas para a galeria da página e salva as imagens.

Para cada placa da SELECAO, grava em `site/img/galeria/`:
  - <n>_foto.jpg     a foto com a caixa da placa desenhada
  - <n>_recorte.png  o recorte (com a margem de 8%) e as caixas dos caracteres
e grava `resultados/tabelas/galeria.csv` com o que o leitor leu em cada uma.

Precisa dos modelos e das imagens, então roda LOCAL (.venv). Depois dele, rode
`scripts/site/gerar.py`, que monta a página só a partir dos CSVs.

Uso:
    .venv/bin/python scripts/site/preparar_galeria.py
    .venv/bin/python scripts/site/preparar_galeria.py --raiz-dados /outro/checkout
"""

import argparse
import os
import sys

import cv2
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RAIZ)

from src.preprocessamento import recortar  # noqa: E402

TAB = f"{RAIZ}/resultados/tabelas"
SAIDA_IMG = f"{RAIZ}/site/img/galeria"

# Escolha editorial: 6 acertos (3 Mercosul, 3 no formato antigo) e 2 erros.
# Os números da página saem das 30 placas inteiras; a galeria só ilustra.
#   n -> por que entrou
SELECAO = {
    2:  "Mercosul de frente, a leitura mais limpa do conjunto.",
    7:  "Mercosul em ângulo; uma das placas que o sistema libera sem revisão.",
    3:  "Carro inteiro, placa pequena; a regra resolve O e 0.",
    20: "Formato antigo, de perto; também liberada sem revisão.",
    24: "Formato antigo; a regra corrige um D lido no lugar de 0.",
    28: "Formato antigo; a regra corrige um I lido no lugar de 1.",
    4:  "Erro típico: um único caractere trocado, 9 lido como 5.",
    22: "Placa escura e borrada: o sistema acha só um caractere e pede revisão.",
}

AZUL_BGR = (153, 51, 0)          # #003399, o azul da faixa Mercosul
LARGURA_FOTO = 832               # 2x os 416 px originais, para ficar nítido na tela
ALTURA_RECORTE = 180


def aparar_faixas_pretas(img, bbox):
    """Tira as faixas pretas que o Roboflow pôs para deixar a foto quadrada.

    Devolve a foto aparada e a caixa da placa nas novas coordenadas.
    """
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    linhas = (cinza.max(axis=1) > 12).nonzero()[0]
    colunas = (cinza.max(axis=0) > 12).nonzero()[0]
    y0, y1 = linhas[0], linhas[-1] + 1
    x0, x1 = colunas[0], colunas[-1] + 1
    a, b, c, d = bbox
    return img[y0:y1, x0:x1], (a - x0, b - y0, c - x0, d - y0)


def desenhar_foto(img, bbox):
    """Apara, amplia a foto e desenha a caixa da placa."""
    img, bbox = aparar_faixas_pretas(img, bbox)
    escala = LARGURA_FOTO / img.shape[1]
    grande = cv2.resize(img, None, fx=escala, fy=escala,
                        interpolation=cv2.INTER_CUBIC)
    x1, y1, x2, y2 = [int(round(v * escala)) for v in bbox]
    cv2.rectangle(grande, (x1, y1), (x2, y2), (255, 255, 255), 7)   # contorno claro
    cv2.rectangle(grande, (x1, y1), (x2, y2), AZUL_BGR, 4)
    return grande


def desenhar_recorte(img, bbox, caracteres):
    """Recorta como o leitor recorta, amplia e desenha a caixa de cada caractere."""
    placa = recortar(img, bbox)
    escala = ALTURA_RECORTE / placa.shape[0]
    grande = cv2.resize(placa, None, fx=escala, fy=escala,
                        interpolation=cv2.INTER_CUBIC)
    for c in caracteres:
        x1, y1, x2, y2 = [int(round(v * escala)) for v in c["caixa"]]
        cv2.rectangle(grande, (x1, y1), (x2, y2), (255, 255, 255), 4)
        cv2.rectangle(grande, (x1, y1), (x2, y2), AZUL_BGR, 2)
    return grande


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raiz-dados", default=RAIZ,
                        help="pasta que contém dados/ e modelos/ (padrão: este checkout)")
    args = parser.parse_args()
    base = f"{args.raiz_dados}/dados/deteccao/vehicle-plates"

    from src.leitor_yolo import LeitorDePlacasYOLO
    leitor = LeitorDePlacasYOLO(f"{args.raiz_dados}/modelos/detector_best.pt",
                                f"{args.raiz_dados}/modelos/chars_best.pt")

    gab = pd.read_csv(f"{TAB}/gabarito_mercosul.csv", dtype=str).fillna("")
    gab["n"] = gab["n"].astype(int)
    detalhe = pd.read_csv(f"{TAB}/leitor_final_detalhe.csv", dtype=str).fillna("")
    detalhe["n"] = detalhe["n"].astype(int)
    os.makedirs(SAIDA_IMG, exist_ok=True)

    linhas = []
    for n, motivo in SELECAO.items():
        r = gab[gab["n"] == n].iloc[0]
        img = cv2.imread(f"{base}/{r['split']}/images/{r['arquivo']}")
        res = leitor.ler(img)

        # Conferência: a leitura de agora tem de bater com a avaliação publicada.
        esperado = detalhe[detalhe["n"] == n].iloc[0]["com_regra"]
        if res.get("placa_com_regra", "") != esperado:
            sys.exit(f"Placa {n}: leu {res.get('placa_com_regra')!r}, "
                     f"a avaliação publicada diz {esperado!r}. Rode "
                     f"scripts/leitor/avaliar.py antes de gerar a galeria.")

        cv2.imwrite(f"{SAIDA_IMG}/{n}_foto.jpg", desenhar_foto(img, res["bbox"]),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        cv2.imwrite(f"{SAIDA_IMG}/{n}_recorte.png",
                    desenhar_recorte(img, res["bbox"], res["caracteres"]))

        linhas.append(dict(
            n=n, real=r["texto"].strip().upper(),
            sem_regra=res["placa_sem_regra"], com_regra=res["placa_com_regra"],
            n_caracteres=res["n_caracteres"], layout=res["layout"],
            conf_minima=res["conf_minima"], status=res["status"],
            motivo=motivo))
        print(f"{n:>3}  {r['texto']}  ->  {res['placa_com_regra']:<8} "
              f"{res['status']}")

    pd.DataFrame(linhas).to_csv(f"{TAB}/galeria.csv", index=False)
    print(f"\n-> {TAB}/galeria.csv\n-> {SAIDA_IMG}/")


if __name__ == "__main__":
    main()
