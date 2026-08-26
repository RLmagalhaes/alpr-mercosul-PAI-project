"""Tratamento da imagem da placa, do recorte bruto às 7 fatias de caractere.

Fluxo: recortar -> endireitar -> redimensionar -> cinza -> CLAHE -> Otsu
       -> remover a tarja azul -> dividir em 7 fatias de 32x32.
"""

from typing import List, Tuple

import cv2
import numpy as np

TAMANHO_PLACA = (200, 62)      # largura, altura após padronizar
TAMANHO_CARACTERE = (32, 32)
CORTE_SUPERIOR = 0.35          # fração de cima descartada (tarja "BRASIL")
N_CARACTERES = 7


def recortar(img: np.ndarray, caixa, margem: float = 0.08) -> np.ndarray:
    """Recorta a região da caixa (x1, y1, x2, y2) com uma folga percentual."""
    x1, y1, x2, y2 = [int(v) for v in caixa]
    mx, my = int((x2 - x1) * margem), int((y2 - y1) * margem)
    y_ini, y_fim = max(0, y1 - my), min(img.shape[0], y2 + my)
    x_ini, x_fim = max(0, x1 - mx), min(img.shape[1], x2 + mx)
    return img[y_ini:y_fim, x_ini:x_fim]


def endireitar(placa_bgr: np.ndarray) -> np.ndarray:
    """Corrige a inclinação da placa usando o retângulo de área mínima.

    Se a medida do ângulo sair absurda (> 20 graus), devolve a imagem original:
    nesses casos a estimativa costuma estar errada e girar só piora.
    """
    cinza = cv2.cvtColor(placa_bgr, cv2.COLOR_BGR2GRAY)
    binaria = cv2.threshold(cinza, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(binaria > 0))
    if len(coords) < 20:
        return placa_bgr

    angulo = cv2.minAreaRect(coords[:, ::-1])[-1]
    if angulo > 45:
        angulo -= 90
    if abs(angulo) > 20:
        return placa_bgr

    altura, largura = placa_bgr.shape[:2]
    matriz = cv2.getRotationMatrix2D((largura / 2, altura / 2), angulo, 1.0)
    return cv2.warpAffine(placa_bgr, matriz, (largura, altura),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def retificar_por_cantos(img: np.ndarray, cantos) -> np.ndarray:
    """Correção de perspectiva quando os 4 cantos da placa são conhecidos.

    `cantos` na ordem: superior-esquerdo, superior-direito,
    inferior-direito, inferior-esquerdo. Use com datasets como o RodoSol-ALPR,
    que anotam os cantos.
    """
    largura, altura = TAMANHO_PLACA
    origem = np.float32(cantos)
    destino = np.float32([[0, 0], [largura - 1, 0],
                          [largura - 1, altura - 1], [0, altura - 1]])
    matriz = cv2.getPerspectiveTransform(origem, destino)
    return cv2.warpPerspective(img, matriz, (largura, altura))


def preparar(placa_bgr: np.ndarray, tamanho: Tuple[int, int] = TAMANHO_PLACA):
    """Padroniza o tamanho, converte para cinza, aplica CLAHE e binariza.

    CLAHE equaliza o contraste em blocos pequenos, então corrige uma placa
    parcialmente na sombra sem estourar a parte iluminada — o que a
    equalização global de histograma não consegue fazer.

    Devolve: (colorida, cinza, realçada, binária)
    """
    colorida = cv2.resize(placa_bgr, tamanho, interpolation=cv2.INTER_CUBIC)
    cinza = cv2.cvtColor(colorida, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    realcada = clahe.apply(cinza)
    binaria = cv2.threshold(realcada, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return colorida, cinza, realcada, binaria


def detectar_layout(placa_bgr: np.ndarray) -> Tuple[str, float]:
    """Placa Mercosul tem tarja azul no topo; a antiga, não.

    Devolve o layout e a fração de pixels azuis encontrada no topo, útil para
    calibrar o limiar caso o dataset tenha iluminação atípica.
    """
    altura_topo = max(1, int(placa_bgr.shape[0] * 0.32))
    hsv = cv2.cvtColor(placa_bgr[:altura_topo, :], cv2.COLOR_BGR2HSV)
    azul = float(((hsv[:, :, 0] >= 100) & (hsv[:, :, 0] <= 130) &
                  (hsv[:, :, 1] >= 80)).mean())
    return ("mercosul" if azul > 0.25 else "antiga"), round(azul, 3)


def projecao_vertical(binaria: np.ndarray,
                      corte_superior: float = CORTE_SUPERIOR):
    """Soma os pixels brancos de cada coluna da faixa dos caracteres.

    Os vales dessa curva são os espaços entre caracteres. Serve para
    visualizar a segmentação e para refiná-la, se necessário.
    """
    faixa = binaria[int(binaria.shape[0] * corte_superior):, :]
    return faixa, faixa.sum(axis=0) / 255.0


def segmentar(binaria: np.ndarray,
              n: int = N_CARACTERES,
              corte_superior: float = CORTE_SUPERIOR,
              saida: Tuple[int, int] = TAMANHO_CARACTERE) -> List[np.ndarray]:
    """Divide a faixa dos caracteres em n fatias iguais e padroniza o tamanho.

    A divisão é aproximada de propósito: a placa tem espaçamento regular, e o
    pequeno desalinhamento funciona como variação natural que a CNN aprende a
    tolerar (ainda mais com augmentation de translação no treino).

    Se as fatias saírem tortas, ajuste `corte_superior` entre 0.30 e 0.40.
    """
    faixa, _ = projecao_vertical(binaria, corte_superior)
    largura_fatia = faixa.shape[1] // n
    return [
        cv2.resize(faixa[:, i * largura_fatia:(i + 1) * largura_fatia],
                   saida, interpolation=cv2.INTER_AREA)
        for i in range(n)
    ]
