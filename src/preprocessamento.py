"""Operações sobre o recorte da placa usadas pela rota final."""

from typing import Tuple

import cv2
import numpy as np

TAMANHO_PLACA = (200, 62)      # largura, altura depois de retificar


def recortar(img: np.ndarray, caixa, margem: float = 0.08) -> np.ndarray:
    """Recorta a região da caixa (x1, y1, x2, y2) com uma folga percentual."""
    x1, y1, x2, y2 = [int(v) for v in caixa]
    mx, my = int((x2 - x1) * margem), int((y2 - y1) * margem)
    y_ini, y_fim = max(0, y1 - my), min(img.shape[0], y2 + my)
    x_ini, x_fim = max(0, x1 - mx), min(img.shape[1], x2 + mx)
    return img[y_ini:y_fim, x_ini:x_fim]


def retificar_por_cantos(img: np.ndarray, cantos) -> np.ndarray:
    """Correção de perspectiva quando os 4 cantos da placa são conhecidos.

    `cantos` na ordem: superior-esquerdo, superior-direito,
    inferior-direito, inferior-esquerdo.
    """
    largura, altura = TAMANHO_PLACA
    origem = np.float32(cantos)
    destino = np.float32([[0, 0], [largura - 1, 0],
                          [largura - 1, altura - 1], [0, altura - 1]])
    matriz = cv2.getPerspectiveTransform(origem, destino)
    return cv2.warpPerspective(img, matriz, (largura, altura))


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
