"""Pipeline completo: foto do veículo -> texto da placa.

Os modelos pesados (YOLO e Keras) são carregados sob demanda, então importar
este módulo é barato e não exige ter tudo instalado.

Uso:
    from src.pipeline import LeitorDePlacas

    leitor = LeitorDePlacas("modelos/detector_best.pt", "modelos/cnn_chars.keras")
    print(leitor.ler("foto.jpg"))
"""

from typing import Optional, Union

import cv2
import numpy as np

from .preprocessamento import (detectar_layout, endireitar, preparar,
                               recortar, segmentar)
from .validacao import CLASSES, aplicar_mascara

LIMIAR_CONFIANCA = 0.70     # abaixo disto o sistema pede revisão humana


class LeitorDePlacas:
    def __init__(self, caminho_detector: str, caminho_cnn: str,
                 classes=None, limiar_confianca: float = LIMIAR_CONFIANCA):
        self.caminho_detector = caminho_detector
        self.caminho_cnn = caminho_cnn
        self.classes = classes or CLASSES
        self.limiar_confianca = limiar_confianca
        self._detector = None
        self._cnn = None

    # ---------- carregamento preguiçoso ----------

    @property
    def detector(self):
        if self._detector is None:
            from ultralytics import YOLO
            self._detector = YOLO(self.caminho_detector)
        return self._detector

    @property
    def cnn(self):
        if self._cnn is None:
            from tensorflow import keras
            self._cnn = keras.models.load_model(self.caminho_cnn)
        return self._cnn

    # ---------- pipeline ----------

    def ler(self, entrada: Union[str, np.ndarray],
            conf: float = 0.25, usar_mascara: bool = True) -> dict:
        """Recebe um caminho de arquivo ou uma imagem BGR e devolve a leitura."""
        img = cv2.imread(entrada) if isinstance(entrada, str) else entrada
        if img is None:
            return {"status": "erro", "motivo": "imagem não pôde ser lida"}

        # 1) detectar a placa e ficar com a de maior confiança
        caixas = self.detector.predict(img, conf=conf, verbose=False)[0].boxes
        if len(caixas) == 0:
            return {"status": "sem_placa"}

        confiancas = caixas.conf.cpu().numpy()
        i = int(np.argmax(confiancas))
        x1, y1, x2, y2 = caixas.xyxy.cpu().numpy()[i].astype(int)

        # 2) recortar com folga e corrigir a inclinação
        placa = recortar(img, (x1, y1, x2, y2))
        if placa.size == 0:
            return {"status": "recorte_invalido"}
        placa = endireitar(placa)

        # 3) padronizar, realçar e binarizar
        colorida, _, _, binaria = preparar(placa)
        layout, score_azul = detectar_layout(colorida)

        # 4) segmentar e classificar os 7 caracteres de uma vez
        fatias = segmentar(binaria)
        lote = np.stack(fatias).astype("float32")[..., None]
        probabilidades = self.cnn.predict(lote, verbose=0)
        indices = probabilidades.argmax(axis=1)
        confs_char = probabilidades.max(axis=1)

        bruto = "".join(self.classes[k] for k in indices)

        # 5) aplicar a regra do formato
        texto = aplicar_mascara(bruto, layout) if usar_mascara else bruto
        conf_minima = float(confs_char.min())

        return {
            "status": "ok" if conf_minima >= self.limiar_confianca else "revisao_manual",
            "placa": texto if conf_minima >= self.limiar_confianca else None,
            "placa_sem_regra": bruto,
            "placa_com_regra": texto,
            "layout": layout,
            "score_azul": score_azul,
            "conf_deteccao": round(float(confiancas[i]), 3),
            "conf_media": round(float(confs_char.mean()), 3),
            "conf_minima": round(conf_minima, 3),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
        }
