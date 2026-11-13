"""Rota final: o YOLO de caracteres dá a caixa E a classe de cada caractere.

Por que não segmentação clássica: ela acha exatamente 7 blobs em apenas 4 de 14
placas, porque os caracteres se encostam entre si e na moldura depois da
binarização. Um detector aprendido não depende de binarização nem de os
caracteres estarem separados (ver docs/adr/0001-rota-unica-yolo.md).

Uso:
    from src.leitor_yolo import LeitorDePlacasYOLO

    leitor = LeitorDePlacasYOLO("modelos/detector_best.pt",
                                "modelos/chars_best.pt")
    print(leitor.ler("foto.jpg"))
"""

from typing import Union

import cv2
import numpy as np

from .preprocessamento import detectar_layout, recortar
from .validacao import CLASSES, LIMIAR_CONFIANCA, aplicar_mascara
N_CARACTERES = 7

# O dataset de caracteres tem 38 classes: as 36 do alfabeto de placa mais
# `EUR` (a tarja azul das placas europeias) e `-` (o hifen do formato antigo).
# As duas não são caractere de placa e precisam sair antes de ordenar por X.
NAO_CARACTERES = {"EUR", "-"}

# Limiar de confianca do detector de caracteres. Bem mais baixo que o 0,25
# habitual DE PROPOSITO: como a placa tem exatamente 7 caracteres e ficamos com
# os 7 de maior confianca, falso positivo e barato -- o caro e faltar caractere,
# que estraga a leitura inteira e ainda invalida a regra de formato.
# Varredura nas 30 placas com gabarito (Dia 6):
#   conf   placas com 7 chars   acc_caractere   acc_placa
#   0.05        29/30              0.7190        0.3333
#   0.10        27/30              0.6952        0.3000
#   0.25        23/30              0.6571        0.2333
#   0.50         8/30              0.4810        0.1667
CONF_CARACTERE = 0.05


class LeitorDePlacasYOLO:
    def __init__(self, caminho_detector: str, caminho_chars: str,
                 limiar_confianca: float = LIMIAR_CONFIANCA):
        self.caminho_detector = caminho_detector
        self.caminho_chars = caminho_chars
        self.limiar_confianca = limiar_confianca
        self._detector = self._chars = None

    # ---------- carregamento preguiçoso ----------

    @property
    def detector(self):
        if self._detector is None:
            from ultralytics import YOLO
            self._detector = YOLO(self.caminho_detector)
        return self._detector

    @property
    def chars(self):
        if self._chars is None:
            from ultralytics import YOLO
            self._chars = YOLO(self.caminho_chars)
        return self._chars

    # ---------- etapas ----------

    def _detectar_caracteres(self, placa_bgr: np.ndarray, conf: float):
        """Roda o YOLO de caracteres e devolve as caixas válidas ordenadas por X.

        Descarta EUR e `-`; se sobrar mais que 7, fica com as 7 de maior
        confiança (e reordena por X depois, senão a leitura sai embaralhada).
        """
        r = self.chars.predict(placa_bgr, conf=conf, verbose=False)[0]
        nomes = r.names
        achados = []
        for caixa, cls, cf in zip(r.boxes.xyxy.cpu().numpy(),
                                  r.boxes.cls.cpu().numpy().astype(int),
                                  r.boxes.conf.cpu().numpy()):
            classe = str(nomes[int(cls)]).upper()
            if classe in NAO_CARACTERES or classe not in CLASSES:
                continue
            achados.append((caixa, classe, float(cf)))

        if len(achados) > N_CARACTERES:
            achados.sort(key=lambda t: -t[2])
            achados = achados[:N_CARACTERES]
        achados.sort(key=lambda t: t[0][0])          # da esquerda pra direita
        return achados

    # ---------- pipeline ----------

    def ler(self, entrada: Union[str, np.ndarray], conf: float = 0.25,
            conf_char: float = CONF_CARACTERE,
            usar_mascara: bool = True) -> dict:
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

        # 2) recortar (sem endireitar nem binarizar -- o detector de caracteres
        #    lida com inclinação sozinho, que é justamente a vantagem dele)
        placa = recortar(img, (x1, y1, x2, y2))
        if placa.size == 0:
            return {"status": "recorte_invalido"}

        # 3) detectar os caracteres dentro da placa
        achados = self._detectar_caracteres(placa, conf_char)
        layout, score_azul = detectar_layout(placa)

        if not achados:
            return {"status": "sem_caracteres", "layout": layout,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]}

        # 4) a classe e a confiança vêm do próprio detector de caracteres
        letras = [c[1] for c in achados]
        confs = [c[2] for c in achados]

        bruto = "".join(letras)
        # 5) regra do formato -- só faz sentido com os 7 caracteres presentes
        texto = (aplicar_mascara(bruto, layout)
                 if usar_mascara and len(bruto) == N_CARACTERES else bruto)
        conf_minima = float(min(confs)) if confs else 0.0
        completa = len(bruto) == N_CARACTERES

        return {
            "status": ("ok" if completa and conf_minima >= self.limiar_confianca
                       else "revisao_manual"),
            "placa": texto if completa and conf_minima >= self.limiar_confianca else None,
            "placa_sem_regra": bruto,
            "placa_com_regra": texto,
            "n_caracteres": len(bruto),
            "layout": layout,
            "score_azul": score_azul,
            "conf_deteccao": round(float(confiancas[i]), 3),
            "conf_media": round(float(np.mean(confs)), 3) if confs else 0.0,
            "conf_minima": round(conf_minima, 3),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            # caixas dos caracteres em coordenadas do recorte (com a margem
            # de 8%), da esquerda pra direita -- usadas para desenhar a galeria
            "caracteres": [{"classe": c[1], "conf": round(c[2], 3),
                            "caixa": [round(float(v), 1) for v in c[0]]}
                           for c in achados],
        }
