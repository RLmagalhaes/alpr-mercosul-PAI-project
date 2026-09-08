"""Pipeline alternativo: os caracteres são DETECTADOS por um YOLO, em vez de
segmentados por processamento clássico de imagem.

Por que existe (ver DIARIO, Dia 6): `segmentar()` acha exatamente 7 blobs em
apenas 4 de 14 placas (8/14 no melhor caso, com resolução 3x maior). Os
caracteres se encostam uns nos outros e na moldura da placa depois da
binarização, então componente conectado não corresponde a caractere. É teto do
método, não parâmetro mal ajustado. Um detector aprendido não depende de
binarização, de `corte_superior`, nem de os caracteres estarem separados.

Duas variantes, para o relatório poder comparar:

  modo="yolo"  -> o próprio YOLO de caracteres dá a caixa E a classe.
                  Pipeline mais curto, dispensa a CNN do Dia 4.
  modo="cnn"   -> o YOLO dá só as caixas; cada recorte é classificado pela CNN
                  do Dia 4, com o MESMO tratamento do treino (CLAHE + Otsu
                  local + 32x32). Mantém a CNN em uso e isola o efeito de
                  trocar só a segmentação.

Uso:
    from src.leitor_yolo import LeitorDePlacasYOLO

    leitor = LeitorDePlacasYOLO("modelos/detector_best.pt",
                                "modelos/chars_best.pt",
                                "modelos/cnn_chars_compat.keras")
    print(leitor.ler("foto.jpg", modo="cnn"))
"""

from typing import List, Tuple, Union

import cv2
import numpy as np

from .preprocessamento import TAMANHO_CARACTERE, detectar_layout, recortar
from .validacao import CLASSES, aplicar_mascara

LIMIAR_CONFIANCA = 0.70
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
                 caminho_cnn: str = None,
                 limiar_confianca: float = LIMIAR_CONFIANCA):
        self.caminho_detector = caminho_detector
        self.caminho_chars = caminho_chars
        self.caminho_cnn = caminho_cnn
        self.limiar_confianca = limiar_confianca
        self._detector = self._chars = self._cnn = None

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

    @property
    def cnn(self):
        if self._cnn is None:
            from tensorflow import keras
            self._cnn = keras.models.load_model(self.caminho_cnn)
        return self._cnn

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

    def _classificar_com_cnn(self, placa_bgr, caixas) -> Tuple[List[str], List[float]]:
        """Recorta cada caractere e classifica com a CNN do Dia 4, reproduzindo
        exatamente o tratamento do treino (`04_cnn_caracteres.py`): recorte justo
        nos dois eixos, CLAHE, Otsu LOCAL, resize 32x32."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        fatias = []
        for (x1, y1, x2, y2) in caixas:
            rec = placa_bgr[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
            if rec.size == 0:
                fatias.append(np.zeros(TAMANHO_CARACTERE, dtype=np.uint8))
                continue
            cinza = clahe.apply(cv2.cvtColor(rec, cv2.COLOR_BGR2GRAY))
            binaria = cv2.threshold(cinza, 0, 255,
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            fatias.append(cv2.resize(binaria, TAMANHO_CARACTERE,
                                     interpolation=cv2.INTER_AREA))
        lote = np.stack(fatias).astype("float32")[..., None]
        probs = self.cnn.predict(lote, verbose=0)
        return ([CLASSES[k] for k in probs.argmax(axis=1)],
                [float(v) for v in probs.max(axis=1)])

    # ---------- pipeline ----------

    def ler(self, entrada: Union[str, np.ndarray], conf: float = 0.25,
            conf_char: float = CONF_CARACTERE, modo: str = "cnn",
            usar_mascara: bool = True) -> dict:
        """`modo`: "yolo" usa a classe do próprio detector de caracteres;
        "cnn" usa só as caixas dele e classifica com a CNN do Dia 4."""
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

        # 4) classificar
        if modo == "cnn" and self.caminho_cnn:
            letras, confs = self._classificar_com_cnn(
                placa, [c[0] for c in achados])
        else:
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
            "modo": modo,
            "conf_deteccao": round(float(confiancas[i]), 3),
            "conf_media": round(float(np.mean(confs)), 3) if confs else 0.0,
            "conf_minima": round(conf_minima, 3),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
        }
