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

# Quanto cortar do topo depende do layout: a Mercosul tem a tarja azul
# "BRASIL", a antiga tem a faixa cinza "CIDADE - UF", e as duas ocupam
# frações diferentes da altura. Usar 0.35 pros dois casos (o que o
# pipeline fazia até o Dia 5) cortava o topo dos caracteres da placa
# antiga -- ver DIARIO, Dia 6.
CORTE_POR_LAYOUT = {"mercosul": 0.35, "antiga": 0.30}

# Diferença mínima entre o pixel mais claro e o mais escuro pra confiar no
# Otsu de uma fatia. Num recorte justo de caractere sólido e estreito ("1",
# "I") quase não sobra fundo dentro da caixa, e Otsu sem contraste devolve
# a fatia inteira preta ou inteira branca.
CONTRASTE_MINIMO = 25


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


def _componentes_de_caracteres(faixa: np.ndarray, altura_min_frac: float = 0.5,
                              largura_min_frac: float = 0.02,
                              altura_max_frac: float = 1.3):
    """Acha os blobs de tinta na faixa binária via componentes conectados.

    Devolve uma lista de caixas (x, y, w, h) ordenada da esquerda pra
    direita -- uma por caractere, na teoria. Três filtros, nessa ordem:

    1. Altura mínima (fração da maior altura bruta) -- descarta ruído
       pequeno (poeira, parafuso, sujeira).
    2. Largura mínima (fração da largura da faixa) -- a MOLDURA da placa
       é tão alta quanto um caractere (passaria o filtro 1), mas é bem
       mais fina.
    3. Altura MÁXIMA (fração da altura mediana dos sobreviventes) -- pega
       o caso oposto: a moldura às vezes é mais alta que os caracteres
       de verdade (atravessa a faixa inteira, de cima a baixo, enquanto
       um caractere real tem uma margem). Sem esse terceiro filtro, a
       moldura nas duas bordas sobrevivia e virava "mais dois
       caracteres" (ver DIARIO, Dia 5).
    """
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(faixa, connectivity=8)
    caixas = [tuple(int(v) for v in stats[i, :4]) for i in range(1, n_labels)]
    if not caixas:
        return []
    altura_max_bruta = max(h for _, _, _, h in caixas)
    largura_min = faixa.shape[1] * largura_min_frac
    caixas = [c for c in caixas
             if c[3] >= altura_max_bruta * altura_min_frac and c[2] >= largura_min]

    if len(caixas) > 1:
        mediana = sorted(c[3] for c in caixas)[len(caixas) // 2]
        caixas = [c for c in caixas if c[3] <= mediana * altura_max_frac]

    caixas.sort(key=lambda c: c[0])
    return caixas


def _apertar_vertical(faixa: np.ndarray, x: int, w: int) -> Tuple[int, int]:
    """Remede (y, altura) pela tinta que existe dentro da coluna [x, x+w).

    A CNN foi treinada com cada caractere recortado justo nos DOIS eixos
    (ver `notebooks/04_cnn_caracteres.py`, que corta `img[y1:y2, x1:x2]`
    pela caixa anotada). Até o Dia 5 a inferência recortava justo só em X
    e mantinha a altura inteira da faixa, então o caractere chegava à rede
    achatado e deslocado dentro do quadro de 32x32 -- diferente de tudo
    que ela viu no treino. Medir aqui, e não confiar no (y, h) que veio de
    `_componentes_de_caracteres`, resolve os três casos de uma vez: caixa
    de componente conectado, caixa dividida no vale (que herda o y/h do
    pai, largo demais) e caixa do fallback de fatias iguais.
    """
    coluna = faixa[:, x:x + w]
    if coluna.size == 0:
        return 0, faixa.shape[0]
    linhas = np.where(coluna.any(axis=1))[0]
    if len(linhas) == 0:                    # coluna sem tinta nenhuma
        return 0, faixa.shape[0]
    return int(linhas[0]), int(linhas[-1] - linhas[0] + 1)


def _dividir_no_vale(perfil: np.ndarray, x: int, w: int) -> int:
    """Acha o corte mais provável dentro de [x, x+w) para separar dois
    caracteres grudados: o ponto de menor tinta, evitando as bordas (pra
    não cortar dentro do próprio caractere, só entre dois)."""
    borda = max(1, w // 6)
    ini, fim = x + borda, x + w - borda
    if fim <= ini:
        return x + w // 2
    return ini + int(np.argmin(perfil[ini:fim]))


def _juntar_ou_dividir(caixas, n: int, perfil: np.ndarray, altura: int = 0):
    """Ajusta a lista de caixas pra ter exatamente n.

    Sobrou caixa (caracteres fragmentados em 2+ componentes, ex. traço
    solto)? Junta as duas mais próximas horizontalmente, repetindo até
    sobrar `n`. Faltou caixa (dois caracteres grudados viraram 1
    componente só)? Divide a mais larga no vale de menor tinta (não no
    meio cego), repetindo até completar `n`.
    """
    caixas = list(caixas)
    while len(caixas) > n and len(caixas) > 1:
        gaps = [caixas[i + 1][0] - (caixas[i][0] + caixas[i][2]) for i in range(len(caixas) - 1)]
        i = int(np.argmin(gaps))
        x1, y1, w1, h1 = caixas[i]
        x2, y2, w2, h2 = caixas[i + 1]
        x, y = min(x1, x2), min(y1, y2)
        caixas[i:i + 2] = [(x, y, max(x1 + w1, x2 + w2) - x, max(y1 + h1, y2 + h2) - y)]

    while len(caixas) < n:
        i = int(np.argmax([c[2] for c in caixas])) if caixas else 0
        if not caixas:
            # `altura` é a altura da faixa; sem ela, cai na largura do
            # perfil (era o que estava aqui antes -- inofensivo enquanto o
            # `h` era ignorado no recorte, errado depois que passou a valer)
            caixas = [(0, 0, len(perfil), altura or len(perfil))]
        x, y, w, h = caixas[i]
        corte = _dividir_no_vale(perfil, x, w)
        corte = min(max(corte, x + 1), x + w - 1)
        caixas[i:i + 1] = [(x, y, corte - x, h), (corte, y, x + w - corte, h)]

    return sorted(caixas, key=lambda c: c[0])


def segmentar(binaria: np.ndarray,
              n: int = N_CARACTERES,
              corte_superior: float = CORTE_SUPERIOR,
              saida: Tuple[int, int] = TAMANHO_CARACTERE,
              cinza: np.ndarray = None,
              margem_vertical: float = 0.0) -> List[np.ndarray]:
    """Divide a faixa dos caracteres em n recortes e padroniza o tamanho.

    Usa componentes conectados (blobs de tinta) pra achar cada caractere
    pelo seu próprio contorno, em vez de assumir largura igual pra todos --
    placas reais têm caracteres de largura bem diferente ("W" bem mais
    largo que "I" ou "1"), e um "M" de largura fixa desalinhava tudo a
    partir do primeiro caractere fora do padrão (ver DIARIO, Dia 5: a
    tentativa anterior, dividir em fatias de largura igual -- com ou sem
    tratamento especial do vão letra/dígito -- não resolvia esse
    desalinhamento). Se os blobs não baterem exatamente com `n` (caractere
    partido em 2 componentes, ou dois grudados em 1), `_juntar_ou_dividir`
    ajusta. Faixa sem nenhum componente (placa ilegível) cai de volta pra
    divisão em partes iguais, só pra não quebrar o pipeline.

    Com `cinza` (a imagem em tons de cinza/CLAHE, ANTES da binarização --
    `realcada`, devolvida por `preparar()`), cada fatia é binarizada com
    Otsu LOCAL, uma por vez, em vez de reaproveitar pedaços da imagem já
    binarizada de uma vez só pra placa inteira. Isso importa muito: é
    exatamente assim que os dados de treino da CNN (Dia 3/4) foram
    gerados -- cada caractere recortado e binarizado sozinho. Testado no
    Dia 5: a mesma placa que a CNN acertava 100% caractere por caractere
    (recorte exato + Otsu local) errava quase tudo com Otsu global da
    placa inteira. Sem `cinza`, cada fatia sai da própria imagem binária.

    Cada fatia é recortada justo nos DOIS eixos (`_apertar_vertical`), não
    só em X -- é assim que os dados de treino da CNN foram gerados, e
    recortar com a altura inteira da faixa entregava à rede um caractere
    achatado e deslocado dentro do quadro de 32x32 (ver DIARIO, Dia 6).
    `margem_vertical` devolve uma folga proporcional à altura do caractere,
    já que a caixa anotada do treino é um pouco mais larga que o blob de
    tinta puro.

    `corte_superior` é uma fração da altura da imagem e assume que a
    entrada é a placa INTEIRA (com a tarja no topo). Se a entrada já for só
    a faixa dos caracteres, passe 0.0 -- senão o corte come o topo das
    letras. Use `CORTE_POR_LAYOUT[layout]` quando o layout for conhecido.
    """
    faixa, perfil = projecao_vertical(binaria, corte_superior)
    largura = faixa.shape[1]

    caixas = _componentes_de_caracteres(faixa)
    if not caixas:
        largura_fatia = largura / n
        caixas = [(round(i * largura_fatia), 0,
                  round((i + 1) * largura_fatia) - round(i * largura_fatia), faixa.shape[0])
                 for i in range(n)]
    caixas = _juntar_ou_dividir(caixas, n, perfil, faixa.shape[0])

    if cinza is not None:
        y0 = int(cinza.shape[0] * corte_superior)
        fonte = cinza[y0:, :]
    else:
        fonte = faixa

    fatias = []
    for x, _, w, _ in caixas:
        # o (y, h) que veio das caixas é descartado de propósito: depois de
        # juntar/dividir ele pode estar largo demais. `_apertar_vertical`
        # remede pela tinta, sempre na binária -- `faixa` e `fonte` têm a
        # mesma origem e altura, então os índices valem pros dois.
        y, h = _apertar_vertical(faixa, x, w)
        folga = int(h * margem_vertical)
        ya, yb = max(0, y - folga), min(fonte.shape[0], y + h + folga)
        pedaco = fonte[ya:yb, x:x + w]
        if pedaco.size == 0:
            fatias.append(np.zeros(saida, dtype=np.uint8))
            continue
        if cinza is not None:
            if int(pedaco.max()) - int(pedaco.min()) >= CONTRASTE_MINIMO:
                pedaco = cv2.threshold(pedaco, 0, 255,
                                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            else:
                # sem contraste pra Otsu decidir: reaproveita a binarização
                # da faixa inteira, que já é conhecida boa, em vez de
                # devolver uma fatia chapada
                pedaco = faixa[ya:yb, x:x + w]
        fatias.append(cv2.resize(pedaco, saida, interpolation=cv2.INTER_AREA))
    return fatias
