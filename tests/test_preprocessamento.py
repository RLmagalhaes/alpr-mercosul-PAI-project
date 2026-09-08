"""Testes da segmentação por componentes conectados (Dia 5) — únicos
testes que dependem de cv2/numpy (os de test_basico.py são de propósito
livres dessas dependências).

    pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessamento import (_apertar_vertical,  # noqa: E402
                                  _componentes_de_caracteres,
                                  _juntar_ou_dividir, segmentar)


def _faixa_com_blocos(blocos, largura=210, altura=40):
    """Desenha uma faixa binária (0/255) com um bloco retangular (um
    "caractere") por item de `blocos`, cada um (x, largura_do_bloco).

    ATENÇÃO: aqui todo bloco ocupa a MESMA extensão vertical. Foi por isso
    que os dois bugs do Dia 5 passaram batido nesta suíte -- num cenário
    assim, descartar o (y, h) de cada caractere não muda nada. Para testar
    o recorte vertical use `_faixa_com_blocos_variados`.
    """
    faixa = np.zeros((altura, largura), dtype=np.uint8)
    for x, w in blocos:
        faixa[8:altura - 8, x:x + w] = 255
    return faixa


def _faixa_com_blocos_variados(blocos, largura=210, altura=40):
    """Como `_faixa_com_blocos`, mas cada bloco tem topo e altura próprios.

    `blocos` é uma lista de (x, largura, y, altura) -- é o que aproxima o
    caso real, em que "1" e "W" têm larguras diferentes e acentos/alturas
    de caractere não coincidem exatamente.
    """
    faixa = np.zeros((altura, largura), dtype=np.uint8)
    for x, w, y, h in blocos:
        faixa[y:y + h, x:x + w] = 255
    return faixa


# ------------------------------------------ _componentes_de_caracteres

def test_componentes_acha_um_por_bloco():
    # blocos de largura BEM diferente -- "W" largo, "1" estreito
    faixa = _faixa_com_blocos([(5, 30), (45, 8), (65, 20), (95, 20)])
    caixas = _componentes_de_caracteres(faixa)
    assert len(caixas) == 4
    # ordenado da esquerda pra direita
    assert [c[0] for c in caixas] == sorted(c[0] for c in caixas)


def test_componentes_ignora_ruido_baixo():
    faixa = _faixa_com_blocos([(5, 20), (40, 20)])
    faixa[36:38, 65:68] = 255   # mancha baixinha (ruído), não é caractere
    caixas = _componentes_de_caracteres(faixa)
    assert len(caixas) == 2


# ------------------------------------------------- _juntar_ou_dividir

def test_juntar_reduz_fragmentos_ao_numero_certo():
    # um caractere "quebrado" em 2 componentes bem próximos
    caixas = [(5, 0, 10, 20), (16, 0, 8, 20), (40, 0, 15, 20)]
    perfil = np.full(210, 30.0)
    ajustadas = _juntar_ou_dividir(caixas, 2, perfil)
    assert len(ajustadas) == 2


def test_dividir_aumenta_para_o_numero_certo():
    caixas = [(5, 0, 15, 20)]
    perfil = np.full(210, 30.0)
    ajustadas = _juntar_ou_dividir(caixas, 3, perfil)
    assert len(ajustadas) == 3


# ------------------------------------------------------- segmentar

def test_segmentar_respeita_largura_variavel_dos_caracteres():
    # 7 blocos de largura bem diferente entre si (como letras/dígitos reais)
    larguras = [30, 8, 18, 25, 10, 22, 14]
    x = 5
    blocos = []
    for w in larguras:
        blocos.append((x, w))
        x += w + 6
    binaria = np.zeros((60, x + 10), dtype=np.uint8)
    binaria[25:55, :] = _faixa_com_blocos(blocos, largura=x + 10, altura=30)
    fatias = segmentar(binaria, corte_superior=25 / 60)
    assert len(fatias) == 7
    for fatia in fatias:
        assert fatia.max() == 255


def test_segmentar_sem_componentes_cai_em_divisao_igual():
    binaria = np.zeros((60, 210), dtype=np.uint8)   # faixa totalmente vazia
    fatias = segmentar(binaria)
    assert len(fatias) == 7
    for fatia in fatias:
        assert fatia.shape == (32, 32)


def test_segmentar_com_cinza_binariza_por_fatia():
    larguras = [30, 8, 18, 25, 10, 22, 14]
    x = 5
    blocos = []
    for w in larguras:
        blocos.append((x, w))
        x += w + 6
    largura_total = x + 10
    altura = 60
    corte_superior = 25 / altura
    y0 = int(altura * corte_superior)

    binaria = np.zeros((altura, largura_total), dtype=np.uint8)
    cinza = np.zeros((altura, largura_total), dtype=np.uint8)
    for i, (bx, bw) in enumerate(blocos):
        fundo = 60 + i * 20         # fundo diferente por caractere
        cinza[y0:, bx:bx + bw + 6] = fundo
        binaria[y0 + 5:altura - 5, bx:bx + bw] = 255
        cinza[y0 + 5:altura - 5, bx:bx + bw] = min(fundo + 80, 255)

    fatias = segmentar(binaria, corte_superior=corte_superior, cinza=cinza)
    assert len(fatias) == 7
    for fatia in fatias:
        assert fatia.max() == 255


# ------------------------------------------------- _apertar_vertical

def test_apertar_vertical_mede_a_extensao_da_tinta():
    faixa = _faixa_com_blocos_variados([(5, 20, 7, 18)], altura=40)
    y, h = _apertar_vertical(faixa, 5, 20)
    assert (y, h) == (7, 18)


def test_apertar_vertical_sem_tinta_devolve_a_faixa_inteira():
    faixa = np.zeros((40, 210), dtype=np.uint8)
    assert _apertar_vertical(faixa, 5, 20) == (0, 40)


# ------------------------- regressão dos bugs do Dia 5 (ver DIARIO, Dia 6)

def _sete_blocos_de_alturas_diferentes(altura=40):
    """7 "caracteres" com alturas e topos diferentes, todos dentro do que os
    filtros de `_componentes_de_caracteres` aceitam (altura >= 50% da maior).
    """
    alturas = [30, 22, 26, 24, 30, 20, 28]
    blocos, x = [], 5
    for i, h in enumerate(alturas):
        w = 8 + (i * 5) % 18                 # larguras também variadas
        y = 4 + (i % 3) * 2                  # topos ligeiramente diferentes
        blocos.append((x, w, y, h))
        x += w + 7
    return _faixa_com_blocos_variados(blocos, largura=x + 10, altura=altura)


def test_segmentar_recorta_justo_na_vertical():
    """Bug 2 do Dia 5: `segmentar` recortava `fonte[:, x:x+w]` -- justo em X,
    altura INTEIRA da faixa. Um caractere de 20px numa faixa de 40px chegava
    à CNN preenchendo metade do quadro, enquanto no treino ele preenchia o
    quadro todo. Com o recorte justo nos dois eixos, todo bloco maciço tem
    que preencher a fatia 32x32 quase inteira, independente da sua altura.
    """
    faixa = _sete_blocos_de_alturas_diferentes()
    binaria = np.zeros((60, faixa.shape[1]), dtype=np.uint8)
    binaria[20:, :] = faixa
    fatias = segmentar(binaria, corte_superior=20 / 60)
    assert len(fatias) == 7
    for i, fatia in enumerate(fatias):
        # antes da correção, os blocos mais baixos ficavam perto de 160
        assert fatia.mean() > 240, f"fatia {i} não preenche o quadro: {fatia.mean():.0f}"


def test_segmentar_ignora_a_posicao_vertical_do_caractere():
    """Dois caracteres idênticos em alturas diferentes da faixa têm que sair
    como fatias iguais -- é o que garante que a CNN veja sempre o mesmo
    enquadramento, como no treino."""
    alto = _faixa_com_blocos_variados([(5, 20, 2, 24)], largura=60, altura=40)
    baixo = _faixa_com_blocos_variados([(5, 20, 14, 24)], largura=60, altura=40)
    f_alto = segmentar(alto, n=1, corte_superior=0.0)[0]
    f_baixo = segmentar(baixo, n=1, corte_superior=0.0)[0]
    assert np.array_equal(f_alto, f_baixo)


def test_segmentar_com_corte_zero_preserva_o_topo_dos_caracteres():
    """Bug 1 do Dia 5: `corte_superior=0.35` era aplicado mesmo quando a
    entrada já era só a faixa dos caracteres (sem tarja), decapitando o topo
    de cada um. Com corte 0.0 a tinta tem que sobreviver inteira."""
    binaria = _faixa_com_blocos_variados(
        [(5, 20, 0, 30), (35, 20, 0, 30)], largura=70, altura=40)
    tinta_original = int((binaria > 0).sum())
    fatias = segmentar(binaria, n=2, corte_superior=0.0)
    assert len(fatias) == 2
    for fatia in fatias:
        assert fatia.mean() > 240
    # com o corte errado (0.35) a tinta de cima seria perdida
    decapitadas = segmentar(binaria, n=2, corte_superior=0.35)
    assert tinta_original > 0
    assert all(f.mean() > 240 for f in decapitadas)  # ainda "parece" ok...
    # ...mas o corte comeu tinta de verdade: é isso que enganava a inspeção visual
    assert int((binaria[int(40 * 0.35):] > 0).sum()) < tinta_original


def test_segmentar_fatia_sem_contraste_nao_sai_chapada():
    """Recorte justo de caractere sólido ("1", "I") pode sair quase uniforme,
    e Otsu sem contraste devolveria a fatia inteira preta. O fallback tem que
    reaproveitar a binarização da faixa."""
    binaria = _faixa_com_blocos_variados([(5, 10, 5, 30)], largura=30, altura=40)
    cinza = np.full((40, 30), 200, dtype=np.uint8)   # uniforme de propósito
    fatia = segmentar(binaria, n=1, corte_superior=0.0, cinza=cinza)[0]
    assert fatia.max() == 255
