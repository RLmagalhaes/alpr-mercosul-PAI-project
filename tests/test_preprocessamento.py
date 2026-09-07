"""Testes da segmentação por componentes conectados (Dia 5) — únicos
testes que dependem de cv2/numpy (os de test_basico.py são de propósito
livres dessas dependências).

    pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessamento import (_componentes_de_caracteres,  # noqa: E402
                                  _juntar_ou_dividir, segmentar)


def _faixa_com_blocos(blocos, largura=210, altura=40):
    """Desenha uma faixa binária (0/255) com um bloco retangular (um
    "caractere") por item de `blocos`, cada um (x, largura_do_bloco)."""
    faixa = np.zeros((altura, largura), dtype=np.uint8)
    for x, w in blocos:
        faixa[8:altura - 8, x:x + w] = 255
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
