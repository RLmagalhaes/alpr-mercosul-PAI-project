"""Testes das operações sobre o recorte da placa — rodam sem GPU e sem modelos.

    pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessamento import (TAMANHO_PLACA, detectar_layout,  # noqa: E402
                                  recortar, retificar_por_cantos)


def test_recortar_aplica_a_margem():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    # caixa de 100x50 com margem de 10% -> 10 px em x e 5 px em y de cada lado
    assert recortar(img, (50, 25, 150, 75), margem=0.10).shape == (60, 120, 3)


def test_recortar_respeita_a_borda_da_imagem():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    assert recortar(img, (0, 0, 100, 50), margem=0.10).shape == (55, 110, 3)


def test_retificar_devolve_o_tamanho_padrao():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    saida = retificar_por_cantos(img, [(10, 10), (150, 20), (140, 70), (20, 60)])
    assert saida.shape == (TAMANHO_PLACA[1], TAMANHO_PLACA[0], 3)


def test_retificar_leva_os_cantos_aos_cantos_da_saida():
    # o quadrante marcado precisa ocupar exatamente a saída inteira
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[20:80, 40:160] = 255
    cantos = [(40, 20), (159, 20), (159, 79), (40, 79)]
    assert retificar_por_cantos(img, cantos).mean() > 250


def test_layout_mercosul_tem_tarja_azul_no_topo():
    placa = np.full((60, 200, 3), 255, dtype=np.uint8)
    placa[:19, :] = (200, 60, 20)     # azul, em BGR
    assert detectar_layout(placa)[0] == "mercosul"


def test_layout_antiga_nao_tem_tarja_azul():
    placa = np.full((60, 200, 3), 200, dtype=np.uint8)
    assert detectar_layout(placa)[0] == "antiga"
