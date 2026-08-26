"""Testes rápidos das funções puras — rodam sem GPU e sem modelos treinados.

    pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metricas import (acuracia_caractere, acuracia_placa,  # noqa: E402
                          erros_por_posicao, iou, previsao_teorica)
from src.validacao import (aplicar_mascara, formato_valido,  # noqa: E402
                           inferir_layout_por_texto)


# ---------------------------------------------------------------- IoU

def test_iou_caixas_identicas():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_caixas_disjuntas():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_metade_sobreposta():
    # interseção = 5x10 = 50 ; união = 100 + 100 - 50 = 150 ; 50/150 = 1/3
    assert round(iou((0, 0, 10, 10), (5, 0, 15, 10)), 4) == 0.3333


def test_iou_uma_dentro_da_outra():
    # interseção = 25 ; união = 100 ; 0,25
    assert iou((0, 0, 10, 10), (0, 0, 5, 5)) == 0.25


# ---------------------------------------------- regra do formato

def test_mascara_corrige_digito_em_posicao_de_letra():
    assert aplicar_mascara("A8C1D23", "mercosul") == "ABC1D23"


def test_mascara_corrige_letra_em_posicao_de_digito():
    assert aplicar_mascara("ABC1O23", "antiga") == "ABC1023"


def test_mascara_nao_estraga_placa_correta():
    assert aplicar_mascara("ABC1D23", "mercosul") == "ABC1D23"
    assert aplicar_mascara("ABC1234", "antiga") == "ABC1234"


def test_formato_valido():
    assert formato_valido("ABC1D23", "mercosul")
    assert not formato_valido("ABC1223", "mercosul")   # posição 5 deveria ser letra
    assert formato_valido("ABC1234", "antiga")


def test_inferir_layout():
    assert inferir_layout_por_texto("ABC1D23") == "mercosul"
    assert inferir_layout_por_texto("ABC1234") == "antiga"


# ------------------------------------------------------- métricas

def test_acuracia_placa():
    reais = ["ABC1D23", "XYZ9K88", "AAA1A11"]
    previstas = ["ABC1D23", "XYZ9K80", "AAA1A11"]
    assert round(acuracia_placa(reais, previstas), 4) == 0.6667


def test_acuracia_caractere():
    reais = ["ABC1D23"]
    previstas = ["ABC1D20"]      # 6 de 7 corretos
    assert round(acuracia_caractere(reais, previstas), 4) == 0.8571


def test_placa_nao_lida_conta_como_erro():
    assert acuracia_caractere(["ABC1D23"], [""]) == 0.0
    assert acuracia_placa(["ABC1D23"], [""]) == 0.0


def test_previsao_teorica():
    # 0,95 por caractere -> ~0,698 por placa. É por isso que a regra existe.
    assert round(previsao_teorica(0.95), 4) == 0.6983


def test_erros_por_posicao():
    reais = ["ABC1D23", "ABC1D23"]
    previstas = ["ABC1D20", "ABC1D20"]   # ambos erram a última posição
    assert erros_por_posicao(reais, previstas) == [0, 0, 0, 0, 0, 0, 2]
