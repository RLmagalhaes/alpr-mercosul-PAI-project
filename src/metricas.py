"""Métricas do projeto ALPR.

Sem dependências pesadas de propósito: roda em qualquer ambiente e é testável
rapidamente com `pytest tests/`.
"""

from typing import Sequence, Tuple

Caixa = Tuple[float, float, float, float]   # (x1, y1, x2, y2) em pixels


def iou(a: Caixa, b: Caixa) -> float:
    """Intersection over Union entre duas caixas no formato (x1, y1, x2, y2).

    Vale 1,0 para caixas idênticas e 0,0 para caixas que não se tocam.
    Uma detecção costuma ser aceita como correta a partir de IoU >= 0,5.
    """
    xi1, yi1 = max(a[0], b[0]), max(a[1], b[1])
    xi2, yi2 = min(a[2], b[2]), min(a[3], b[3])

    intersecao = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    uniao = area_a + area_b - intersecao

    return intersecao / uniao if uniao > 0 else 0.0


def acuracia_caractere(reais: Sequence[str], previstas: Sequence[str],
                       n_caracteres: int = 7) -> float:
    """Fração de caracteres corretos sobre o total.

    Placas não lidas (string vazia) contam como 7 erros — é a medida honesta,
    porque não detectar também é errar.
    """
    if not reais:
        return 0.0
    corretos = sum(
        sum(a == b for a, b in zip(real, prev))
        for real, prev in zip(reais, previstas)
    )
    return corretos / (n_caracteres * len(reais))


def acuracia_placa(reais: Sequence[str], previstas: Sequence[str]) -> float:
    """Fração de placas com TODOS os caracteres corretos.

    É a métrica que importa: uma placa com 6 de 7 caracteres certos é inútil
    na prática.
    """
    if not reais:
        return 0.0
    return sum(r == p for r, p in zip(reais, previstas)) / len(reais)


def previsao_teorica(acc_caractere: float, n_caracteres: int = 7) -> float:
    """Acurácia por placa esperada se os erros fossem independentes.

    Compare com a acurácia medida:
      - medida MAIOR  -> os erros estão concentrados em poucas placas difíceis;
      - medida MENOR  -> há algo sistemático errado em alguma posição.
    """
    return acc_caractere ** n_caracteres


def erros_por_posicao(reais: Sequence[str], previstas: Sequence[str],
                      n_caracteres: int = 7):
    """Conta quantos erros ocorreram em cada uma das posições da placa."""
    contagem = [0] * n_caracteres
    for real, prev in zip(reais, previstas):
        if not prev:
            continue
        for i, (a, b) in enumerate(zip(real, prev)):
            if i < n_caracteres and a != b:
                contagem[i] += 1
    return contagem
