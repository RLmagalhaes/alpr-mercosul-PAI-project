"""Regra de validação do formato das placas brasileiras.

Sabendo o layout, cada posição da placa só aceita letra OU só dígito. Isso
permite corrigir boa parte dos erros do classificador sem treinar mais nada —
é conhecimento de domínio somado ao modelo.

    Mercosul (carro)  : 3 letras · 1 dígito · 1 letra · 2 dígitos  -> ABC1D23
    Brasileira antiga : 3 letras · 4 dígitos                       -> ABC1234
"""

CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

MASCARAS = {
    "mercosul": "LLLDLDD",
    "antiga":   "LLLDDDD",
}

# Trocas derivadas da matriz de confusão da CNN (Dia 4).
# Ajuste estes dicionários com os pares que aparecerem no SEU modelo.
PARA_LETRA = {
    "0": "O", "1": "I", "2": "Z", "4": "A",
    "5": "S", "6": "G", "7": "T", "8": "B",
}

PARA_DIGITO = {
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1",
    "Z": "2", "A": "4", "S": "5", "G": "6", "T": "7", "B": "8",
}


def aplicar_mascara(texto: str, layout: str = "mercosul") -> str:
    """Substitui caracteres impossíveis para a posição pelo equivalente visual.

    >>> aplicar_mascara("A8C1D23", "mercosul")   # o 8 na posição 2 vira B
    'ABC1D23'
    >>> aplicar_mascara("ABC1O23", "antiga")     # o O na posição 5 vira 0
    'ABC1023'
    """
    mascara = MASCARAS.get(layout, MASCARAS["mercosul"])
    saida = []
    for caractere, tipo in zip(texto, mascara):
        if tipo == "L" and caractere.isdigit():
            saida.append(PARA_LETRA.get(caractere, caractere))
        elif tipo == "D" and caractere.isalpha():
            saida.append(PARA_DIGITO.get(caractere, caractere))
        else:
            saida.append(caractere)
    return "".join(saida)


def formato_valido(texto: str, layout: str = "mercosul") -> bool:
    """Diz se o texto respeita a máscara do layout informado."""
    mascara = MASCARAS.get(layout, MASCARAS["mercosul"])
    if len(texto) != len(mascara):
        return False
    for caractere, tipo in zip(texto, mascara):
        if tipo == "L" and not caractere.isalpha():
            return False
        if tipo == "D" and not caractere.isdigit():
            return False
    return True


def inferir_layout_por_texto(texto: str) -> str:
    """Palpite de layout a partir do próprio texto lido.

    Só use como reforço: o método principal é olhar a tarja azul da placa
    (ver `preprocessamento.detectar_layout`), que não depende do acerto da CNN.
    """
    if len(texto) != 7:
        return "mercosul"
    return "mercosul" if texto[4].isalpha() else "antiga"
