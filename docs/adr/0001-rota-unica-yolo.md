# Rota única: YOLO de caracteres

O código mantém só a rota em que o YOLO dá a caixa e a classe de cada caractere. A
segmentação clássica e a CNN saíram porque tiveram acurácia menor nas mesmas 30
placas (0,300 e 0,619, contra 0,719) e obrigavam a manter o TensorFlow. A comparação
continua documentada no README e pode ser refeita pela tag `entrega-v1`.
