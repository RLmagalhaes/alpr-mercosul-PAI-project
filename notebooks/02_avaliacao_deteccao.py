# Dia 2 — Avaliacao da deteccao e recorte das placas
#
# Roda por blocos no Colab (sessao "dia1"), reaproveitando o ambiente/dados
# ja preparados no Dia 1 (drive montado, dataset em /content/dados/deteccao,
# pesos treinados em /content/modelos/detector/weights).

# ---------------------------------------------------------------
# 2.1 Metricas no conjunto de teste
# ---------------------------------------------------------------
import os, pandas as pd
from ultralytics import YOLO

RAIZ = '/content/drive/MyDrive/alpr-mercosul'
DADOS = '/content/dados/deteccao'

det = YOLO('/content/modelos/detector/weights/best.pt')
m = det.val(data=f'{DADOS}/data.yaml', split='test', plots=True)

metricas = {
    'mAP@0.5':      round(float(m.box.map50), 4),
    'mAP@0.5:0.95': round(float(m.box.map),   4),
    'Precisao':     round(float(m.box.mp),    4),
    'Recall':       round(float(m.box.mr),    4),
}
for k, v in metricas.items():
    print(f'{k:>14}: {v}')

os.makedirs(f'{RAIZ}/resultados/tabelas', exist_ok=True)
pd.DataFrame([metricas]).to_csv(f'{RAIZ}/resultados/tabelas/metricas_deteccao.csv', index=False)
print('CSV salvo em', f'{RAIZ}/resultados/tabelas/metricas_deteccao.csv')

# Resultado obtido (conjunto de teste, 257 imagens, 268 instancias):
#   mAP@0.5      = 0.992
#   mAP@0.5:0.95 = 0.8344
#   Precisao     = 0.9776
#   Recall       = 0.9768
#
# Observacao de qualidade de dados: o ultralytics avisou que 33 das 257
# imagens de teste tem anotacao em formato de poligono (segmentacao) em vez
# de caixa retangular (268 instancias, 33 com "segments"). Ele converteu
# automaticamente pra bounding box e seguiu normal — nao invalida a metrica,
# mas fica registrado como inconsistencia do dataset original (Roboflow).


# ---------------------------------------------------------------
# 2.2 IoU implementado do zero
# ---------------------------------------------------------------
def iou(a, b):
    """IoU entre duas caixas no formato (x1, y1, x2, y2), em pixels."""
    xi1, yi1 = max(a[0], b[0]), max(a[1], b[1])
    xi2, yi2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    uniao = area_a + area_b - inter
    return inter / uniao if uniao > 0 else 0.0

# Testes rapidos de sanidade (rodados fora do Colab, sao so matematica pura)
print(iou((0, 0, 10, 10), (0, 0, 10, 10)))    # esperado 1.0  -> obtido 1.0
print(iou((0, 0, 10, 10), (20, 20, 30, 30)))  # esperado 0.0  -> obtido 0.0
print(round(iou((0, 0, 10, 10), (5, 0, 15, 10)), 4))  # esperado 0.3333 -> obtido 0.3333


# ---------------------------------------------------------------
# 2.3 Efeito do limiar de NMS
# ---------------------------------------------------------------
# O NMS remove caixas duplicadas do mesmo objeto. Limiar baixo remove mais.
import glob, os

# Primeira tentativa: imagem qualquer do teste (a maioria tem so 1 placa,
# entao o NMS nao tem o que remover -- resultado e sempre 1 deteccao)
exemplo = sorted(glob.glob(f'{DADOS}/test/images/*'))[0]
print('Imagem de exemplo:', exemplo)
for limiar in [0.3, 0.5, 0.7, 0.9]:
    r = det.predict(exemplo, iou=limiar, conf=0.25, verbose=False)[0]
    print(f'NMS iou={limiar} -> {len(r.boxes)} deteccao(oes)')
# Resultado obtido: 1 deteccao em todos os limiares (imagem so tem 1 placa)

# Segunda tentativa: busca uma imagem com mais de 1 placa anotada, pra ver
# o NMS de fato entrando em acao (268 instancias em 257 imagens -> ao menos
# 11 fotos tem 2+ placas)
melhor = None
for rot in sorted(glob.glob(f'{DADOS}/test/labels/*.txt')):
    n = sum(1 for _ in open(rot))
    if n > 1:
        melhor = rot
        break
img_multi = melhor.replace('/labels/', '/images/').rsplit('.', 1)[0] + '.jpg'
print('Imagem com multiplas placas:', img_multi)
for limiar in [0.3, 0.5, 0.7, 0.9]:
    r = det.predict(img_multi, iou=limiar, conf=0.25, verbose=False)[0]
    print(f'NMS iou={limiar} -> {len(r.boxes)} deteccao(oes)')

# Resultado obtido:
#   iou=0.3 -> 2 deteccoes
#   iou=0.5 -> 2 deteccoes
#   iou=0.7 -> 2 deteccoes
#   iou=0.9 -> 3 deteccoes  <- limiar muito permissivo deixa passar uma
#                              caixa duplicada da mesma placa


# ---------------------------------------------------------------
# 2.4 Onde o modelo erra
# ---------------------------------------------------------------
# Compara predicao com anotacao usando a nossa funcao iou() e separa os
# piores casos.
import glob, os, cv2, pandas as pd

registros = []
for caminho in sorted(glob.glob(f'{DADOS}/test/images/*')):
    img = cv2.imread(caminho)
    h, w = img.shape[:2]
    rot = caminho.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
    reais = []
    if os.path.exists(rot):
        for l in open(rot):
            partes = l.split()
            if len(partes) < 5:
                continue
            xc, yc, bw, bh = map(float, partes[1:5])
            reais.append(((xc-bw/2)*w, (yc-bh/2)*h, (xc+bw/2)*w, (yc+bh/2)*h))
    if not reais:
        continue
    pred = det.predict(caminho, conf=0.25, verbose=False)[0].boxes
    caixas_pred = pred.xyxy.cpu().numpy() if len(pred) else []
    melhor = max([iou(r, p) for r in reais for p in caixas_pred], default=0.0)
    registros.append(dict(arquivo=os.path.basename(caminho), melhor_iou=round(melhor, 3),
                          brilho=round(float(img.mean()), 1)))

erros = pd.DataFrame(registros).sort_values('melhor_iou')
print(f'Placas nao detectadas (IoU = 0): {(erros.melhor_iou == 0).sum()} de {len(erros)}')
print(erros.head(10).to_string(index=False))
erros.to_csv(f'{RAIZ}/resultados/tabelas/analise_erros.csv', index=False)

# Resultado obtido: 2 de 257 placas nao detectadas (IoU=0). As demais nos
# piores casos ainda tem IoU muito baixo (0.01-0.03), sugerindo deteccoes
# quase completamente erradas, nao so imprecisas. Fotos escuras aparecem
# entre os piores (brilho ~60-90 em alguns casos), mas nao e um padrao
# absoluto (tem casos com brilho >110 tambem entre os piores).


# ---------------------------------------------------------------
# 2.5 Recortar todas as placas
# ---------------------------------------------------------------
# Usa as caixas ANOTADAS (nao as previstas pelo detector): sao exatas e
# garantem que o Dia 3 comece com material limpo. Os recortes ficam so no
# disco local da VM (/content) -- nunca no Drive, pra nao repetir o problema
# de cota que ja aconteceu uma vez (ver DIARIO, Dia 2). Como e rapido de
# refazer (poucos minutos, sem GPU), essa celula deve ser rodada de novo no
# inicio da sessao do Dia 3, em vez de tentar persistir os recortes entre
# sessoes.
DESTINO_LOCAL = "/content/dados/placas_recortadas"


def parse_anotacao(partes):
    """Devolve (xc, yc, bw, bh) normalizados. Se a linha vier em poligono
    (mais de 4 pares de coordenadas, formato que ja apareceu em 2.1),
    converte para a caixa delimitadora minima -- mesma conversao que o
    Ultralytics ja faz sozinho na validacao."""
    valores = list(map(float, partes[1:]))
    if len(valores) == 4:
        return tuple(valores), False
    if len(valores) >= 6 and len(valores) % 2 == 0:
        xs, ys = valores[0::2], valores[1::2]
        xc, yc = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        bw, bh = max(xs) - min(xs), max(ys) - min(ys)
        return (xc, yc, bw, bh), True
    return None, False


def recortar_todas(split, margem=0.08, largura_min=30, altura_min=10):
    destino = f"{DESTINO_LOCAL}/{split}"
    os.makedirs(destino, exist_ok=True)
    c = dict(ok=0, poligono_convertido=0, imagem_ilegivel=0, sem_rotulo=0,
             anotacao_invalida=0, recorte_pequeno_demais=0,
             recortes_muito_pequenos_mas_aceitos=0)

    for caminho in sorted(glob.glob(f"{DADOS}/{split}/images/*")):
        img = cv2.imread(caminho)
        if img is None:
            c["imagem_ilegivel"] += 1
            continue
        h, w = img.shape[:2]
        rot = caminho.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        if not os.path.exists(rot):
            c["sem_rotulo"] += 1
            continue
        base = os.path.splitext(os.path.basename(caminho))[0]
        for i, l in enumerate(open(rot)):
            partes = l.split()
            if len(partes) < 5:
                c["anotacao_invalida"] += 1
                continue
            anotacao, era_poligono = parse_anotacao(partes)
            if anotacao is None:
                c["anotacao_invalida"] += 1
                continue
            if era_poligono:
                c["poligono_convertido"] += 1
            xc, yc, bw, bh = anotacao
            x1, y1 = (xc - bw / 2) * w, (yc - bh / 2) * h
            x2, y2 = (xc + bw / 2) * w, (yc + bh / 2) * h
            mx, my = (x2 - x1) * margem, (y2 - y1) * margem
            x1, y1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
            x2, y2 = min(w, int(x2 + mx)), min(h, int(y2 + my))
            recorte = img[y1:y2, x1:x2]
            if recorte.size == 0 or recorte.shape[0] < altura_min or recorte.shape[1] < largura_min:
                c["recorte_pequeno_demais"] += 1
                continue
            if recorte.shape[0] < 40 or recorte.shape[1] < 100:
                c["recortes_muito_pequenos_mas_aceitos"] += 1
            cv2.imwrite(f"{destino}/{base}_{i}.jpg", recorte)
            c["ok"] += 1
    return c


resumo_recorte = {s: recortar_todas(s) for s in ["train", "valid", "test"]}
for s, c in resumo_recorte.items():
    print(s, c)

import json
with open(f"{RAIZ}/resultados/tabelas/recorte_placas_resumo.json", "w") as f:
    json.dump(resumo_recorte, f, indent=2, ensure_ascii=False)

# Resultado obtido (rodado em 03/09):
#   train: ok=12539, poligono_convertido=621, recorte_pequeno_demais=847,
#          muito_pequenos_mas_aceitos=9359
#   valid: ok=941,  poligono_convertido=66,  recorte_pequeno_demais=54,
#          muito_pequenos_mas_aceitos=825
#   test:  ok=245,  poligono_convertido=33,  recorte_pequeno_demais=23,
#          muito_pequenos_mas_aceitos=236
#   Total recortado: 13725 placas.
#
# PROBLEMA DE DADOS REGISTRADO (nao contornado, so documentado): 76% dos
# recortes aceitos (10420 de 13725) sao "muito pequenos" (altura<40px ou
# largura<100px). A galeria salva em
# resultados/figuras/recortes_placas_tamanhos.png mostra a diferenca: os
# recortes normais tem o texto nitido e legivel; os pequenos ja saem
# borrados so pelo redimensionamento da miniatura, antes mesmo de qualquer
# pre-processamento do Dia 3. Isso e uma limitacao do dataset de origem
# (fotos de resolucao/enquadramento variados), nao um bug do recorte -- o
# efeito esperado e que a segmentacao de caracteres do Dia 3 tenha uma taxa
# de falha bem maior nesse subconjunto pequeno, e isso deve aparecer
# explicitamente na analise de erros, sem tentar mascarar com upscaling ou
# heuristica de "salvamento" desses casos.
