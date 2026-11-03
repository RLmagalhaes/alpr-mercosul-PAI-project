# Roteiro Express — 7 Dias

## ALPR Mercosul: Detecção e Reconhecimento de Placas Veiculares

**Aluno:** Raphael Magalhães · **Prazo:** 7 dias · **Ambiente:** Google Colab (GPU T4) + Google Drive
**Modo:** o agente escreve todo o código; você lê, valida e roda.

---

## Briefing do agente

> Cole o bloco abaixo na primeira mensagem, junto com este roteiro.

```
Você é meu par de programação neste projeto final de pós-graduação. Vamos
construir em 7 dias um sistema ALPR (reconhecimento de placas Mercosul).
Rodo tudo no Google Colab gratuito. Tenho Python intermediário: eu leio e
valido código, mas não quero escrever do zero.

COMO TRABALHAR

1. VOCÊ ESCREVE TODO O CÓDIGO. Sempre completo, comentado em português e
   pronto para colar numa célula do Colab. Nunca me peça para escrever ou
   completar trechos.

2. ANTES DO CÓDIGO: 2 a 3 linhas dizendo o que o bloco faz e por quê.

3. DEPOIS DO CÓDIGO: diga qual saída eu devo esperar. Vou colar a saída real
   de volta para você.

4. QUANDO EU COLAR A SAÍDA: explique o que aconteceu em linguagem simples,
   diga se o resultado está bom ou ruim, e sugira correções se for o caso.

5. PERGUNTE ANTES DE AVANÇAR. Ao fim de cada etapa: "seguimos para a etapa X
   ou quer ajustar algo aqui?". Espere minha resposta.

6. SE DER ERRO: corrija direto, com o bloco corrigido inteiro, e explique em
   2 linhas qual era o problema. Não me faça adivinhar.

7. SEJA HONESTO COM MÉTRICAS. Se ficou ruim, diga que ficou ruim e proponha
   o que tentar. Não maquie resultado.

8. PRIORIZE TER ALGO FUNCIONANDO. Diante de dúvida entre elegante e pronto,
   escolha pronto. Refinamento é bônus.

RESTRIÇÕES
- Colab gratuito: cota diária de GPU e sessão que cai por inatividade. Salve
  tudo no Drive e evite treinos maiores que ~40 minutos seguidos.
- Comece cada sessão me dizendo, em 3 linhas, o objetivo do dia e o entregável.
```

---

## O projeto em uma página

```
 FOTO DO VEÍCULO
      │
      ▼
 [1] YOLO detecta a placa ................ bounding box + confiança
      ▼
 [2] Recorte + endireitamento ............ warpAffine / warpPerspective
      ▼
 [3] Realce: cinza → CLAHE → Otsu ........ binarização
      ▼
 [4] Segmentação dos 7 caracteres ........ faixa inferior dividida em 7
      ▼
 [5] CNN classifica cada caractere ....... 36 classes (A–Z, 0–9)
      ▼
 [6] Regra do formato Mercosul ........... corrige O↔0, I↔1 por posição
      ▼
   "ABC1D23" + confiança  →  API FastAPI
```

**Métricas de entrega**

| Métrica | Meta |
| --- | --- |
| mAP@0.5 da detecção | > 0,90 |
| Acurácia por caractere | > 0,95 |
| Acurácia por placa (os 7 corretos) | > 0,80 |

Com 95% por caractere, a chance de acertar os 7 é 0,95⁷ ≈ **70%**. É por isso que a etapa [6] existe — e esse cálculo rende um bom parágrafo no relatório.

**Cronograma**

| Dia | Entrega |
| --- | --- |
| 1 | Ambiente, dados, análise rápida e **detector treinado** |
| 2 | Métricas da detecção, análise de erros e placas recortadas |
| 3 | Pré-processamento e **dataset de caracteres** |
| 4 | **CNN de caracteres** treinada e avaliada |
| 5 | **Pipeline fim a fim** + regra de formato + métricas finais |
| 6 | ONNX, **API FastAPI** e latência |
| 7 | Relatório e slides |

**Núcleo inegociável:** Dias 1, 3, 4, 5 e 7. Se atrasar, corte nesta ordem: comparativo de arquiteturas (Dia 4, opcional) → Docker (Dia 6) → análise de erros aprofundada (Dia 2).

---

## Antes do Dia 1 — 30 minutos

**1. Baixe hoje um dataset do Roboflow Universe** (sai pronto, em formato YOLO):

- [License Plate Recognition — com caracteres anotados](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e) ← **este é o mais importante**, porque as caixas de caractere viram o dataset do Dia 3
- [Placas Brasil](https://universe.roboflow.com/license-plates-brazil/placas-brasil-no2nm-fhbl7)

Crie uma conta gratuita, entre no dataset, clique em **Download this Dataset → YOLOv8 → show download code** e guarde o trecho gerado.

**2. Peça o RodoSol-ALPR** (20 mil imagens brasileiras/Mercosul com o texto da placa anotado): <https://github.com/raysonlaroca/rodosol-alpr-dataset>. A liberação leva de 1 a 5 dias úteis — se chegar até o Dia 3, você faz um upgrade no dataset; se não chegar, o projeto acontece igual.

**3. Crie o repositório** `alpr-mercosul` no GitHub e ative a GPU no Colab (*Ambiente de execução → Alterar o tipo de ambiente → T4*).

---

# DIA 1 — Ambiente, dados e detector treinado

**Objetivo:** ao final do dia, um modelo YOLO que encontra placas em fotos.

### 1.1 Montar o ambiente

```python
# Monta o Drive e cria a estrutura de pastas do projeto.
from google.colab import drive
drive.mount('/content/drive')

import os
RAIZ = '/content/drive/MyDrive/alpr-mercosul'
for p in ['dados', 'modelos', 'resultados/figuras', 'resultados/tabelas']:
    os.makedirs(f'{RAIZ}/{p}', exist_ok=True)

!nvidia-smi -L
!pip -q install ultralytics
print('Raiz do projeto:', RAIZ)
```

**Saída esperada:** o nome de uma GPU (`Tesla T4`) e o caminho da raiz. Se não aparecer GPU, ative-a no menu e rode de novo.

### 1.2 Baixar o dataset

```python
# Cole aqui o trecho que o Roboflow gerou, mudando apenas o location.
!pip -q install roboflow
from roboflow import Roboflow

rf = Roboflow(api_key="SUA_CHAVE_AQUI")
projeto = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
ds = projeto.version(4).download("yolov8", location=f"{RAIZ}/dados/placas")

DADOS = f"{RAIZ}/dados/placas"
print(os.listdir(DADOS))
```

### 1.3 Corrigir o `data.yaml`

O `data.yaml` do Roboflow costuma vir com caminhos relativos que quebram no Colab. Este bloco reescreve com caminhos absolutos — é um dos erros que mais fazem perder tempo.

```python
import yaml

cfg = yaml.safe_load(open(f'{DADOS}/data.yaml'))
cfg['path']  = DADOS
cfg['train'] = 'train/images'
cfg['val']   = 'valid/images'
cfg['test']  = 'test/images'
yaml.safe_dump(cfg, open(f'{DADOS}/data.yaml', 'w'), allow_unicode=True)

print('Classes:', cfg['names'])
print('Total  :', cfg['nc'])
```

**Atenção:** anote se este dataset tem **1 classe** (só a placa) ou **37 classes** (placa + os 36 caracteres). Isso decide o caminho do Dia 3.

### 1.4 Inventário do dataset

```python
# Percorre as três partições e monta uma tabela com tamanho, nº de caixas,
# área relativa da placa e brilho médio de cada imagem.
import glob, cv2, numpy as np, pandas as pd

def inventario(split):
    linhas = []
    for caminho in sorted(glob.glob(f'{DADOS}/{split}/images/*')):
        img = cv2.imread(caminho)
        if img is None:
            continue
        h, w = img.shape[:2]
        rot = caminho.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
        caixas = []
        if os.path.exists(rot):
            caixas = [l.split() for l in open(rot) if l.strip()]
        area = float(caixas[0][3]) * float(caixas[0][4]) * 100 if caixas else np.nan
        linhas.append(dict(split=split, arquivo=os.path.basename(caminho),
                           largura=w, altura=h, n_caixas=len(caixas),
                           area_pct=area, brilho=float(img.mean())))
    return pd.DataFrame(linhas)

df = pd.concat([inventario(s) for s in ['train', 'valid', 'test']], ignore_index=True)

resumo = df.groupby('split').agg(imagens=('arquivo', 'count'),
                                 caixas=('n_caixas', 'sum'),
                                 area_media_pct=('area_pct', 'mean'),
                                 brilho_medio=('brilho', 'mean')).round(2)
print(resumo)
df.to_csv(f'{RAIZ}/resultados/tabelas/inventario.csv', index=False)
```

**O que olhar na saída:** a coluna `area_media_pct` costuma ficar abaixo de 2% — a placa é minúscula dentro da foto, e essa é a principal dificuldade da detecção. Guarde esse número, ele vai para o relatório.

### 1.5 Galeria de amostras

```python
import matplotlib.pyplot as plt

amostras = df[df.split == 'train'].sample(6, random_state=0)
fig, eixos = plt.subplots(2, 3, figsize=(15, 7))

for ax, (_, linha) in zip(eixos.ravel(), amostras.iterrows()):
    caminho = f"{DADOS}/train/images/{linha.arquivo}"
    img = cv2.cvtColor(cv2.imread(caminho), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    rot = caminho.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
    if os.path.exists(rot):
        for l in open(rot):
            _, xc, yc, bw, bh = map(float, l.split()[:5])
            x1, y1 = int((xc - bw/2)*w), int((yc - bh/2)*h)
            x2, y2 = int((xc + bw/2)*w), int((yc + bh/2)*h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    ax.imshow(img); ax.axis('off')

plt.tight_layout()
plt.savefig(f'{RAIZ}/resultados/figuras/amostras.png', dpi=120, bbox_inches='tight')
plt.show()
```

### 1.6 Treinar o detector

Antes do treino de verdade, um ensaio de 3 épocas: se houver erro de caminho ou de formato, ele aparece em 2 minutos em vez de 40.

```python
from ultralytics import YOLO

ensaio = YOLO('yolo11n.pt')
ensaio.train(data=f'{DADOS}/data.yaml', epochs=3, imgsz=640, batch=16,
             project=f'{RAIZ}/modelos', name='ensaio', exist_ok=True, verbose=True)
```

Se o ensaio rodou, vá para o treino completo:

```python
# 40 épocas, com checkpoint a cada 5 e parada antecipada por falta de melhora.
modelo = YOLO('yolo11n.pt')
resultados = modelo.train(
    data=f'{DADOS}/data.yaml',
    epochs=40, imgsz=640, batch=16, seed=42,
    project=f'{RAIZ}/modelos', name='detector', exist_ok=True,
    save_period=5, patience=10, plots=True,
)
print('Pesos salvos em:', f'{RAIZ}/modelos/detector/weights/best.pt')
```

**Saída esperada:** uma tabela por época com `box_loss`, `cls_loss`, `dfl_loss` e as métricas `mAP50` e `mAP50-95`. As perdas devem cair e o `mAP50` subir e estabilizar. Se a sessão cair no meio, retome com `YOLO('.../weights/last.pt')` e `resume=True`.

### 1.7 Olhar as curvas

```python
from IPython.display import Image, display
display(Image(f'{RAIZ}/modelos/detector/results.png', width=1000))
```

**Entregável do Dia 1:** `01_dados_e_detector.ipynb` · `modelos/detector/weights/best.pt` · `inventario.csv` · `amostras.png` · `results.png`

---

# DIA 2 — Avaliar a detecção e recortar as placas

**Objetivo:** números confiáveis sobre o detector e uma pasta com todas as placas recortadas, prontas para o Dia 3.

### 2.1 Métricas no conjunto de teste

```python
from ultralytics import YOLO

det = YOLO(f'{RAIZ}/modelos/detector/weights/best.pt')
m = det.val(data=f'{DADOS}/data.yaml', split='test', plots=True)

metricas = {
    'mAP@0.5':      round(float(m.box.map50), 4),
    'mAP@0.5:0.95': round(float(m.box.map),   4),
    'Precisão':     round(float(m.box.mp),    4),
    'Recall':       round(float(m.box.mr),    4),
}
for k, v in metricas.items():
    print(f'{k:>14}: {v}')

pd.DataFrame([metricas]).to_csv(f'{RAIZ}/resultados/tabelas/metricas_deteccao.csv', index=False)
```

**Leitura:** `mAP@0.5:0.95` é sempre menor que `mAP@0.5` porque exige sobreposição cada vez mais justa. Se o `mAP@0.5` passar de 0,90, o detector está bom o bastante — siga em frente.

### 2.2 IoU implementado do zero

Vale ter a função própria: além de aparecer no relatório, ela é usada na análise de erros abaixo.

```python
def iou(a, b):
    """IoU entre duas caixas no formato (x1, y1, x2, y2), em pixels."""
    xi1, yi1 = max(a[0], b[0]), max(a[1], b[1])
    xi2, yi2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    uniao = area_a + area_b - inter
    return inter / uniao if uniao > 0 else 0.0

# Testes rápidos de sanidade
print(iou((0, 0, 10, 10), (0, 0, 10, 10)))    # 1.0  — caixas idênticas
print(iou((0, 0, 10, 10), (20, 20, 30, 30)))  # 0.0  — caixas disjuntas
print(round(iou((0, 0, 10, 10), (5, 0, 15, 10)), 4))  # 0.3333 — metade sobreposta
```

### 2.3 Efeito do limiar de NMS

```python
# O NMS remove caixas duplicadas do mesmo objeto. Limiar baixo remove mais.
exemplo = sorted(glob.glob(f'{DADOS}/test/images/*'))[0]
for limiar in [0.3, 0.5, 0.7, 0.9]:
    r = det.predict(exemplo, iou=limiar, conf=0.25, verbose=False)[0]
    print(f'NMS iou={limiar} → {len(r.boxes)} detecção(ões)')
```

### 2.4 Onde o modelo erra

```python
# Compara predição com anotação usando a nossa função iou() e separa os piores casos.
registros = []
for caminho in sorted(glob.glob(f'{DADOS}/test/images/*')):
    img = cv2.imread(caminho)
    h, w = img.shape[:2]
    rot = caminho.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
    reais = []
    if os.path.exists(rot):
        for l in open(rot):
            _, xc, yc, bw, bh = map(float, l.split()[:5])
            reais.append(((xc-bw/2)*w, (yc-bh/2)*h, (xc+bw/2)*w, (yc+bh/2)*h))
    if not reais:
        continue
    pred = det.predict(caminho, conf=0.25, verbose=False)[0].boxes
    caixas_pred = pred.xyxy.cpu().numpy() if len(pred) else []
    melhor = max([iou(r, p) for r in reais for p in caixas_pred], default=0.0)
    registros.append(dict(arquivo=os.path.basename(caminho), melhor_iou=round(melhor, 3),
                          brilho=round(float(img.mean()), 1)))

erros = pd.DataFrame(registros).sort_values('melhor_iou')
print(f'Placas não detectadas (IoU = 0): {(erros.melhor_iou == 0).sum()} de {len(erros)}')
print(erros.head(10))
erros.to_csv(f'{RAIZ}/resultados/tabelas/analise_erros.csv', index=False)
```

**O que fazer com isso:** peça ao agente para exibir as 6 piores imagens lado a lado. As causas costumam ser placa muito pequena, foto noturna ou ângulo extremo — e essa figura vai para o relatório.

### 2.5 Recortar todas as placas

Use as caixas **anotadas** (não as previstas): elas são exatas e garantem que o Dia 3 comece com material limpo.

```python
# Recorta cada placa anotada com uma margem de 8% e salva em pastas por partição.
def recortar_todas(split, margem=0.08):
    destino = f'{RAIZ}/dados/placas_recortadas/{split}'
    os.makedirs(destino, exist_ok=True)
    n = 0
    for caminho in sorted(glob.glob(f'{DADOS}/{split}/images/*')):
        img = cv2.imread(caminho)
        if img is None:
            continue
        h, w = img.shape[:2]
        rot = caminho.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
        if not os.path.exists(rot):
            continue
        base = os.path.splitext(os.path.basename(caminho))[0]
        for i, l in enumerate(open(rot)):
            partes = l.split()
            if len(partes) < 5:
                continue
            xc, yc, bw, bh = map(float, partes[1:5])
            mx, my = bw * margem, bh * margem
            x1 = max(0, int((xc - bw/2 - mx) * w)); y1 = max(0, int((yc - bh/2 - my) * h))
            x2 = min(w, int((xc + bw/2 + mx) * w)); y2 = min(h, int((yc + bh/2 + my) * h))
            recorte = img[y1:y2, x1:x2]
            if recorte.size and recorte.shape[0] > 10 and recorte.shape[1] > 30:
                cv2.imwrite(f'{destino}/{base}_{i}.jpg', recorte)
                n += 1
    return n

for s in ['train', 'valid', 'test']:
    print(s, '→', recortar_todas(s), 'placas recortadas')
```

**Entregável do Dia 2:** `02_avaliacao_deteccao.ipynb` · `metricas_deteccao.csv` · `analise_erros.csv` · figura dos piores casos · pasta `placas_recortadas/`

---

# DIA 3 — Pré-processamento e dataset de caracteres

**Objetivo:** as funções de tratamento de imagem prontas e uma pasta `chars/` com dezenas de milhares de caracteres rotulados.

### 3.1 As funções de tratamento

```python
import cv2, numpy as np

def endireitar(placa_bgr):
    """Corrige a inclinação da placa usando o retângulo de área mínima."""
    cinza = cv2.cvtColor(placa_bgr, cv2.COLOR_BGR2GRAY)
    binaria = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(binaria > 0))
    if len(coords) < 20:
        return placa_bgr
    angulo = cv2.minAreaRect(coords[:, ::-1])[-1]
    if angulo > 45:
        angulo -= 90
    if abs(angulo) > 20:          # ângulo absurdo = medida ruim, não mexe
        return placa_bgr
    h, w = placa_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angulo, 1.0)
    return cv2.warpAffine(placa_bgr, M, (w, h),
                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def preparar(placa_bgr, tamanho=(200, 62)):
    """Redimensiona, converte para cinza, aplica CLAHE e binariza por Otsu."""
    p = cv2.resize(placa_bgr, tamanho, interpolation=cv2.INTER_CUBIC)
    cinza = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    realcada = clahe.apply(cinza)
    binaria = cv2.threshold(realcada, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return p, cinza, realcada, binaria

def detectar_layout(placa_bgr):
    """Placa Mercosul tem tarja azul no topo; a antiga, não."""
    topo = placa_bgr[:max(1, int(placa_bgr.shape[0] * 0.32)), :]
    hsv = cv2.cvtColor(topo, cv2.COLOR_BGR2HSV)
    azul = float(((hsv[:, :, 0] >= 100) & (hsv[:, :, 0] <= 130) &
                  (hsv[:, :, 1] >= 80)).mean())
    return ('mercosul' if azul > 0.25 else 'antiga'), round(azul, 3)
```

**Por que CLAHE e não equalização global:** o CLAHE equaliza o contraste em blocos pequenos, então corrige uma placa parcialmente na sombra sem estourar a parte iluminada. A equalização global trataria a imagem inteira de uma vez e perderia essa região.

### 3.2 Ver o efeito do tratamento

```python
import matplotlib.pyplot as plt, glob

exemplo = cv2.imread(sorted(glob.glob(f'{RAIZ}/dados/placas_recortadas/test/*.jpg'))[0])
exemplo = endireitar(exemplo)
p, cinza, realcada, binaria = preparar(exemplo)
layout, score_azul = detectar_layout(p)

fig, eixos = plt.subplots(1, 4, figsize=(16, 3))
for ax, img, titulo in zip(eixos,
        [cv2.cvtColor(p, cv2.COLOR_BGR2RGB), cinza, realcada, binaria],
        ['recortada', 'escala de cinza', 'CLAHE', 'binarizada (Otsu)']):
    ax.imshow(img, cmap=None if img.ndim == 3 else 'gray')
    ax.set_title(titulo, fontsize=10); ax.axis('off')

plt.suptitle(f'Layout detectado: {layout} (score azul = {score_azul})')
plt.tight_layout()
plt.savefig(f'{RAIZ}/resultados/figuras/preprocessamento.png', dpi=140, bbox_inches='tight')
plt.show()
```

Esta figura é uma das melhores do relatório — mostra as quatro etapas do tratamento em uma linha.

### 3.3 Segmentar os 7 caracteres

A placa tem 7 caracteres em espaçamento regular. A tarja azul superior sai com um corte fixo, e a faixa restante é dividida em 7 fatias.

```python
def projecao_vertical(binaria, corte_superior=0.35):
    """Soma os pixels brancos de cada coluna — os vales são os espaços."""
    faixa = binaria[int(binaria.shape[0] * corte_superior):, :]
    return faixa, faixa.sum(axis=0) / 255.0

def segmentar(binaria, n=7, corte_superior=0.35, saida=(32, 32)):
    """Divide a faixa dos caracteres em n fatias e padroniza o tamanho."""
    faixa, _ = projecao_vertical(binaria, corte_superior)
    largura = faixa.shape[1] // n
    return [cv2.resize(faixa[:, i*largura:(i+1)*largura], saida,
                       interpolation=cv2.INTER_AREA) for i in range(n)]
```

Visualize antes de confiar:

```python
faixa, proj = projecao_vertical(binaria)
recortes = segmentar(binaria)

fig = plt.figure(figsize=(14, 5))
ax1 = plt.subplot(3, 1, 1); ax1.imshow(faixa, cmap='gray'); ax1.axis('off')
ax1.set_title('faixa dos caracteres')
ax2 = plt.subplot(3, 1, 2); ax2.plot(proj); ax2.set_xlim(0, len(proj))
ax2.set_title('projeção vertical — vales = espaços entre caracteres')
for i, r in enumerate(recortes):
    ax = plt.subplot(3, 7, 15 + i); ax.imshow(r, cmap='gray'); ax.axis('off')
plt.tight_layout()
plt.savefig(f'{RAIZ}/resultados/figuras/segmentacao.png', dpi=140, bbox_inches='tight')
plt.show()
```

**Se as fatias saírem tortas:** ajuste `corte_superior` (tente 0.30 ou 0.40) e confira de novo. Divisão em fatias iguais é aproximada de propósito — o pequeno desalinhamento funciona como variação natural e a CNN do Dia 4 aprende a tolerar.

### 3.4 Gerar o dataset de caracteres

Se o dataset do Roboflow tiver os **caracteres anotados** (37 classes), este é o caminho: cada caixa vira uma imagem já rotulada.

```python
# Recorta cada caractere anotado, aplica o mesmo tratamento e salva em chars/<CLASSE>/
CLASSES = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
nomes = cfg['names'] if isinstance(cfg['names'], list) else list(cfg['names'].values())

def gerar_chars(split, saida=(32, 32)):
    base_destino = f'{RAIZ}/dados/chars/{split}'
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    total = 0
    for caminho in sorted(glob.glob(f'{DADOS}/{split}/images/*')):
        img = cv2.imread(caminho)
        if img is None:
            continue
        h, w = img.shape[:2]
        rot = caminho.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
        if not os.path.exists(rot):
            continue
        nome_base = os.path.splitext(os.path.basename(caminho))[0]
        for i, l in enumerate(open(rot)):
            partes = l.split()
            if len(partes) < 5:
                continue
            classe = str(nomes[int(partes[0])]).upper()
            if classe not in CLASSES:      # ignora a classe "placa"
                continue
            xc, yc, bw, bh = map(float, partes[1:5])
            x1, y1 = max(0, int((xc-bw/2)*w)), max(0, int((yc-bh/2)*h))
            x2, y2 = min(w, int((xc+bw/2)*w)), min(h, int((yc+bh/2)*h))
            rec = img[y1:y2, x1:x2]
            if rec.size == 0 or rec.shape[0] < 8 or rec.shape[1] < 5:
                continue
            cinza = clahe.apply(cv2.cvtColor(rec, cv2.COLOR_BGR2GRAY))
            binaria = cv2.threshold(cinza, 0, 255,
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            destino = f'{base_destino}/{classe}'
            os.makedirs(destino, exist_ok=True)
            cv2.imwrite(f'{destino}/{nome_base}_{i}.png',
                        cv2.resize(binaria, saida, interpolation=cv2.INTER_AREA))
            total += 1
    return total

for s in ['train', 'valid', 'test']:
    print(s, '→', gerar_chars(s), 'caracteres')
```

**Confira o balanceamento:**

```python
contagem = {c: len(glob.glob(f'{RAIZ}/dados/chars/train/{c}/*'))
            for c in CLASSES if os.path.isdir(f'{RAIZ}/dados/chars/train/{c}')}
serie = pd.Series(contagem).sort_values()
print('Classes com menos exemplos:\n', serie.head(8))
print('\nTotal de caracteres de treino:', serie.sum())
serie.plot.bar(figsize=(14, 3), title='Exemplos por classe (treino)')
plt.tight_layout(); plt.show()
```

**Se o seu dataset só tiver a classe "placa"** (sem caracteres anotados): peça ao agente o bloco alternativo que usa `segmentar()` sobre as placas recortadas e pega o rótulo do nome do arquivo — funciona quando o texto da placa está no nome, como no RodoSol. Sem texto anotado em nenhum lugar, a saída é trocar para o dataset do Roboflow indicado na preparação.

**Entregável do Dia 3:** `03_preprocessamento.ipynb` · funções `endireitar`, `preparar`, `detectar_layout`, `segmentar` · pasta `chars/` · figuras `preprocessamento.png` e `segmentacao.png`
---

# DIA 4 — A CNN de caracteres

**Objetivo:** um classificador de 36 classes com acurácia acima de 95% e a matriz de confusão que revela os pares problemáticos.

### 4.1 Carregar os dados

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

TAM, LOTE = (32, 32), 128

def carregar(split, embaralhar):
    return keras.utils.image_dataset_from_directory(
        f'{RAIZ}/dados/chars/{split}',
        labels='inferred', label_mode='int', color_mode='grayscale',
        image_size=TAM, batch_size=LOTE, shuffle=embaralhar, seed=42)

treino = carregar('train', True)
val    = carregar('valid', False)
teste  = carregar('test',  False)

CLASSES_MODELO = treino.class_names
print(f'{len(CLASSES_MODELO)} classes:', CLASSES_MODELO)

AUTO = tf.data.AUTOTUNE
treino = treino.cache().prefetch(AUTO)
val    = val.cache().prefetch(AUTO)
teste  = teste.cache().prefetch(AUTO)
```

**Saída esperada:** 36 classes e o número de imagens de cada partição. Se aparecerem menos de 36, alguma classe ficou sem exemplos — normal para letras raras; o agente deve avisar quais.

### 4.2 Montar e treinar a rede

Arquitetura pequena de propósito: caracteres binarizados de 32×32 são um problema simples, e rede grande aqui só traz tempo de treino e overfitting.

```python
aumentacao = keras.Sequential([
    layers.RandomTranslation(0.08, 0.08),   # tolera recorte deslocado
    layers.RandomZoom(0.08),
    layers.RandomRotation(0.02),
], name='aumentacao')

modelo_cnn = keras.Sequential([
    layers.Input(shape=(32, 32, 1)),
    layers.Rescaling(1./255),               # normaliza para [0, 1]
    aumentacao,
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(2),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(2),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(CLASSES_MODELO), activation='softmax'),
], name='cnn_caracteres')

modelo_cnn.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
modelo_cnn.summary()
```

```python
CAMINHO_CNN = f'{RAIZ}/modelos/cnn_chars.keras'
callbacks = [
    keras.callbacks.ModelCheckpoint(CAMINHO_CNN, save_best_only=True,
                                    monitor='val_accuracy'),
    keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True,
                                  monitor='val_accuracy'),
    keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, monitor='val_loss'),
]

historico = modelo_cnn.fit(treino, validation_data=val, epochs=30, callbacks=callbacks)

perda, acc = modelo_cnn.evaluate(teste, verbose=0)
print(f'\nAcurácia no teste: {acc:.4f}')
```

**Saída esperada:** acurácia de teste acima de 0,95. Abaixo de 0,90, os suspeitos são segmentação ruim no Dia 3 ou classes muito desbalanceadas — cole o resultado para o agente diagnosticar.

### 4.3 Curvas de treino

```python
import matplotlib.pyplot as plt

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4))
a1.plot(historico.history['loss'], label='treino')
a1.plot(historico.history['val_loss'], label='validação')
a1.set_title('Perda'); a1.set_xlabel('época'); a1.legend(); a1.grid(alpha=.3)
a2.plot(historico.history['accuracy'], label='treino')
a2.plot(historico.history['val_accuracy'], label='validação')
a2.set_title('Acurácia'); a2.set_xlabel('época'); a2.legend(); a2.grid(alpha=.3)
plt.tight_layout()
plt.savefig(f'{RAIZ}/resultados/figuras/curvas_cnn.png', dpi=140, bbox_inches='tight')
plt.show()
```

**Como ler:** se a curva de validação descola da de treino e começa a subir, houve overfitting — mas o `EarlyStopping` já devolveu os melhores pesos.

### 4.4 Matriz de confusão e os pares problemáticos

```python
import numpy as np, pandas as pd, seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

y_real, y_prev = [], []
for lote_x, lote_y in teste:
    y_real.extend(lote_y.numpy())
    y_prev.extend(np.argmax(modelo_cnn.predict(lote_x, verbose=0), axis=1))

cm = confusion_matrix(y_real, y_prev)

plt.figure(figsize=(13, 11))
sns.heatmap(cm, xticklabels=CLASSES_MODELO, yticklabels=CLASSES_MODELO,
            cmap='Blues', square=True, cbar_kws={'shrink': .6})
plt.xlabel('previsto'); plt.ylabel('real'); plt.title('Matriz de confusão')
plt.tight_layout()
plt.savefig(f'{RAIZ}/resultados/figuras/matriz_confusao.png', dpi=140, bbox_inches='tight')
plt.show()

# Os 12 pares mais confundidos — esta lista é usada no Dia 5
pares = [(CLASSES_MODELO[i], CLASSES_MODELO[j], int(cm[i, j]))
         for i in range(len(cm)) for j in range(len(cm)) if i != j and cm[i, j] > 0]
pares.sort(key=lambda t: -t[2])
print('Pares mais confundidos (real → previsto):')
for real, prev, n in pares[:12]:
    print(f'  {real} → {prev}: {n}')
```

**Aposta antes de rodar:** O↔0, I↔1, S↔5, B↔8, Z↔2, G↔6. Se acertou, o Dia 5 fica trivial.

### 4.5 Comparativo com ResNet50 e ViT *(opcional — corte se estiver atrasado)*

```python
# ResNet50 pré-treinado: os caracteres precisam virar 3 canais e 96x96.
def preparar_rgb(ds):
    return ds.map(lambda x, y: (tf.image.resize(tf.image.grayscale_to_rgb(x), (96, 96)), y),
                  num_parallel_calls=AUTO).prefetch(AUTO)

base = keras.applications.ResNet50(weights='imagenet', include_top=False,
                                   input_shape=(96, 96, 3))
base.trainable = False

modelo_rn = keras.Sequential([
    layers.Input(shape=(96, 96, 3)),
    layers.Rescaling(1./127.5, offset=-1),
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(CLASSES_MODELO), activation='softmax'),
])
modelo_rn.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
modelo_rn.fit(preparar_rgb(treino), validation_data=preparar_rgb(val), epochs=8)

perda_rn, acc_rn = modelo_rn.evaluate(preparar_rgb(teste), verbose=0)
print(f'ResNet50 congelado — acurácia no teste: {acc_rn:.4f}')
```

Monte a tabela final com o que tiver:

```python
comparativo = pd.DataFrame([
    dict(modelo='CNN do zero',        acuracia=round(acc, 4),
         parametros=modelo_cnn.count_params()),
    dict(modelo='ResNet50 congelado', acuracia=round(acc_rn, 4),
         parametros=modelo_rn.count_params()),
])
print(comparativo)
comparativo.to_csv(f'{RAIZ}/resultados/tabelas/comparativo.csv', index=False)
```

**O que provavelmente vai acontecer:** a CNN pequena empata ou vence a ResNet50, com uma fração dos parâmetros. Caractere binarizado 32×32 é uma tarefa simples, e a ResNet foi pré-treinada em texturas de objetos naturais, não em glifos. Se confirmar, você tem uma ótima conclusão para o relatório: **mais parâmetros não é mais desempenho.**

**Entregável do Dia 4:** `04_cnn_caracteres.ipynb` · `modelos/cnn_chars.keras` · `curvas_cnn.png` · `matriz_confusao.png` · lista dos pares confundidos · `comparativo.csv` (se fez o opcional)

---

# DIA 5 — Pipeline fim a fim e a regra do formato

**Objetivo:** uma função que recebe a foto e devolve o texto da placa, com as métricas finais medidas.

### 5.1 A regra do formato

Sabendo o layout, cada posição só aceita letra **ou** só dígito. Isso corrige boa parte dos erros da CNN de graça.

| Layout | Formato | Máscara | Exemplo |
| --- | --- | --- | --- |
| Mercosul (carro) | 3 letras · 1 dígito · 1 letra · 2 dígitos | `LLLDLDD` | `ABC1D23` |
| Brasileiro antigo | 3 letras · 4 dígitos | `LLLDDDD` | `ABC1234` |

```python
MASCARAS = {'mercosul': 'LLLDLDD', 'antiga': 'LLLDDDD'}

# Derivadas da matriz de confusão do Dia 4 — ajuste conforme os seus pares
PARA_LETRA  = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A',
               '5': 'S', '6': 'G', '7': 'T', '8': 'B'}
PARA_DIGITO = {'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1', 'Z': '2',
               'A': '4', 'S': '5', 'G': '6', 'T': '7', 'B': '8'}

def aplicar_mascara(texto, layout):
    """Troca caracteres impossíveis para a posição pelo equivalente visual."""
    mascara = MASCARAS.get(layout, MASCARAS['mercosul'])
    saida = []
    for ch, tipo in zip(texto, mascara):
        if tipo == 'L' and ch.isdigit():
            saida.append(PARA_LETRA.get(ch, ch))
        elif tipo == 'D' and ch.isalpha():
            saida.append(PARA_DIGITO.get(ch, ch))
        else:
            saida.append(ch)
    return ''.join(saida)

# Teste rápido
print(aplicar_mascara('A8C1D23', 'mercosul'))   # 'ABC1D23' — o 8 na posição 2 vira B
print(aplicar_mascara('ABC1O23', 'antiga'))     # 'ABC1023' — o O na posição 5 vira 0
```

### 5.2 A função `ler_placa`

```python
from ultralytics import YOLO
from tensorflow import keras
import numpy as np, cv2

det = YOLO(f'{RAIZ}/modelos/detector/weights/best.pt')
cnn = keras.models.load_model(f'{RAIZ}/modelos/cnn_chars.keras')

def ler_placa(caminho_ou_img, conf=0.25, usar_mascara=True):
    """Recebe uma foto e devolve o texto da placa e as confianças."""
    img = cv2.imread(caminho_ou_img) if isinstance(caminho_ou_img, str) else caminho_ou_img
    if img is None:
        return {'status': 'erro', 'motivo': 'imagem não pôde ser lida'}

    # 1) detectar a placa
    pred = det.predict(img, conf=conf, verbose=False)[0].boxes
    if len(pred) == 0:
        return {'status': 'sem_placa'}
    i = int(np.argmax(pred.conf.cpu().numpy()))
    x1, y1, x2, y2 = pred.xyxy.cpu().numpy()[i].astype(int)
    conf_det = float(pred.conf.cpu().numpy()[i])

    # 2) recortar com margem e endireitar
    mx, my = int((x2-x1)*0.08), int((y2-y1)*0.08)
    recorte = img[max(0, y1-my):y2+my, max(0, x1-mx):x2+mx]
    if recorte.size == 0:
        return {'status': 'recorte_invalido'}
    recorte = endireitar(recorte)

    # 3) tratar e identificar o layout
    p, _, _, binaria = preparar(recorte)
    layout, _ = detectar_layout(p)

    # 4) segmentar e classificar
    fatias = segmentar(binaria)
    lote = np.stack(fatias).astype('float32')[..., None]
    probs = cnn.predict(lote, verbose=0)
    idx = probs.argmax(axis=1)
    confs = probs.max(axis=1)
    bruto = ''.join(CLASSES_MODELO[k] for k in idx)

    texto = aplicar_mascara(bruto, layout) if usar_mascara else bruto
    return {'status': 'ok', 'placa': texto, 'placa_bruta': bruto, 'layout': layout,
            'conf_deteccao': round(conf_det, 3),
            'conf_media': round(float(confs.mean()), 3),
            'conf_minima': round(float(confs.min()), 3),
            'bbox': [int(x1), int(y1), int(x2), int(y2)]}

# Teste em uma imagem
import glob
print(ler_placa(sorted(glob.glob(f'{DADOS}/test/images/*'))[0]))
```

### 5.3 Medir o sistema inteiro

Você precisa do texto verdadeiro de cada placa. Se o dataset traz o texto no nome do arquivo, o bloco abaixo já funciona; se não, monte à mão um CSV com 60 a 100 placas de teste — é trabalhoso, mas é o que dá credibilidade ao número final.

```python
import re, pandas as pd

def texto_verdadeiro(caminho):
    """Extrai a placa do nome do arquivo. Ajuste o padrão ao seu dataset."""
    m = re.search(r'([A-Z]{3}[0-9][A-Z0-9][0-9]{2})', os.path.basename(caminho).upper())
    return m.group(1) if m else None

registros = []
for caminho in sorted(glob.glob(f'{DADOS}/test/images/*')):
    real = texto_verdadeiro(caminho)
    if not real:
        continue
    r = ler_placa(caminho)
    if r['status'] != 'ok':
        registros.append(dict(arquivo=os.path.basename(caminho), real=real,
                              bruto='', final='', acertos=0, status=r['status']))
        continue
    acertos_final = sum(a == b for a, b in zip(real, r['placa']))
    registros.append(dict(arquivo=os.path.basename(caminho), real=real,
                          bruto=r['placa_bruta'], final=r['placa'],
                          acertos=acertos_final, status='ok'))

res = pd.DataFrame(registros)

def resumo(coluna):
    corretos = sum(sum(a == b for a, b in zip(r.real, getattr(r, coluna)))
                   for r in res.itertuples() if getattr(r, coluna))
    total_ch = 7 * len(res)
    placas_ok = (res[coluna] == res.real).sum()
    return corretos/total_ch, placas_ok/len(res)

acc_char_bruto, acc_placa_bruto = resumo('bruto')
acc_char_final, acc_placa_final = resumo('final')

tabela = pd.DataFrame([
    dict(versao='CNN sozinha',        acc_caractere=round(acc_char_bruto, 4),
         acc_placa=round(acc_placa_bruto, 4)),
    dict(versao='CNN + regra Mercosul', acc_caractere=round(acc_char_final, 4),
         acc_placa=round(acc_placa_final, 4)),
])
print(tabela)
print(f'\nPrevisão teórica de acurácia por placa: {acc_char_final**7:.4f}')
tabela.to_csv(f'{RAIZ}/resultados/tabelas/metricas_finais.csv', index=False)
res.to_csv(f'{RAIZ}/resultados/tabelas/predicoes_teste.csv', index=False)
```

**A frase da apresentação:** *"a CNN sozinha acertou X% das placas; a CNN mais uma regra de 15 linhas derivada da legislação acertou Y%."* Modelo mais conhecimento de domínio supera modelo sozinho.

### 5.4 Onde estão os erros

```python
# Qual das 7 posições erra mais?
posicoes = np.zeros(7)
for r in res[res.final != ''].itertuples():
    for i, (a, b) in enumerate(zip(r.real, r.final)):
        if a != b:
            posicoes[i] += 1

plt.figure(figsize=(7, 3))
plt.bar(range(1, 8), posicoes)
plt.xlabel('posição na placa'); plt.ylabel('nº de erros')
plt.title('Distribuição dos erros por posição')
plt.tight_layout()
plt.savefig(f'{RAIZ}/resultados/figuras/erros_por_posicao.png', dpi=140, bbox_inches='tight')
plt.show()
```

**Entregável do Dia 5:** `05_pipeline.ipynb` · `src/pipeline.py` com as funções · `metricas_finais.csv` · `predicoes_teste.csv` · `erros_por_posicao.png`

---

# DIA 6 — ONNX, API e latência

**Objetivo:** o sistema deixa de ser notebook e vira serviço.

### 6.1 Exportar para ONNX

```python
!pip -q install tf2onnx onnxruntime

# Detector YOLO
det.export(format='onnx', imgsz=640)
print('YOLO ONNX:', f'{RAIZ}/modelos/detector/weights/best.onnx')

# CNN de caracteres
import tf2onnx, tensorflow as tf
spec = (tf.TensorSpec((None, 32, 32, 1), tf.float32, name='entrada'),)
tf2onnx.convert.from_keras(cnn, input_signature=spec, opset=13,
                           output_path=f'{RAIZ}/modelos/cnn_chars.onnx')
print('CNN ONNX exportada')
```

### 6.2 Medir a latência

```python
import time, numpy as np, onnxruntime as ort

lote = np.random.rand(7, 32, 32, 1).astype('float32')

def cronometrar(fn, n=100):
    fn()                                   # aquecimento
    t = [ ]
    for _ in range(n):
        ini = time.perf_counter(); fn(); t.append((time.perf_counter()-ini)*1000)
    t = np.array(t)
    return dict(media=round(t.mean(), 2), p50=round(np.percentile(t, 50), 2),
                p95=round(np.percentile(t, 95), 2))

sessao = ort.InferenceSession(f'{RAIZ}/modelos/cnn_chars.onnx',
                              providers=['CPUExecutionProvider'])
nome_entrada = sessao.get_inputs()[0].name

lat = pd.DataFrame([
    dict(versao='Keras', **cronometrar(lambda: cnn.predict(lote, verbose=0))),
    dict(versao='ONNX',  **cronometrar(lambda: sessao.run(None, {nome_entrada: lote}))),
])
print(lat)
lat.to_csv(f'{RAIZ}/resultados/tabelas/latencia.csv', index=False)
```

**O que esperar:** o ONNX costuma ser bem mais rápido em CPU. Reporte média, p50 e p95 — a p95 é o que importa em produção, porque descreve o pior caso comum.

### 6.3 A API

Salve como `api/app.py`:

```python
"""API de leitura de placas — FastAPI."""
import io, time
import numpy as np, cv2
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from tensorflow import keras

app = FastAPI(title="ALPR Mercosul", docs_url="/")

DET = YOLO("modelos/detector_best.pt")
CNN = keras.models.load_model("modelos/cnn_chars.keras")
CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
LIMIAR_CONFIANCA = 0.70

# Cole aqui as funções endireitar, preparar, detectar_layout,
# segmentar e aplicar_mascara — as mesmas do notebook.

@app.get("/health_check")
def health_check():
    return {"status": "ok"}

@app.post("/read_plate")
async def read_plate(file: UploadFile = File(...)):
    inicio = time.perf_counter()
    dados = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(dados, cv2.IMREAD_COLOR)
    if img is None:
        return {"status": "erro", "motivo": "arquivo não é uma imagem válida"}

    pred = DET.predict(img, conf=0.25, verbose=False)[0].boxes
    if len(pred) == 0:
        return {"status": "sem_placa",
                "tempo_ms": round((time.perf_counter()-inicio)*1000, 1)}

    i = int(np.argmax(pred.conf.cpu().numpy()))
    x1, y1, x2, y2 = pred.xyxy.cpu().numpy()[i].astype(int)
    mx, my = int((x2-x1)*0.08), int((y2-y1)*0.08)
    recorte = endireitar(img[max(0, y1-my):y2+my, max(0, x1-mx):x2+mx])

    p, _, _, binaria = preparar(recorte)
    layout, _ = detectar_layout(p)
    lote = np.stack(segmentar(binaria)).astype("float32")[..., None]
    probs = CNN.predict(lote, verbose=0)
    texto = aplicar_mascara("".join(CLASSES[k] for k in probs.argmax(axis=1)), layout)
    conf_min = float(probs.max(axis=1).min())

    return {
        "status": "ok" if conf_min >= LIMIAR_CONFIANCA else "revisao_manual",
        "placa": texto if conf_min >= LIMIAR_CONFIANCA else None,
        "layout": layout,
        "confianca_media": round(float(probs.max(axis=1).mean()), 3),
        "confianca_minima": round(conf_min, 3),
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "tempo_ms": round((time.perf_counter()-inicio)*1000, 1),
    }
```

**O limiar de confiança é o detalhe que rende na apresentação:** abaixo dele o sistema devolve `revisao_manual` em vez de arriscar uma placa errada. Recusar-se a responder quando não se tem certeza é o comportamento correto em produção.

Para testar no Colab:

```python
!pip -q install fastapi uvicorn python-multipart pyngrok
# Suba a API em segundo plano e exponha com ngrok (precisa de token gratuito)
!nohup uvicorn app:app --host 0.0.0.0 --port 8000 &
from pyngrok import ngrok
print('Swagger em:', ngrok.connect(8000).public_url)
```

Alternativa mais simples: rode a API **no seu Mac** — é só inferência, não precisa de GPU. Baixe os modelos do Drive, `pip install -r requirements.txt`, `uvicorn app:app --reload` e abra <http://localhost:8000>.

### 6.4 Empacotar

`api/requirements.txt`:

```
fastapi==0.115.0
uvicorn==0.30.6
python-multipart==0.0.9
ultralytics==8.3.0
tensorflow==2.17.0
opencv-python-headless==4.10.0.84
numpy==1.26.4
```

`api/Dockerfile`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Entregável do Dia 6:** `api/` completa · modelos `.onnx` · `latencia.csv` · print do Swagger funcionando

---

# DIA 7 — Relatório e slides

**Objetivo:** transformar seis dias de trabalho em algo avaliável em 15 minutos.

### 7.1 Reunir os resultados

```python
# Junta todas as tabelas geradas num único arquivo, pronto para o relatório.
import glob, pandas as pd

for caminho in sorted(glob.glob(f'{RAIZ}/resultados/tabelas/*.csv')):
    print('\n===', os.path.basename(caminho), '===')
    print(pd.read_csv(caminho).to_markdown(index=False))
```

Cole a saída no relatório — sai em Markdown, já formatada.

### 7.2 Estrutura do relatório

1. **Introdução** — o problema, aplicações de ALPR, o formato das placas Mercosul
2. **Materiais e métodos** — dataset, o pipeline em 6 etapas com a figura, arquiteturas usadas
3. **Resultados**
   - métricas de detecção (`metricas_deteccao.csv`) + curvas do YOLO
   - figura do pré-processamento e da segmentação
   - curvas e matriz de confusão da CNN
   - comparativo de arquiteturas, se fez
   - **tabela do antes/depois da regra de formato** ← o destaque
   - latência Keras × ONNX
4. **Análise de erros** — piores detecções, pares confundidos, erros por posição
5. **Deploy** — arquitetura da API, limiar de confiança, exemplo de resposta JSON
6. **Limitações** — motos com placa em duas linhas ficaram de fora, condições noturnas, cota de GPU
7. **Conclusão e trabalhos futuros**
8. **Considerações de privacidade** — meia página; ver abaixo

### 7.3 Slides — 10 lâminas

| # | Conteúdo |
| --- | --- |
| 1 | Título e o problema em uma frase |
| 2 | O pipeline (a figura das 6 etapas) |
| 3 | Dataset: números e amostras |
| 4 | Detecção: mAP e curvas |
| 5 | Pré-processamento: a placa antes e depois |
| 6 | Segmentação: a projeção vertical e as 7 fatias |
| 7 | CNN: arquitetura e matriz de confusão |
| 8 | **A regra de formato: antes e depois** |
| 9 | API funcionando + latência |
| 10 | Limitações, conclusão e próximos passos |

### 7.4 Privacidade — parágrafo obrigatório

Placa veicular é dado que pode identificar indiretamente uma pessoa. Inclua uma seção curta dizendo que: os dados usados são públicos e licenciados para pesquisa acadêmica; nenhuma imagem de placa real fotografada por você foi publicada sem tratamento; e o sistema é um exercício acadêmico, sem finalidade de vigilância. Meia página bem escrita demonstra maturidade e costuma render pontos.

### 7.5 Limpeza final

```
□ Notebooks numerados, executados de ponta a ponta sem erro
□ README com instruções de reprodução e link para os modelos
□ .gitignore excluindo dados/ e modelos/ (arquivos grandes)
□ requirements.txt com versões fixas
□ Relatório em PDF e slides no repositório
```

---

# Anexos

## A. Estrutura do repositório

```
alpr-mercosul/
├── README.md
├── RELATORIO.pdf
├── requirements.txt
├── notebooks/
│   ├── 01_dados_e_detector.ipynb
│   ├── 02_avaliacao_deteccao.ipynb
│   ├── 03_preprocessamento.ipynb
│   ├── 04_cnn_caracteres.ipynb
│   ├── 05_pipeline.ipynb
│   └── 06_deploy.ipynb
├── src/
│   ├── preprocessamento.py    # endireitar, preparar, detectar_layout, segmentar
│   ├── validacao.py           # aplicar_mascara, MASCARAS, PARA_LETRA, PARA_DIGITO
│   └── pipeline.py            # ler_placa
├── api/
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
└── resultados/
    ├── figuras/
    └── tabelas/
```

## B. Riscos e o que fazer

| Risco | O que fazer |
| --- | --- |
| Dataset sem caracteres anotados | Trocar pelo dataset do Roboflow indicado na preparação, que tem 37 classes |
| Cota de GPU do Colab esgotou | Esperar a virada do dia; enquanto isso, escrever relatório e API (não precisam de GPU) |
| Sessão caiu no meio do treino | Retomar com `YOLO('.../weights/last.pt')` e `resume=True` |
| Segmentação saindo torta | Ajustar `corte_superior` entre 0.30 e 0.40; se persistir, pedir ao agente o segmentador por projeção com detecção de vales |
| Acurácia da CNN abaixo de 0,90 | Conferir o balanceamento das classes e a qualidade dos recortes do Dia 3 |
| Atraso acumulado | Cortar nesta ordem: comparativo (4.5) → Docker (6.4) → análise de erros (2.4) |

## C. Perguntas prováveis da banca

1. Por que a acurácia por placa é tão menor que a acurácia por caractere?
2. O que o CLAHE faz, e por que não usar equalização de histograma global?
3. Explique o IoU. Por que `mAP@0.5:0.95` é menor que `mAP@0.5`?
4. Por que endireitar a placa antes de classificar os caracteres?
5. Por que a CNN pequena empatou ou venceu a ResNet50 neste problema?
6. Como você garantiu que não houve vazamento entre treino e teste?
7. Qual o gargalo de latência do pipeline? Como mediu?
8. O que o ONNX resolve?
9. Que riscos de privacidade um sistema desses traz?
10. Se fosse para produção amanhã, o que faltaria?

## D. Checklist de entrega

```
□ mAP@0.5 e mAP@0.5:0.95 da detecção
□ Acurácia por caractere e por placa
□ Tabela antes/depois da regra de formato
□ Matriz de confusão + erros por posição
□ Figura do pré-processamento e da segmentação
□ Tabela de latência Keras × ONNX
□ API respondendo em /read_plate com Swagger
□ Relatório em PDF
□ 10 slides
□ Repositório público e limpo
```

---

*Roteiro express de 7 dias, derivado da versão completa de 9 dias. Se sobrar tempo, os blocos opcionais da versão longa (poda de pesos, CLIP zero-shot, ViT) são os primeiros candidatos a voltar.*
