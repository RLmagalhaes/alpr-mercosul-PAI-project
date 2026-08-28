# ============================================================
# Dia 1 - Ambiente, dados e detector treinado
# Roda no Colab via `colab exec -s dia1 -f notebooks/01_dados_e_detector.py`
# IMPORTANTE: rode `colab drivemount -s dia1` (no seu terminal, interativo)
# ANTES deste script — o Drive precisa estar montado em /content/drive.
#
# Estrategia de armazenamento:
# - RAIZ (Drive) guarda so o que precisa sobreviver entre sessoes:
#   modelos treinados e resultados (figuras/tabelas).
# - DADOS (disco local da VM, /content) guarda as imagens brutas.
#   Escrever milhares de arquivos pequenos direto no Drive e muito
#   lento (mount em rede) e pode travar o kernel — por isso os dados
#   ficam locais e sao baixados de novo a cada sessao (e rapido).
# ============================================================

# --- 1.1 Montar o ambiente ---------------------------------
# Cria a estrutura de pastas de resultados no Drive (persiste entre sessoes).
import os

RAIZ = '/content/drive/MyDrive/alpr-mercosul'
for p in ['modelos', 'resultados/figuras', 'resultados/tabelas']:
    os.makedirs(f'{RAIZ}/{p}', exist_ok=True)

!nvidia-smi -L
!pip -q install ultralytics
print('Raiz do projeto (Drive):', RAIZ)

# --- 1.2 Baixar o dataset de deteccao -----------------------
# Mesmo dataset baixado localmente na Fase 0 (trafficbr/vehicle-plate-color v2),
# agora direto no disco local da VM (rapido, nao trava o kernel).
!pip -q install roboflow
from roboflow import Roboflow

rf = Roboflow(api_key="LgH8VW8NaRGPfLvPXv95")
projeto = rf.workspace("trafficbr").project("vehicle-plate-color")
ds = projeto.version(2).download("yolov8", location="/content/dados/deteccao")

DADOS = "/content/dados/deteccao"
print(os.listdir(DADOS))

# --- 1.3 Corrigir o data.yaml --------------------------------
# O data.yaml do Roboflow vem com caminhos relativos que quebram
# fora da pasta onde foi baixado. Reescreve com caminhos absolutos.
import yaml

cfg = yaml.safe_load(open(f'{DADOS}/data.yaml'))
cfg['path']  = DADOS
cfg['train'] = 'train/images'
cfg['val']   = 'valid/images'
cfg['test']  = 'test/images'
yaml.safe_dump(cfg, open(f'{DADOS}/data.yaml', 'w'), allow_unicode=True)

print('Classes:', cfg['names'])
print('Total  :', cfg['nc'])

# --- 1.4 Inventario do dataset --------------------------------
# Percorre as tres particoes e monta uma tabela com tamanho, nro de caixas,
# area relativa da placa e brilho medio de cada imagem.
# Imprime um "heartbeat" a cada 2000 imagens para o comando nao parecer
# travado (o CLI do Colab desiste se ficar tempo demais sem nenhuma saida).
import glob, cv2, numpy as np, pandas as pd

def inventario(split):
    linhas = []
    caminhos = sorted(glob.glob(f'{DADOS}/{split}/images/*'))
    for i, caminho in enumerate(caminhos):
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
        if (i + 1) % 2000 == 0:
            print(f'  {split}: {i + 1}/{len(caminhos)} imagens processadas')
    return pd.DataFrame(linhas)

df = pd.concat([inventario(s) for s in ['train', 'valid', 'test']], ignore_index=True)

resumo = df.groupby('split').agg(imagens=('arquivo', 'count'),
                                 caixas=('n_caixas', 'sum'),
                                 area_media_pct=('area_pct', 'mean'),
                                 brilho_medio=('brilho', 'mean')).round(2)
print(resumo)
df.to_csv(f'{RAIZ}/resultados/tabelas/inventario.csv', index=False)
print('Salvo em:', f'{RAIZ}/resultados/tabelas/inventario.csv')


# --- 1.5 Galeria de amostras ----------------------------------
# Mostra 6 fotos de treino com a caixa da placa desenhada, pra
# checar visualmente se as anotacoes estao corretas.
# Salva local (rapido) e so entao copia pro Drive - evita operacoes
# lentas de mais sem nenhuma saida no meio (o CLI desiste de esperar).
print('Gerando galeria de amostras...')
import matplotlib.pyplot as plt
import shutil

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
print('Salvando figura local...')
plt.savefig('/content/amostras.png', dpi=120, bbox_inches='tight')
print('Copiando para o Drive...')
shutil.copy('/content/amostras.png', f'{RAIZ}/resultados/figuras/amostras.png')
print('Pronto:', f'{RAIZ}/resultados/figuras/amostras.png')
