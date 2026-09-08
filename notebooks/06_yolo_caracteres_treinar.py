"""Dia 6 - Treino completo do YOLO detector de caracteres.

POR QUE (ver DIARIO, Dia 6): a segmentacao classica por componentes conectados
nao isola os caracteres nestas imagens -- eles se encostam entre si e na
moldura da placa depois da binarizacao. Medido nas 14 placas de teste:
exatamente 7 blobs em so 4/14 casos (200x62) e 8/14 no melhor caso (600x186).
E teto do metodo, nao parametro mal ajustado. Este YOLO substitui segmentar().

Roda o treino em SUBPROCESSO em background e devolve o controle na hora, pra
`colab exec` nao estourar timeout (mesma estrategia do Dia 1). Progresso em
/content/log_chars.txt; acompanhe com 06_yolo_caracteres_status.py.

Resiliencia a queda de sessao (o Colab gratuito caiu varias vezes no Dia 1):
- save_period=1 salva last.pt a cada epoca
- uma thread copia last.pt pro Drive a cada 3 min
- ao reiniciar, se achar checkpoint no Drive, retoma com resume=True

Uso:
    colab exec -s dia6 -f notebooks/06_yolo_caracteres_treinar.py --timeout 120
"""

import subprocess
import sys

EPOCAS = 40

treino = f"""
import os, shutil, threading, time
from ultralytics import YOLO

DADOS = '/content/dados/caracteres'
RAIZ = '/content/drive/MyDrive/alpr-mercosul'
CKPT_DRIVE = f'{{RAIZ}}/modelos/chars_checkpoint'
LOCAL = '/content/modelos/chars'

os.makedirs(f'{{CKPT_DRIVE}}/weights', exist_ok=True)


def sync_loop():
    while True:
        time.sleep(180)
        src = f'{{LOCAL}}/weights/last.pt'
        if os.path.exists(src):
            try:
                shutil.copy(src, f'{{CKPT_DRIVE}}/weights/last.pt')
                print('[sync] checkpoint copiado pro Drive', flush=True)
            except Exception as e:
                print(f'[sync] erro: {{e}}', flush=True)


threading.Thread(target=sync_loop, daemon=True).start()

TREINO = dict(data=f'{{DADOS}}/data.yaml', epochs={EPOCAS}, imgsz=640,
              batch=16, seed=42, project='/content/modelos', name='chars',
              exist_ok=True, save_period=1, patience=10, plots=True)

ckpt_drive = f'{{CKPT_DRIVE}}/weights/last.pt'
ckpt_local = f'{{LOCAL}}/weights/last.pt'

if os.path.exists(ckpt_local) or os.path.exists(ckpt_drive):
    if not os.path.exists(ckpt_local) and os.path.exists(ckpt_drive):
        os.makedirs(os.path.dirname(ckpt_local), exist_ok=True)
        shutil.copy(ckpt_drive, ckpt_local)
        print('Checkpoint recuperado do Drive.', flush=True)
    print('Retomando treino...', flush=True)
    try:
        YOLO(ckpt_local).train(resume=True)
    except Exception as e:
        print(f'resume falhou ({{e}}); seguindo como fine-tune', flush=True)
        YOLO(ckpt_local).train(**TREINO)
else:
    print('Comecando do zero...', flush=True)
    YOLO('yolo11n.pt').train(**TREINO)

print('=== TREINO DE CARACTERES CONCLUIDO ===', flush=True)
shutil.copytree(LOCAL, f'{{RAIZ}}/modelos/chars', dirs_exist_ok=True)
print('Copiado pro Drive:', f'{{RAIZ}}/modelos/chars', flush=True)
print('=== TUDO PRONTO ===', flush=True)
"""

open("/content/treinar_chars.py", "w").write(treino)

log = open("/content/log_chars.txt", "w")
subprocess.Popen([sys.executable, "/content/treinar_chars.py"],
                 stdout=log, stderr=subprocess.STDOUT)

print("Treino disparado em background.")
print(f"Epocas: {EPOCAS}. Log em /content/log_chars.txt")
print("Acompanhe: colab exec -s dia6 -f notebooks/06_yolo_caracteres_status.py")
