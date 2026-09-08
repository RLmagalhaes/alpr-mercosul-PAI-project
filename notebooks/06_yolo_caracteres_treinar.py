"""Dia 6 - Treino completo do YOLO detector de caracteres.

Roda o treino em SUBPROCESSO em background e devolve o controle na hora, pra
`colab exec` nao estourar timeout (mesma estrategia do Dia 1). O progresso vai
pra /content/log_chars.txt; acompanhe com 06_yolo_caracteres_status.py.

NAO depende do Drive: checkpoints e pesos ficam no disco local da VM. Se a
sessao cair, o treino recomeca do `last.pt` local -- e por isso o numero de
epocas e menor que o do detector de placas (deteccao de caractere em recorte
de placa e uma tarefa bem mais facil: o ensaio ja deve mostrar mAP alto).

Uso (o Raphael roda no terminal dele):
    colab exec -s dia6 -f notebooks/06_yolo_caracteres_treinar.py --timeout 120
"""

import subprocess
import sys

EPOCAS = 30

treino = f"""
import os
from ultralytics import YOLO

LOCAL = '/content/modelos/chars'
ckpt = f'{{LOCAL}}/weights/last.pt'

if os.path.exists(ckpt):
    print('Checkpoint local encontrado - retomando...', flush=True)
    try:
        m = YOLO(ckpt)
        m.train(resume=True)
    except Exception as e:
        print(f'resume falhou ({{e}}); refazendo como fine-tune', flush=True)
        m = YOLO(ckpt)
        m.train(data='/content/dados/caracteres/data.yaml', epochs={EPOCAS},
                imgsz=640, batch=16, seed=42, project='/content/modelos',
                name='chars', exist_ok=True, save_period=1, patience=8, plots=True)
else:
    print('Comecando do zero...', flush=True)
    m = YOLO('yolo11n.pt')
    m.train(data='/content/dados/caracteres/data.yaml', epochs={EPOCAS},
            imgsz=640, batch=16, seed=42, project='/content/modelos',
            name='chars', exist_ok=True, save_period=1, patience=8, plots=True)

print('=== TREINO DE CARACTERES CONCLUIDO ===', flush=True)
"""

open("/content/treinar_chars.py", "w").write(treino)

log = open("/content/log_chars.txt", "w")
subprocess.Popen([sys.executable, "/content/treinar_chars.py"],
                 stdout=log, stderr=subprocess.STDOUT)

print("Treino disparado em background.")
print(f"Epocas: {EPOCAS}. Log em /content/log_chars.txt")
print("Acompanhe com: colab exec -s dia6 -f notebooks/06_yolo_caracteres_status.py")
