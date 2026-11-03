# Ensaio de 3 epocas - roda em background na VM, log em /content/log_ensaio.txt
# Se der erro de caminho/formato, aparece em minutos em vez de nos 40 do treino completo.
from ultralytics import YOLO

DADOS = "/content/dados/deteccao"

ensaio = YOLO('yolo11n.pt')
ensaio.train(data=f'{DADOS}/data.yaml', epochs=3, imgsz=640, batch=16,
             project='/content/modelos', name='ensaio', exist_ok=True, verbose=True)
print("=== ENSAIO CONCLUIDO ===", flush=True)
