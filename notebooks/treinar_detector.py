# Treino completo (40 epocas), com resiliencia a queda de sessao:
# - salva last.pt a cada epoca (save_period=1)
# - uma thread em background copia o checkpoint pro Drive a cada 3 min
# - ao reiniciar, se achar checkpoint no Drive, retoma o treino dali
#   em vez de comecar do zero (resume=True do ultralytics)
import os, shutil, threading, time
from ultralytics import YOLO

DADOS = "/content/dados/deteccao"
RAIZ = "/content/drive/MyDrive/alpr-mercosul"
CKPT_DRIVE_DIR = f"{RAIZ}/modelos/detector_checkpoint"
CKPT_LOCAL_DIR = "/content/modelos/detector"

os.makedirs(f"{CKPT_DRIVE_DIR}/weights", exist_ok=True)

def sync_loop():
    while True:
        time.sleep(180)  # a cada 3 minutos
        src = f"{CKPT_LOCAL_DIR}/weights/last.pt"
        if os.path.exists(src):
            try:
                shutil.copy(src, f"{CKPT_DRIVE_DIR}/weights/last.pt")
                print("[sync] checkpoint copiado para o Drive", flush=True)
            except Exception as e:
                print(f"[sync] erro ao copiar checkpoint: {e}", flush=True)

threading.Thread(target=sync_loop, daemon=True).start()

ckpt_drive_path = f"{CKPT_DRIVE_DIR}/weights/last.pt"
if os.path.exists(ckpt_drive_path):
    print("Checkpoint encontrado no Drive - retomando treino...", flush=True)
    local_ckpt = f"{CKPT_LOCAL_DIR}/weights/last.pt"
    os.makedirs(os.path.dirname(local_ckpt), exist_ok=True)
    shutil.copy(ckpt_drive_path, local_ckpt)
    try:
        modelo = YOLO(local_ckpt)
        resultados = modelo.train(resume=True)
    except Exception as e:
        print(f"resume=True falhou ({e}); continuando como fine-tune a partir do checkpoint.", flush=True)
        modelo = YOLO(local_ckpt)
        resultados = modelo.train(
            data=f"{DADOS}/data.yaml",
            epochs=40, imgsz=640, batch=16, seed=42,
            project="/content/modelos", name="detector", exist_ok=True,
            save_period=1, patience=10, plots=True,
        )
else:
    print("Nenhum checkpoint encontrado - comecando treino do zero...", flush=True)
    modelo = YOLO("yolo11n.pt")
    resultados = modelo.train(
        data=f"{DADOS}/data.yaml",
        epochs=40, imgsz=640, batch=16, seed=42,
        project="/content/modelos", name="detector", exist_ok=True,
        save_period=1, patience=10, plots=True,
    )

print("=== TREINO COMPLETO CONCLUIDO ===", flush=True)
shutil.copytree(CKPT_LOCAL_DIR, f"{RAIZ}/modelos/detector", dirs_exist_ok=True)
print("Copiado para o Drive (final):", f"{RAIZ}/modelos/detector", flush=True)
print("=== TUDO PRONTO ===", flush=True)
