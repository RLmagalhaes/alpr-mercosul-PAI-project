"""Dia 6 - Prepara o dataset e roda o ensaio de 3 epocas do YOLO de caracteres.

POR QUE existe este script (ver DIARIO, Dia 6): a segmentacao classica por
componentes conectados nao consegue isolar os caracteres nestas imagens -- eles
se encostam uns nos outros e na moldura da placa depois da binarizacao. Medido
nas 14 placas de teste: so 4 de 14 produzem exatamente 7 blobs em 200x62, e
8 de 14 no melhor caso (600x186). O resto cai na cirurgia grosseira de
`_juntar_ou_dividir`. Nao e parametro mal ajustado, e o teto do metodo.

A saida: trocar `segmentar()` por um YOLO treinado para DETECTAR cada caractere.
O dataset `project-swcsj` ja anota cada caractere individualmente (31.718
caixas), entao os dados necessarios ja estao na mao -- e a infra e a mesma do
Dia 1, que ja funcionou.

Este script NAO depende do Drive montado: tudo fica no disco local da VM e os
pesos voltam pro Mac via `colab download`.

Uso (o Raphael roda no terminal dele):
    export ROBOFLOW_API_KEY=...
    colab exec -s dia6 -f notebooks/06_yolo_caracteres_preparar.py --timeout 900
"""

import os
import subprocess
import sys

DADOS = "/content/dados/caracteres"

# ---------------------------------------------------------------
# 6.1 Dataset de caracteres (o MESMO do Dia 3/4, agora usado como
#     dataset de DETECCAO em vez de fonte de recortes)
# ---------------------------------------------------------------
# A chave NAO fica no codigo: este repo vai ser publicado. Ela e lida da
# variavel de ambiente da VM ou de um arquivo criado na VM antes de rodar
# (ver o comando no DIARIO, Dia 6). O `export` feito no Mac nao chega aqui --
# este script roda na VM do Colab, que tem ambiente proprio.
chave = os.environ.get("ROBOFLOW_API_KEY")
if not chave and os.path.exists("/content/.roboflow_key"):
    chave = open("/content/.roboflow_key").read().strip()
if not chave:
    print("ERRO: chave do Roboflow nao encontrada na VM.")
    print("Rode antes, no terminal do Mac:")
    print("  echo 'open(\"/content/.roboflow_key\",\"w\").write(\"SUA_CHAVE\")'"
          " | colab exec -s dia6")
    sys.exit(1)

if not os.path.exists(f"{DADOS}/data.yaml"):
    from roboflow import Roboflow
    rf = Roboflow(api_key=chave)
    (rf.workspace("project-swcsj")
       .project("license-plate-character-extraction")
       .version(2)
       .download("yolov8", location=DADOS))
else:
    print("Dataset ja presente na VM, pulando download.")

# O Roboflow escreve caminhos relativos ("../train/images") que quebram
# dependendo de onde o processo roda. Mesma correcao do Dia 1.
import yaml  # noqa: E402

cfg = yaml.safe_load(open(f"{DADOS}/data.yaml"))
cfg["train"] = f"{DADOS}/train/images"
cfg["val"] = f"{DADOS}/valid/images"
cfg["test"] = f"{DADOS}/test/images"
yaml.safe_dump(cfg, open(f"{DADOS}/data.yaml", "w"))

nomes = cfg["names"] if isinstance(cfg["names"], list) else list(cfg["names"].values())
print(f"\nClasses ({cfg['nc']}): {nomes}")

import glob  # noqa: E402

for split in ["train", "valid", "test"]:
    imgs = glob.glob(f"{DADOS}/{split}/images/*")
    caixas = sum(len(open(r).readlines())
                 for r in glob.glob(f"{DADOS}/{split}/labels/*.txt"))
    print(f"  {split:6s}: {len(imgs):5d} imagens, {caixas:6d} caixas de caractere")

# ---------------------------------------------------------------
# 6.2 Ensaio de 3 epocas -- erro de caminho aparece em minutos,
#     nao nos 40 do treino completo (licao do Dia 1)
# ---------------------------------------------------------------
ensaio = """
from ultralytics import YOLO
m = YOLO('yolo11n.pt')
m.train(data='/content/dados/caracteres/data.yaml', epochs=3, imgsz=640,
        batch=16, seed=42, project='/content/modelos', name='chars_ensaio',
        exist_ok=True, verbose=True)
print('=== ENSAIO CONCLUIDO ===', flush=True)
"""
open("/content/ensaio_chars.py", "w").write(ensaio)

print("\n=== rodando ensaio de 3 epocas (alguns minutos) ===", flush=True)
r = subprocess.run([sys.executable, "/content/ensaio_chars.py"],
                   capture_output=True, text=True)
# so as ultimas linhas interessam: a tabela de metricas por epoca
print("\n".join(r.stdout.strip().splitlines()[-25:]))
if r.returncode != 0:
    print("\n--- STDERR ---")
    print("\n".join(r.stderr.strip().splitlines()[-25:]))
print("\n=== PREPARACAO CONCLUIDA ===", flush=True)
