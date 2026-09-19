"""Dia 6 - Avalia o YOLO detector de caracteres nos splits valid e test,
gravando as metricas em CSV.

Ate agora o mAP do detector de caracteres so existia no log de treino
(0,871 no valid, epoca 25). Este script mede de novo, LOCAL e na CPU, e
inclui o split de TESTE, que nunca tinha sido avaliado.

O `data.yaml` do Roboflow vem com caminhos relativos que nao resolvem fora do
ambiente onde foi baixado (`../test/images`) e usa a chave `val`. Este script
reescreve um yaml proprio com caminhos absolutos antes de validar - e o mesmo
cuidado do item 1.3 do roteiro.

Uso:
    .venv/bin/python notebooks/09_avaliar_yolo_caracteres.py
"""

import os
import sys

import pandas as pd
import yaml

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

DADOS = f"{RAIZ}/dados/caracteres"
CHARS = f"{RAIZ}/modelos/chars_best.pt"
TAB = f"{RAIZ}/resultados/tabelas"


def yaml_corrigido() -> str:
    """Reescreve o data.yaml com caminhos absolutos e devolve o novo caminho."""
    cfg = yaml.safe_load(open(f"{DADOS}/data.yaml"))
    cfg["path"] = DADOS
    cfg["train"] = "train/images"
    cfg["val"] = "valid/images"
    cfg["test"] = "test/images"
    destino = f"{DADOS}/data_local.yaml"
    yaml.safe_dump(cfg, open(destino, "w"), allow_unicode=True)
    return destino


def main():
    from ultralytics import YOLO

    dados = yaml_corrigido()
    modelo = YOLO(CHARS)

    linhas = []
    for split, pasta in [("val", "valid"), ("test", "test")]:
        n_img = len(os.listdir(f"{DADOS}/{pasta}/images"))
        print(f"\n===== split={pasta} ({n_img} imagens) =====")
        m = modelo.val(data=dados, split=split, device="cpu",
                       verbose=False, plots=False,
                       project=f"{RAIZ}/runs", name=f"chars_{pasta}",
                       exist_ok=True)
        linhas.append(dict(
            split=pasta,
            n_imagens=n_img,
            mAP50=round(float(m.box.map50), 4),
            mAP50_95=round(float(m.box.map), 4),
            precisao=round(float(m.box.mp), 4),
            recall=round(float(m.box.mr), 4),
            classes_avaliadas=int(len(m.box.ap_class_index))))

    df = pd.DataFrame(linhas)
    df.to_csv(f"{TAB}/metricas_yolo_caracteres.csv", index=False)
    print(f"\n-> {TAB}/metricas_yolo_caracteres.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
