"""API de leitura de placas — FastAPI.

Rodar localmente:
    uvicorn app:app --reload
Depois abra http://localhost:8000 (a documentação Swagger fica na raiz).

Os modelos são carregados uma única vez, na primeira requisição.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import LeitorDePlacas  # noqa: E402

app = FastAPI(
    title="ALPR Mercosul",
    description="Detecção e reconhecimento de placas veiculares brasileiras.",
    version="1.0.0",
    docs_url="/",
)

DETECTOR = "modelos/detector_best.pt"
CNN = "modelos/cnn_chars.keras"
leitor = LeitorDePlacas(DETECTOR, CNN)


@app.get("/health_check", tags=["infra"])
def health_check():
    """Verifica se o serviço está no ar."""
    return {"status": "ok"}


@app.post("/read_plate", tags=["alpr"])
async def read_plate(file: UploadFile = File(...)):
    """Recebe a foto de um veículo e devolve o texto da placa.

    Quando a confiança mínima entre os 7 caracteres fica abaixo do limiar, o
    serviço devolve `status: revisao_manual` e não arrisca uma placa errada.
    Recusar-se a responder sem certeza é o comportamento correto em produção.
    """
    inicio = time.perf_counter()

    dados = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(dados, cv2.IMREAD_COLOR)
    if img is None:
        return {"status": "erro", "motivo": "arquivo não é uma imagem válida"}

    resultado = leitor.ler(img)
    resultado["tempo_ms"] = round((time.perf_counter() - inicio) * 1000, 1)
    return resultado
