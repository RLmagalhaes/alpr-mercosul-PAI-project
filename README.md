# ALPR Mercosul — Deteccao e Reconhecimento de Placas Veiculares

Trabalho final da disciplina **Processamento e Análise de Imagens** (Pós-Graduação).
Sistema que recebe a foto de um veículo e devolve o texto da placa.

```
foto → [YOLO detecta a placa] → [recorte + endireitamento] → [cinza + CLAHE + Otsu]
     → [segmenta 7 caracteres] → [CNN classifica cada um] → [regra do formato] → "ABC1D23"
```

## Resultados

_(preencher ao longo do projeto — os números saem de `resultados/tabelas/`)_

| Métrica | Valor |
| --- | --- |
| mAP@0.5 (detecção) | — |
| mAP@0.5:0.95 | — |
| Acurácia por caractere | — |
| Acurácia por placa — sem a regra de formato | — |
| Acurácia por placa — com a regra de formato | — |
| Latência p95 (ONNX, CPU) | — |

## Como reproduzir

```bash
git clone <url-do-repo> && cd alpr-mercosul
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v                     # 14 testes das funções puras
```

Os modelos treinados (`.pt`, `.keras`, `.onnx`) não estão versionados por causa do tamanho.
Baixe em: _(link do Drive)_ e coloque em `modelos/`.

Para subir a API localmente:

```bash
cd api && uvicorn app:app --reload
# Swagger em http://localhost:8000
```

Exemplo de resposta de `POST /read_plate`:

```json
{
  "status": "ok",
  "placa": "ABC1D23",
  "placa_sem_regra": "A8C1D23",
  "layout": "mercosul",
  "conf_deteccao": 0.94,
  "conf_media": 0.97,
  "conf_minima": 0.81,
  "bbox": [412, 265, 508, 296],
  "tempo_ms": 143.2
}
```

Quando a confiança mínima fica abaixo do limiar, o serviço devolve
`status: revisao_manual` e **não** arrisca uma placa errada.

## Estrutura

```
├── CLAUDE.md              instruções do agente de IA
├── DIARIO.md              estado do projeto, dia a dia
├── docs/                  roteiro de execução
├── notebooks/             01_ a 06_, na ordem de execução
├── src/                   módulos reutilizáveis
├── api/                   FastAPI + Dockerfile
├── tests/                 testes das funções puras
└── resultados/            figuras e tabelas do relatório
```

## Dados

- **Treino/validação/teste:** dataset público do Roboflow Universe com placas e caracteres anotados.
- **Upgrade opcional:** [RodoSol-ALPR](https://github.com/raysonlaroca/rodosol-alpr-dataset) — 20 mil imagens brasileiras/Mercosul, liberado para pesquisa acadêmica.

## Privacidade

Placa veicular é dado que pode identificar indiretamente uma pessoa. Este projeto
é um exercício acadêmico, sem finalidade de vigilância. Foram usados apenas
conjuntos públicos licenciados para pesquisa, e nenhuma imagem de placa real
fotografada pelo autor foi publicada sem tratamento.

## Licença

Uso acadêmico.
