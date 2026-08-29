# Diário do projeto — ALPR Mercosul

> Atualizado ao fim de **toda** sessão pelo agente. É a memória do projeto entre os dias.
> Ao iniciar uma sessão, leia este arquivo antes de qualquer outra coisa.

**Prazo de entrega:** _(preencher a data)_
**Onde parei:** Dia 1 concluído — detector YOLO11n treinado (40 épocas completas via checkpoint + resume no Drive). Métricas finais: mAP50 = 0,994 · mAP50-95 = 0,851 · precisão = 0,987 · recall = 0,979. Pesos em `modelos/detector` (Drive). Pronto pra começar o Dia 2 (avaliação da detecção e recorte das placas).

---

## Painel de métricas

Preencher conforme os números forem saindo. Estes são os valores que vão para o relatório.

| Métrica | Meta | Obtido | Dia |
| --- | --- | --- | --- |
| mAP@0.5 (detecção) | > 0,90 | 0,994 (val, no treino — falta medir no test no Dia 2) | 1 |
| mAP@0.5:0.95 | — | 0,851 (val, no treino — falta medir no test no Dia 2) | 1 |
| Acurácia por caractere (CNN, teste) | > 0,95 | — | 4 |
| Acurácia por placa — CNN sozinha | — | — | 5 |
| Acurácia por placa — CNN + regra | > 0,80 | — | 5 |
| Latência Keras (p95) | — | — | 6 |
| Latência ONNX (p95) | — | — | 6 |

---

## Dia 0 — Preparação ✅

**Datasets**

- Caracteres: project-swcsj/license-plate-character-extraction v2 (Roboflow)
  - 36 classes (0-9, A-Z), 4711/144/36 imagens, 640x640 px
  - Mediana de 7 caixas por imagem = recortes de placa com caracteres anotados
  - ~33 mil caracteres rotulados no total
  - PENDÊNCIA: redividir 70/15/15 POR IMAGEM (split original é 97/3/0.7)
  - ATENÇÃO: imagens em 640x640, verificar se os caracteres estão deformados

- Detecção: trafficbr/vehicle-plate-color v2 (Roboflow) — 12780/960/257 imagens, classe única "plate", fotos de veículo inteiro (placas Mercosul BR). Salvo em `dados/deteccao/vehicle-plates/`.

**Feito**

- Repositório criado com a estrutura de pastas e os módulos base em `src/`.
- `CLAUDE.md` e `DIARIO.md` criados.
- Roteiro completo em `docs/ROTEIRO_ALPR_7_DIAS.md`.

**Pendências antes do Dia 1**

- [x] Colab CLI testada (`uv tool install google-colab-cli` + `colab new --gpu t4`) — sessão T4 criada e parada com sucesso.
- [x] Dataset de caracteres baixado (ver `dados/caracteres/` acima).
- [x] RodoSol-ALPR: **decisão** — não solicitar, sem tempo pra esperar a liberação (1-5 dias úteis). Projeto segue com o dataset de detecção do Roboflow (`trafficbr/vehicle-plate-color`).
- [x] `git init` e primeiro commit feitos (`58e53e2`).
- [x] `pytest tests/` — 14 testes passaram.

**Próximo passo:** Dia 1 — ambiente, dados e detector treinado.

---

## Dia 1 — Ambiente, dados e detector treinado ✅

**Objetivo:** um modelo YOLO que encontra placas em fotos.

**O que foi feito**
- Ambiente Colab configurado via Colab CLI (`colab new -s dia1 --gpu T4`), Drive montado, ultralytics instalado.
- Dataset de deteccao baixado direto no Colab (trafficbr/vehicle-plate-color v2), salvo local na VM (`/content/dados/deteccao`) em vez do Drive — escrever milhares de arquivos pequenos direto no Drive trava o kernel (mount em rede).
- Inventario do dataset gerado (ver tabela abaixo) e galeria de 6 amostras com bbox conferida visualmente — anotacoes corretas.
- Ensaio de 3 epocas rodado com sucesso (script `notebooks/treinar_ensaio.py`, via subprocess em background pra nao travar o CLI em treinos longos).
- Treino completo (40 epocas, `yolo11n.pt`, `patience=10`) rodado via `notebooks/treinar_detector.py`, com checkpoint periodico (`last.pt` copiado pro Drive a cada 180s) e retomada automatica (`resume=True`) apos quedas de sessao — a sessao gratuita do Colab caiu varias vezes ao longo do treino, mas nenhum progresso foi perdido gracas ao checkpoint. Treino concluido nas 40 epocas configuradas, pesos finais copiados pra `modelos/detector` no Drive.

**Métricas obtidas**
- Inventario: train 12780 imagens/13386 caixas (area media da placa 5,73% da foto), valid 960/995 (4,97%), test 257/268 (5,86%).
- Ensaio (3 epocas, yolo11n.pt) — preliminar, so pra validar que o pipeline funciona:
  - epoca 3: mAP50 = 0,99 · mAP50-95 = 0,773 · precisao = 0,980 · recall = 0,954
- **Treino completo (40 epocas, yolo11n.pt) — resultado final:**
  - mAP50 = 0,994 · mAP50-95 = 0,851 · precisao = 0,987 · recall = 0,979
  - val/box_loss = 0,611 · val/cls_loss = 0,299 · val/dfl_loss = 0,860 (losses de val abaixo das de treino, sem sinal de overfitting nesse ponto)
  - LIMITACAO HONESTA: o disco local do Colab (`/content`) e apagado a cada sessao nova, entao o `results.csv`/`results.png` finais so guardam a ULTIMA sessao de treino (1 epoca, a de numero 40), nao a curva completa das 40 epocas. Os pesos sao cumulativos e reais (carregam o aprendizado de todas as sessoes via `resume=True`), so o historico visual da curva epoca-a-epoca ficou fragmentado entre sessoes. Nao afeta a validade do modelo final, so a rastreabilidade da evolucao.

**Decisões**
- Dataset de deteccao: `trafficbr/vehicle-plate-color` v2 (Roboflow) em vez do sugerido originalmente no roteiro — 1 classe ("plate"), fotos de veiculo inteiro, placas Mercosul BR.
- Modelo base: `yolo11n.pt` (nano), como no roteiro.
- Dados brutos (imagens) ficam no disco local da VM, nao no Drive — persistem so durante a sessao, mas o download do Roboflow e rapido o suficiente pra refazer a cada sessao nova. So os resultados (pesos, figuras, tabelas) vao pro Drive.
- RodoSol-ALPR descartado (ver Dia 0) — segue so com este dataset.
- Estrategia de resiliencia do treino: `save_period=1` + thread em background copiando `last.pt` pro Drive a cada 180s + deteccao automatica de checkpoint no inicio do script (retoma com `resume=True` se achar checkpoint no Drive) — necessario porque a sessao gratuita do Colab caiu repetidas vezes durante o treino de 40 epocas.

**Pendências**
- Nenhuma pendencia do Dia 1 — treino completo, pesos e curvas conferidos (ver limitacao do historico do `results.csv` acima).

**Próximo passo**
- Dia 2 — avaliar o detector no conjunto de TEST (nao so o val visto no treino) e recortar as placas detectadas, alimentando o dataset de caracteres do Dia 3.

---

## Dia 2 — Avaliação da detecção e recorte das placas

**Objetivo:** métricas confiáveis do detector e as placas recortadas para o Dia 3.

**O que foi feito**
_(preencher)_

**Métricas obtidas**
_(preencher — mAP@0.5, mAP@0.5:0.95, precisão, recall, nº de placas não detectadas)_

**Decisões**
_(preencher)_

**Pendências**
_(preencher)_

**Próximo passo**
_(preencher)_

---

## Dia 3 — Pré-processamento e dataset de caracteres

**Objetivo:** funções de tratamento prontas e a pasta `chars/` gerada.

**O que foi feito**
_(preencher)_

**Métricas obtidas**
_(preencher — total de caracteres por partição, classes mais raras, valor final de `corte_superior`)_

**Decisões**
_(preencher)_

**Pendências**
_(preencher)_

**Próximo passo**
_(preencher)_

---

## Dia 4 — CNN de caracteres

**Objetivo:** classificador de 36 classes com acurácia acima de 95%.

**O que foi feito**
_(preencher)_

**Métricas obtidas**
_(preencher — acurácia de teste, nº de parâmetros, épocas até parar)_

**Pares mais confundidos** — esta lista alimenta a regra do Dia 5
_(preencher: ex. O→0, I→1, S→5, B→8)_

**Decisões**
_(preencher)_

**Próximo passo**
_(preencher)_

---

## Dia 5 — Pipeline fim a fim e regra do formato

**Objetivo:** `ler_placa()` funcionando e as métricas finais medidas.

**O que foi feito**
_(preencher)_

**Métricas obtidas**
_(preencher — acurácia por caractere e por placa, antes e depois da regra; posição que mais erra)_

**Decisões**
_(preencher)_

**Próximo passo**
_(preencher)_

---

## Dia 6 — ONNX, API e latência

**Objetivo:** o sistema virando serviço.

**O que foi feito**
_(preencher)_

**Métricas obtidas**
_(preencher — latência média/p50/p95 em Keras e ONNX, limiar de confiança escolhido)_

**Decisões**
_(preencher)_

**Próximo passo**
_(preencher)_

---

## Dia 7 — Relatório e slides

**O que foi feito**
_(preencher)_

**Entregues**
- [ ] `RELATORIO.pdf`
- [ ] 10 slides
- [ ] Repositório limpo e público
- [ ] Seção de privacidade escrita

---

## Registro de problemas

Anote aqui o que quebrou e como foi resolvido. Vira a seção "Limitações" do relatório.

| Dia | Problema | Solução |
| --- | --- | --- |
| 1 | `colab exec` tem timeout padrão de 30s, causando `TimeoutError` em scripts mais demorados (matplotlib, downloads, treino) | Sempre passar `--timeout` explícito (600 pra scripts longos, 60 pra snippets rápidos) |
| 1 | `colab upload` falhava de forma inconsistente (`File or directory not found` no caminho remoto) mesmo com sintaxe correta | Escrever o conteúdo do script via stdin em `colab exec` (`open(caminho,'w').write(codigo)`) em vez de usar `colab upload` |
| 1 | Sessões gratuitas do Colab caem imprevisivelmente (perda de conexão, kernel perdido) durante o treino de 40 épocas — uma vez perdendo ~1h de progresso | Checkpoint periódico (`last.pt` copiado pro Drive a cada 180s) + detecção automática e retomada (`resume=True`) no início do script |
| 1 | Disco local do Colab é apagado a cada sessão nova, então `results.csv` só guarda a última sessão de treino, não o histórico completo das 40 épocas | Aceito como limitação conhecida — pesos finais são cumulativos e válidos, só o gráfico de evolução ficou fragmentado |
