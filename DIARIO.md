# Diário do projeto — ALPR Mercosul

> Atualizado ao fim de **toda** sessão pelo agente. É a memória do projeto entre os dias.
> Ao iniciar uma sessão, leia este arquivo antes de qualquer outra coisa.

**Prazo de entrega:** _(preencher a data)_
**Onde parei:** Dia 0 concluído — estrutura criada, projeto ainda não iniciado.

---

## Painel de métricas

Preencher conforme os números forem saindo. Estes são os valores que vão para o relatório.

| Métrica | Meta | Obtido | Dia |
| --- | --- | --- | --- |
| mAP@0.5 (detecção) | > 0,90 | — | 1–2 |
| mAP@0.5:0.95 | — | — | 2 |
| Acurácia por caractere (CNN, teste) | > 0,95 | — | 4 |
| Acurácia por placa — CNN sozinha | — | — | 5 |
| Acurácia por placa — CNN + regra | > 0,80 | — | 5 |
| Latência Keras (p95) | — | — | 6 |
| Latência ONNX (p95) | — | — | 6 |

---

## Dia 0 — Preparação ✅

**Feito**

- Repositório criado com a estrutura de pastas e os módulos base em `src/`.
- `CLAUDE.md` e `DIARIO.md` criados.
- Roteiro completo em `docs/ROTEIRO_ALPR_7_DIAS.md`.

**Pendências antes do Dia 1**

- [ ] Testar a Colab CLI na conta gratuita: `pip install google-colab-cli && colab new --gpu t4`
- [ ] Baixar o dataset do Roboflow **com os caracteres anotados** ([License Plate Recognition](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e))
- [ ] Solicitar o RodoSol-ALPR (liberação leva de 1 a 5 dias úteis)
- [ ] `git init` e primeiro commit
- [ ] Rodar `pytest tests/` e ver os testes passando

**Próximo passo:** Dia 1 — ambiente, dados e detector treinado.

---

## Dia 1 — Ambiente, dados e detector treinado

**Objetivo:** um modelo YOLO que encontra placas em fotos.

**O que foi feito**
_(preencher)_

**Métricas obtidas**
_(preencher — nº de imagens por partição, área média da placa em % da imagem, mAP do treino)_

**Decisões**
_(preencher — qual dataset, qual modelo base, quantas épocas e por quê)_

**Pendências**
_(preencher)_

**Próximo passo**
_(preencher)_

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
| | | |
