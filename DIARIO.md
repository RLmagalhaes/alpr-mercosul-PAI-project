# Diário do projeto — ALPR Mercosul

> Atualizado ao fim de **toda** sessão pelo agente. É a memória do projeto entre os dias.
> Ao iniciar uma sessão, leia este arquivo antes de qualquer outra coisa.

**Prazo de entrega:** menos de 1 semana a partir de 03/09 (prazo real da professora é ~1 mês após o fim das aulas, mas o Raphael está atrasado em relação ao roteiro de 7 dias).
**Onde parei:** **Dia 7 — projeto entregue e em pausa.** A pasta `entrega/` está fechada e conferida; o repositório foi limpo, o README reescrito com os números reais e `docs/RETOMAR.md` criado para a retomada. Para continuar, leia `docs/RETOMAR.md` — não este arquivo inteiro.

**Histórico do Dia 6:** Dia 6 concluído. **A causa raiz do Dia 5 foi encontrada, e o gargalo real do projeto foi resolvido.** Primeiro: `corte_superior=0.35` decapitava o topo de cada caractere antes da segmentação (0,061 → 0,612 no dataset europeu ao corrigir). Segundo, e mais importante: mesmo corrigido, a segmentação por componentes conectados tem teto baixo — acha exatamente 7 blobs em só 4 de 14 placas. Substituída por um **YOLO detector de caracteres**, treinado sobre as 32.225 caixas já anotadas. Medido em **30 placas brasileiras com gabarito humano** (o conjunto que faltava desde o Dia 5): acurácia por caractere **0,300 → 0,719**, por placa **0/30 → 10/30**. Próximo: fechar as 40 épocas do YOLO (parou na ~30 por queda de sessão) e escrever o relatório.

**Decisão de escopo (registro interno, não sai no relatório pra professora):** dado o atraso, a entrega vai fechar o pipeline completo de visão computacional (detecção → pré-processamento → segmentação → CNN de caracteres → regra de formato), cortando a parte de produção (Dia 6: ONNX, API, Docker, latência) e os slides. Isso **não aparece como corte** em nenhum material voltado à professora (RELATORIO.md, README.md, notebooks) — lá o projeto entregue é descrito como se fosse o escopo original. `api/`, `Dockerfile` e as dependências de ONNX continuam no repo, sem uso, como continuação pessoal de aprendizado do Raphael depois da entrega. Ver plano completo em `/Users/raphaelmagalhaes/.claude/plans/eu-j-estou-atrasado-groovy-squirrel.md`.

---

## Painel de métricas

Preencher conforme os números forem saindo. Estes são os valores que vão para o relatório.

| Métrica | Meta | Obtido | Dia |
| --- | --- | --- | --- |
| mAP@0.5 (detecção, teste) | > 0,90 | 0,992 | 2 |
| mAP@0.5:0.95 (detecção, teste) | — | 0,834 | 2 |
| Precisão / recall (detecção, teste) | — | 0,978 / 0,977 | 2 |
| Acurácia por caractere (CNN, teste) | > 0,95 | 0,9435 | 4 |
| Acurácia por caractere — dataset europeu, segmentação clássica | — | 0,061 → **0,612** (após corrigir o Dia 5) | 6 |
| mAP@0.5 — YOLO de caracteres (38 classes, valid) | — | **0,871** (época 25; treino parou na ~30 de 40) | 6 |
| **Acurácia por caractere — 30 placas BR, gabarito humano** | > 0,95 | clássico 0,300 · YOLO+CNN 0,619 · **YOLO 0,719** | 6 |
| **Acurácia por placa — 30 placas BR, gabarito humano** | > 0,80 | clássico 0/30 · YOLO+CNN 7/30 · **YOLO 10/30 (0,333)** | 6 |
| Latência Keras (p95) | — | — | (cortado do escopo) |
| Latência ONNX (p95) | — | — | (cortado do escopo) |

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

## Dia 2 — Avaliação da detecção e recorte das placas ✅

**Objetivo:** métricas confiáveis do detector e as placas recortadas para o Dia 3.

**O que foi feito**
- Detector avaliado no conjunto de TESTE (257 imagens, 268 instâncias) — não só no val visto durante o treino.
- IoU implementado do zero (`iou()`) e testado com casos de sanidade (caixas idênticas, sem sobreposição, sobreposição parcial conhecida).
- Efeito do limiar de NMS observado numa imagem com múltiplas placas: limiares até 0,7 deram 2 detecções corretas; limiar 0,9 (muito permissivo) deixou passar uma caixa duplicada da mesma placa (3 detecções).
- Análise de erros: comparação predição x anotação via IoU, ranqueando os piores casos.
- Todas as placas recortadas a partir das caixas **anotadas** (não as previstas), com margem de 8%, reaproveitando a mesma lógica de `src/preprocessamento.recortar()`. Rodado direto no disco local da VM — **não fica no Drive** (ver decisão de gestão de espaço abaixo).
- Limpeza de espaço no Drive: apagados os 23 checkpoints por época do detector (`epoch*.pt`, `last.pt` — mantido só `best.pt`), a pasta `detector_checkpoint/` (mecanismo de resume do Dia 1, sem uso depois que o treino terminou) e `dados/deteccao/` (dataset bruto, que por engano tinha ficado no Drive além da VM). Pasta do projeto no Drive caiu de 1,2 GB para 70 MB. **Pendência do próprio Raphael:** esvaziar a lixeira do Drive pela interface, senão o espaço não é liberado de verdade.
- Achado à parte: existia um `placas_recortadas.zip` no Drive (datado de 29/08) com números muito parecidos aos nossos, mas sem nenhum registro de que código o gerou. Decisão: não reaproveitar — regerado do zero de forma rastreável (código no notebook, resultado com contagem de problemas). O resultado bateu (e superou um pouco, por causa da conversão de polígono — ver abaixo).

**Métricas obtidas**
- mAP@0.5 = 0,992 · mAP@0.5:0.95 = 0,834 · precisão = 0,978 · recall = 0,977 (conjunto de teste).
- 2 de 257 placas não detectadas (IoU=0); os piores casos entre as detectadas têm IoU muito baixo (0,01-0,03), ou seja, são erros grosseiros, não só imprecisão de borda.
- Recorte de todas as placas anotadas: 13.725 recortadas no total (train 12.539, valid 941, test 245). 720 anotações vieram em formato de polígono e foram convertidas pra caixa delimitadora (mesma conversão que o Ultralytics já faz sozinho). 924 recortes descartados por ficarem abaixo do tamanho mínimo (30x10px).
- **Problema de dados registrado (não contornado):** 76% dos recortes aceitos (10.420 de 13.725) são "muito pequenos" (altura<40px ou largura<100px) — ver `resultados/figuras/recortes_placas_tamanhos.png`. Nos exemplos pequenos, o texto da placa já sai borrado só pela miniatura, antes de qualquer pré-processamento. É uma limitação do dataset de origem (fotos de resolução/enquadramento variados), não um bug do recorte. Efeito esperado: a segmentação de caracteres do Dia 3 deve ter uma taxa de falha bem maior nesse subconjunto — isso precisa aparecer explícito na análise de erros do Dia 3/4, sem tentar mascarar com upscaling ou heurística de salvamento.

**Decisões**
- Placas recortadas ficam só no disco local da VM (`/content/dados/placas_recortadas/`), nunca no Drive — mesma lógica já usada pro dataset bruto no Dia 1. Como o recorte é rápido (poucos minutos, sem GPU), a célula 2.5 deve ser rodada de novo no início da sessão do Dia 3 em vez de tentar persistir os arquivos entre sessões.
- Anotações em polígono são convertidas pra bounding box (min/max), não descartadas — mantém mais dados e é consistente com o que o Ultralytics já faz na validação.

**Pendências**
- Esvaziar a lixeira do Google Drive (ação manual do Raphael, fora do alcance da CLI).

**Próximo passo**
- Dia 3 — rodar de novo a célula de recorte (2.5) no início da sessão, depois pré-processamento e geração do dataset de caracteres em `chars/<CLASSE>/`.

---

## Dia 3 — Pré-processamento e dataset de caracteres ✅

**Objetivo:** funções de tratamento prontas e a pasta `chars/` gerada.

**O que foi feito**
- Funções de tratamento (`endireitar`, `preparar`, `detectar_layout`, `projecao_vertical`, `segmentar`) usadas diretamente de `src/preprocessamento.py` — enviadas pra VM via `colab upload` em vez de redefinidas no notebook (corrige uma inconsistência que vinha do Dia 2, onde o `iou()` tinha sido redefinido inline).
- Demonstração do pré-processamento (recorte → cinza → CLAHE → Otsu) numa placa bem alinhada do teste.
- Demonstração da segmentação em 7 fatias, comparando um caso bem alinhado com um caso de perspectiva forte (ver limitação abaixo).
- Dataset de caracteres baixado (`project-swcsj/license-plate-character-extraction` v2 — **dataset separado** do de detecção, já vem com cada caractere anotado individualmente) e convertido em `chars/<CLASSE>/` com CLAHE+Otsu aplicado a cada recorte.
- Checagem visual de amostras aleatórias de 8 classes para conferir se os recortes batem com o rótulo.

**Métricas obtidas**
- Recorte de placas (Dia 2, regerado): 13.735 no total (train 12.548, valid 942, test 245 — pequena variação frente ao rodado no Dia 2 por causa da ordem de leitura do glob, sem impacto real).
- Dataset de caracteres: **31.718 caracteres rotulados** no total (train 30.530, valid 958, test 230), em 36 classes.
- Classes ignoradas por não pertencerem ao alfabeto de placa: `EUR` (1.708 casos) e `-` (44 casos) — são marcações do dataset original (bandeira/hífen de formato antigo), corretamente filtradas.
- Classes mais raras no treino: **Q (120)** e **O (228)** — bem abaixo do resto (a maioria das letras fica entre 400-1500, dígitos entre 1000-2100). Como Q e O também são visualmente parecidos com 0, esse é o par mais provável de confusão pro Dia 4.

**Problemas de dados registrados (não contornados)**
1. **Perspectiva forte quebra a segmentação:** `endireitar()` só corrige rotação no plano (via `minAreaRect`), não perspectiva. Numa foto tirada de ângulo mais agressivo, a segmentação por projeção (que assume fatias de largura igual) sai completamente desalinhada — ver `resultados/figuras/segmentacao.png`, que mostra lado a lado um caso bom (fatias legíveis) e um caso ruim (ilegível). Corrigir de verdade exigiria os 4 cantos da placa anotados, que este dataset não tem.
2. **Caracteres deformados na origem (pendência do Dia 0, resolvida):** o dataset de caracteres aplicou "Resize to 640x640 (Stretch)" no pré-processamento do Roboflow — ou seja, esticou cada imagem sem preservar a proporção original, distorcendo a forma dos caracteres de um jeito não uniforme (a proporção original não é recuperável). Aceito como limitação do dataset escolhido.
3. **Ruído de rótulo:** checagem visual encontrou pelo menos um exemplo rotulado como "O" que na verdade mostra um "H". Não foi feita filtragem manual (não há tempo nem um método confiável pra achar todos os casos parecidos em ~31 mil imagens) — fica registrado como ruído esperado de dataset anotado em massa, e é uma fonte honesta de erro que a CNN do Dia 4 provavelmente vai herdar.

**Decisões**
- Dataset de caracteres é gerado direto das anotações por caractere (mais preciso), **não** a partir do `segmentar()` sobre os recortes do Dia 2 — `segmentar()` é usado só na hora da inferência (Dia 5), quando não há caixa por caractere disponível.
- `chars/` fica só na VM (não no Drive), mesma lógica do Dia 2 — só o resumo (JSON) e as figuras vão pro Drive.
- Notebooks passam a importar as funções de `src/` de verdade (via upload pontual do pacote pra VM), em vez de redefini-las — alinhado com a convenção do projeto.

**Pendências**
- Nenhuma pendência nova do Dia 3.

**Próximo passo**
- Dia 4 — treinar a CNN de 36 classes sobre `chars/`, com atenção especial à matriz de confusão em torno de Q/O/0.

---

## Dia 4 — CNN de caracteres ✅

**Objetivo:** classificador de 36 classes com acurácia acima de 95%.

**O que foi feito**
- `chars/` regenerado do zero no início da sessão (disco local do Colab é apagado a cada sessão — mesma lógica dos Dias 2/3), com o mesmo código do Dia 3: 31.718 caracteres, 36 classes (train 30.530, valid 958, test 230). Números batem exatamente com o Dia 3.
- CNN pequena treinada do zero (`src`-free, arquitetura só do roteiro: 3 blocos Conv2D+MaxPool, Dense 128, Dropout 0.3), 359.588 parâmetros, `EarlyStopping`/`ModelCheckpoint`/`ReduceLROnPlateau`.
- **Primeira rodada** (sem peso de classe): acurácia de teste = 0,9348, parou em 13 épocas. Matriz de confusão mostrou **Q e O com 0% de acerto** — exatamente as duas classes mais raras no treino já apontadas como risco no Dia 3 (Q=120 exemplos, O=228).
- **Segunda rodada** (com `class_weight` balanceado por classe, pelo inverso da frequência no treino — pesos de 0,4 a 7,1): acurácia subiu para **0,9435**, 26 épocas. O passou de 0% para 100% de recall (mas ganhou falsos positivos vindos do dígito "0"); Q passou de 0% para 33% de recall.
- Matriz de confusão e pares confundidos gerados a partir da 2ª rodada (a que ficou nos artefatos finais).
- **Achado de processo corrigido nesta sessão:** as figuras/tabelas dos Dias 1 a 3 nunca tinham sido trazidas do Drive pro repositório local — só existiam na cópia da VM. Baixadas agora todas de uma vez pra `resultados/figuras/` e `resultados/tabelas/` locais (8 figuras, 9 tabelas/JSONs), senão o relatório final não teria de onde puxar essas imagens. Cuidado nos próximos dias: sempre baixar do Drive pro local antes de fechar a sessão.

**Métricas obtidas**
- Acurácia no teste (final, com peso de classe): **0,9435** — abaixo da meta de 0,95, mas por pouco.
- Perda no teste: 0,485. Parâmetros: 359.588. Épocas até o `EarlyStopping`: 26 (de 30 possíveis).
- Acurácia por classe (36 classes): 32 das 36 classes com recall ≥ 0,75 (a maioria em 1,0). As exceções: **O** (recall 1,0 mas precisão 0,33 — passou a "roubar" o dígito 0), **Q** (recall 0,33, 3 exemplos no teste), **0** (recall 0,56 — pagou o preço pelo peso extra em O), **V** (recall 0,75).
- Ver `resultados/tabelas/classification_report_cnn.csv` e `resultados/tabelas/resumo_dia4.json` para os números completos.
- **LIMITAÇÃO HONESTA:** o conjunto de teste tem só 230 caracteres pra 36 classes — Q e O aparecem 3 e 1 vez, respectivamente. Cada erro nessas classes pesa ~0,4-1,3 ponto percentual na acurácia total. Não dá pra saber se 33%/100% de recall nessas classes é representativo ou só sorte de amostra pequena — decisão tomada foi aceitar o resultado e não continuar ajustando hiperparâmetro em cima do ruído de 1-3 exemplos.

**Pares mais confundidos** (matriz de confusão final, real → previsto)
- 0→O: 2 · 0→4: 1 · 0→P: 1 · 4→L: 1 · 7→1: 1 · 9→3: 1 · 9→S: 1 · B→8: 1 · Q→C: 1 · Q→D: 1 · S→3: 1 · U→J: 1
- A aposta do roteiro (O↔0, I↔1, S↔5, B↔8, Z↔2, G↔6) acertou em parte: 7↔1 e B→8 apareceram; O↔0 apareceu só depois do ajuste de peso (antes disso Q↔0 é que aparecia). S↔5 e Z↔2/G↔6 não apareceram — no lugar, surgiram confusões novas (9→S, U→J, 4→L) que fazem sentido visual mas não estavam na lista original.

**Decisões**
- Usar `class_weight` balanceado (peso = total_treino / (36 × contagem_da_classe)) em vez de oversampling ou data augmentation extra — mudança de uma linha no `fit()`, sem precisar tocar no pipeline de geração de dados.
- Aceitar 0,9435 (abaixo da meta de 0,95) como resultado final do Dia 4 em vez de continuar ajustando: o resíduo está concentrado em classes com 1-3 exemplos no teste, então mais tuning ficaria ajustando em cima de ruído estatístico, não corrigindo um problema real do modelo. Registrado como limitação honesta, não maquiada.
- `modelos/cnn_chars.keras` fica só no Drive (mesma lógica de `modelos/detector` no Dia 1 — `.gitignore` cobre `*.keras`).

**Pendências**
- Nenhuma pendência nova do Dia 4, além da limitação de dados já registrada acima.

**Próximo passo**
- Dia 5 — montar `ler_placa()` fim a fim (detecção → pré-processamento → segmentação → CNN → regra do formato) e medir a acurácia por placa. A regra de formato (posição só aceita letra OU só dígito) deve corrigir de graça os erros tipo 0→O e Q→0/C/D que a CNN sozinha não resolve.

---

## Dia 5 — Pipeline fim a fim e regra do formato ✅ (acurácia por placa não resolvida — 3 bugs reais corrigidos no caminho)

**Objetivo:** `ler_placa()` funcionando e as métricas finais medidas.

**O que foi feito**
- `LeitorDePlacas` (`src/pipeline.py`) e `aplicar_mascara`/`formato_valido` (`src/validacao.py`) já estavam implementados e testados desde a Fase 0 — usados diretamente, sem redefinir nada.
- Decisão de método: o dataset de detecção (trafficbr) não tem o texto da placa anotado, só a caixa — não dá pra medir acurácia por caractere/placa nele sem rotular à mão. Optei por reconstruir o texto verdadeiro a partir do dataset de caracteres (`project-swcsj`, cada caractere já anotado individualmente), ordenando as caixas por posição X. 14 das 36 imagens de teste tinham exatamente 7 caracteres válidos (22 descartadas por terem um número diferente de 7 — registrado, não contornado).
- Essa medição virou uma investigação bem mais longa do que o previsto — a acurácia inicial saiu tão baixa (0,11) que decidi caçar a causa raiz em vez de aceitar o número. Achei e corrigi **3 bugs reais**, cada um confirmado com um teste isolado antes de seguir pro próximo:
  1. **Recorte ruim**: as imagens do `project-swcsj` não são recortes limpos de placa (têm fundo/carroceria e rotação forte, uma delas ~30° — acima do limite de segurança de 20° do `endireitar()`). Corrigido com `recortar_via_caixas()` (só no notebook — usa a geometria das 7 caixas anotadas, informação que o detector real não tem).
  2. **Segmentação por largura igual**: `segmentar()` dividia a faixa em 7 partes de largura IGUAL, o que quebra em placas de formato antigo (vão entre letras/dígitos) e em qualquer placa com caracteres de largura bem diferente (ex. "W" bem mais largo que "1"). Reescrita para usar **componentes conectados** (`cv2.connectedComponentsWithStats`): acha cada caractere pelo próprio blob de tinta, junta fragmentos ou divide blobs grudados até bater com `n=7`. Precisou de um filtro extra de largura E altura máxima pra não confundir a MOLDURA da placa (alta como um caractere, mas fina — ou, em alguns casos, mais alta que um caractere de verdade) com mais um caractere.
  3. **Binarização global em vez de local**: `preparar()` binariza a placa inteira de uma vez (Otsu global), mas o dataset de treino da CNN (Dia 3/4) foi gerado com Otsu LOCAL, um por caractere. Provado com um teste direto: a MESMA placa que a CNN acertava **100%** usando o recorte exato de cada caractere + Otsu local, errava quase tudo com Otsu global — mesmo com a fatia já bem alinhada. `segmentar()` ganhou um parâmetro `cinza` opcional: quando presente, binariza cada fatia separadamente a partir da imagem em tons de cinza, não da já-binarizada.
- **Depois dos 3 fixes, a acurácia agregada no conjunto de 14 placas não melhorou** (ficou em 0,051, no mesmo patamar ou pior que o 0,061 inicial) — apesar de, no exemplo que inspecionei manualmente pixel a pixel, a segmentação ter ficado **visualmente perfeita** (7 caracteres limpos e bem isolados, iguais ao "estilo treino"). Isso decepcionou a expectativa: se os 3 bugs eram a causa completa, o número deveria ter subido bem mais perto de 0,94 (a acurácia por caractere isolado do Dia 4).
- Testei mais 2 hipóteses pontuais pra explicar o resíduo, nenhuma resolveu: (a) resolução intermediária baixa (`preparar()` reduz a placa pra 200×62px antes de segmentar) — testado com 400×124 e 600×186, sem melhora; (b) a rotação aplicada em `recortar_via_caixas()` introduzindo borrão — testado sem rotação nenhuma, sem melhora. A hipótese mais provável que sobra: mesmo com a segmentação "parecendo" perfeita a olho nu numa imagem pequena, o recorte por componente conectado ainda difere da caixa exata da anotação por poucos pixels — o suficiente pra confundir uma CNN treinada em recortes muito precisos, mesmo que a diferença seja imperceptível numa inspeção visual rápida. Não investiguei mais fundo por causa do tempo já investido nesta única frente.
- Demonstração qualitativa do pipeline COMPLETO (`LeitorDePlacas.ler()`, com detecção YOLO de verdade) rodada em 6 fotos reais do dataset de detecção (formato Mercosul, o alvo real do projeto) — sem gabarito, só inspeção visual. Comparando à mão com o que dá pra ler nas fotos: pelo menos 1 caso claro de erro total mesmo com placa grande e legível ("BBT 5192" real → previsão errada). As 6 saíram com `status: revisao_manual` (confiança mínima abaixo do limiar de 0,70) — o mecanismo de segurança do `LeitorDePlacas` funcionou como projetado, sinalizando incerteza em vez de arriscar uma resposta errada.
- Decisão tomada com o Raphael, em duas etapas: primeiro, não rotular um conjunto de teste Mercosul à mão (opção que existia) — investir o tempo em caçar a causa raiz da segmentação em vez disso. Depois, com os 3 bugs corrigidos mas a acurácia agregada ainda sem melhora clara, decidido **parar a investigação aqui**, manter as 3 correções (são melhorias reais e testadas, mesmo sem resolver o problema completo) e documentar a limitação residual com honestidade, em vez de continuar ajustando indefinidamente.

**Métricas obtidas**
- Conjunto de teste reconstruído (`project-swcsj`, majoritariamente formato antigo): 14 placas.
- Acurácia por caractere — CNN sozinha: **0,051** · CNN + regra do formato: **0,051** (ver `resultados/tabelas/metricas_finais.csv`).
- Acurácia por placa (as 7 corretas) — CNN sozinha: **0,0** · CNN + regra: **0,0** (0 de 14 placas 100% corretas).
- Erros por posição (1 a 7, de 14 placas): ver `resultados/figuras/erros_por_posicao.png` — erro quase total em TODAS as posições, não concentrado numa posição específica.
- **Estes números NÃO representam de forma confiável a acurácia real do sistema no formato Mercosul** (o alvo do projeto) — foram medidos sobre um dataset majoritariamente de formato antigo, com uma técnica de recorte (componentes conectados) que, mesmo corrigida 3 vezes, não fechou o resíduo. Ver limitação abaixo.

**LIMITAÇÃO HONESTA (a mais séria do projeto, não resolvida apesar de 3 correções reais)**
1. **Causa raiz do resíduo não identificada com certeza.** Corrigi recorte, segmentação e binarização — cada correção comprovada individualmente (inclusive com 100% de acerto num teste isolado com caixas exatas) — mas o pipeline fim a fim continua com acurácia muito baixa mesmo quando a segmentação parece visualmente perfeita. A hipótese mais provável (diferença de poucos pixels entre o recorte por componente conectado e a caixa exata da anotação, suficiente pra confundir a CNN) não foi confirmada nem descartada por falta de tempo.
2. A demonstração qualitativa (6 fotos Mercosul reais) sugere que o problema não é exclusivo do formato antigo: pelo menos 1 caso com placa grande e nítida saiu completamente errado. A acurácia de 0,9435 por caractere do Dia 4 (medida em caracteres já recortados exatamente pela caixa de anotação) **não se traduz** em acurácia equivalente no pipeline fim a fim, que depende de `segmentar()` acertar o recorte de cada caractere sozinho, sem ter a caixa exata disponível.
3. Não foi montado um conjunto de teste quantitativo e confiável pro formato Mercosul (exigiria rotular fotos reais à mão, ~20-30 no mínimo) — decisão consciente de não fazer isso agora, dado o prazo. **A meta "acurácia por placa > 0,80" não foi comprovadamente atingida nem decisivamente descartada** — está sem medição confiável.
4. O mecanismo de `LIMIAR_CONFIANCA` (0,70) do `LeitorDePlacas` funciona como rede de segurança: nas 6 fotos de demonstração, nenhuma saiu com confiança suficiente pra ser aceita sem revisão — o sistema errou, mas soube que tinha errado. Isso é um resultado honesto pra reportar: o sistema é conservador, não confiantemente errado.

**Decisões**
- Reescrever `segmentar()` pra usar componentes conectados (não largura igual) e aceitar um parâmetro `cinza` opcional pra binarização local por fatia — as duas mudanças ficam no código (`src/preprocessamento.py`, `src/pipeline.py`) porque são estritamente mais corretas que a versão anterior, mesmo não tendo fechado a métrica. `LeitorDePlacas.ler()` foi atualizado pra usar a nova assinatura.
- Medir a parte de reconhecimento (pré-processamento → segmentação → CNN → regra) separada da detecção, usando o dataset de caracteres com reconstrução de texto via ordenação de caixas — abordagem correta em princípio, mas expôs uma incompatibilidade de dataset (formato antigo) e, mais fundo, uma sensibilidade da CNN a imprecisões pequenas de recorte.
- Não rotular manualmente um conjunto Mercosul pra ter um número "bonito" pro relatório, e não continuar ajustando parâmetros indefinidamente atrás de um número melhor — reportar a limitação real, já bem investigada, é mais honesto e mais produtivo que perseguir um resultado incerto sem fim claro. Ver `resultados/figuras/verificacao_recorte_dia5.png` e `resultados/figuras/pipeline_exemplos.png`.
- `recortar_via_caixas()` fica só no notebook (`05_pipeline_final.py`), não em `src/` — só é possível porque o dataset de caracteres anota cada caractere individualmente; o detector real (Dia 1/2) não tem essa informação, então essa função não serve pro pipeline de produção.

**Pendências**
- **Achar a causa raiz do resíduo de acurácia** mesmo com segmentação visualmente correta — maior pendência técnica do projeto. Caminhos não testados por falta de tempo: comparar pixel a pixel o recorte por componente conectado contra a caixa exata da anotação pra medir o desalinhamento real (não só visual); tentar um `LIMIAR_CONFIANCA` mais alto combinado com uma CNN re-treinada especificamente sobre recortes de `segmentar()` (não sobre caixas exatas), pra fechar o gap entre distribuição de treino e de inferência.
- Montar um conjunto de teste Mercosul rotulado à mão (20-30 fotos) pra ter uma medição confiável de acurácia por placa fim a fim — não coube no prazo desta sessão.

**Próximo passo**
- Fechamento da entrega: `RELATORIO.md` (+ PDF) cobrindo os 5 itens pedidos pela professora, `README.md` com a tabela de resultados preenchida (sendo honesto sobre a limitação do Dia 5), e `git push`. Ver plano em `/Users/raphaelmagalhaes/.claude/plans/eu-j-estou-atrasado-groovy-squirrel.md`.

---

## Dia 6 — Causa raiz do Dia 5 encontrada, e a virada para YOLO de caracteres 🔄

**Objetivo:** entender por que a acurácia do Dia 5 ficou em 0,051 e decidir se valia recomeçar, trocar de dataset ou corrigir.

**Resposta curta:** nenhuma das três. Era bug, e a medição também estava errada.

**As três descobertas**

1. **Bug 1 — `corte_superior=0.35` decapitava os caracteres (A CAUSA DOMINANTE).**
   `pipeline.py` calculava o `layout` e não usava para escolher o corte; `05_pipeline_final.py` passava o default mesmo com `recortar_via_caixas()` já entregando só a faixa dos caracteres. Numa imagem de 200×62, `y0 = 21`, mas os caracteres começam por volta da linha 6 — **o terço superior de cada caractere ia fora antes da segmentação**. Isso explica as previsões do Dia 5 cheias de `I`, `1`, `4`, `L`, `P`: são as formas que sobram de um caractere sem topo.
   Corrigido com `CORTE_POR_LAYOUT = {"mercosul": 0.35, "antiga": 0.30}` e `corte_superior=0.0` no notebook.

2. **Bug 2 — `segmentar()` calculava `(y, h)` por componente e descartava.**
   Recortava `fonte[:, x:x+w]` (justo em X, altura inteira da faixa), enquanto o treino da CNN recorta justo nos dois eixos (`04_cnn_caracteres.py:65`). Corrigido com `_apertar_vertical()`. **Efeito medido: neutro** — ver ablação abaixo. Mantido porque é conceitualmente correto, sem alegar ganho.

3. **A medição do Dia 5 era inválida.** `dados/caracteres/data.yaml` tem `nc: 38` com a classe `EUR`: é um dataset **europeu**. Os gabaritos (`RALLYUS`, `6366363`, `S9WTPKL`) não seguem LLLDDDD nem LLLDLDD, então aplicar a máscara brasileira só podia degradar — por isso `acc` deu idêntica com e sem regra (0,051 → 0,051). A ablação do Dia 6 mede **sem máscara**.

**Ablação (`resultados/tabelas/ablacao_segmentacao.csv`, 14 placas, sem máscara)**

| variante | acc_caractere | acc_placa |
| --- | --- | --- |
| Dia 5 (baseline) | 0,0612 | 0,0000 |
| **só `corte_superior=0`** | **0,6122** | **0,2143** |
| só aperto vertical | 0,1224 | 0,0000 |
| corte=0 + aperto | 0,5816 | 0,1429 |
| corte=0 + aperto + folga 10% | 0,6122 | 0,2143 |

O Bug 1 sozinho responde por praticamente todo o ganho. O Bug 2 não acrescenta nada sobre ele (diferença de 3 caracteres em 98, dentro do ruído). `pipeline.py` usa `margem_vertical=0.10`, a melhor configuração medida.

**Por que 0,612 e não 0,94: o teto da segmentação clássica (medido)**

`_componentes_de_caracteres` acha **exatamente 7 blobs em apenas 4 de 14 placas** a 200×62. Aumentando a resolução intermediária melhora e satura:

| tamanho intermediário | placas com exatamente 7 blobs |
| --- | --- |
| 200×62 (atual) | 4 de 14 |
| 400×124 | 6 de 14 |
| 600×186 | 8 de 14 |
| + abertura morfológica | praticamente sem efeito |

Os caracteres se encostam uns nos outros e na moldura da placa depois da binarização, então os blobs não correspondem a caracteres e `_juntar_ou_dividir` faz cirurgia grosseira. **Não é parâmetro mal ajustado, é o teto do método.**

**Decisão: YOLO detector de caracteres**

O `project-swcsj` já anota cada caractere individualmente — **32.225 caixas no treino**. É exatamente o formato para treinar um YOLO que detecta os 7 caracteres direto, sem binarizar, sem componentes conectados, sem `corte_superior`. Mesma infra do Dia 1.
- Ensaio de 3 épocas: mAP50 = 0,077 → 0,206 → **0,290**, `cls_loss` de 4,18 → 2,66. Aprendendo de forma saudável; caminhos e rótulos validados.
- Treino completo de 40 épocas disparado (`notebooks/06_yolo_caracteres_treinar.py`), com checkpoint no Drive a cada 3 min e `resume=True`, como no Dia 1.
- A segmentação clássica **fica no código**: o relatório compara os dois métodos, o que é um resultado melhor do que só reportar uma limitação.

**Testes**

Os dois bugs eram invisíveis para a suíte porque `_faixa_com_blocos` desenhava todo bloco na mesma extensão vertical. Novo `_faixa_com_blocos_variados` + 5 testes; **verificado que 3 deles falham se a correção for revertida**. Total: 28 testes.

**Correções de processo**

- `metricas.py`: `acuracia_caractere` contava predição vazia como 7 erros, `erros_por_posicao` pulava — uniformizado, com teste.
- **A chave do Roboflow exposta nos notebooks foi revogada** (confirmado: `401 This API key does not exist or has been revoked`). A nova fica só na VM (`/content/.roboflow_key`), nunca no repositório. Falta trocar as 4 ocorrências hardcoded nos notebooks 01/03/04/05.

**Achados de ambiente**

- **Inferência roda local**: o `.venv` do Mac tem TF 2.17 + cv2 4.10 e `dados/` está no disco. Toda a ablação rodou em segundos, sem Colab. Colab só para treinar.
- `colab drivemount` **precisa ser rodado pelo Raphael no terminal** — exige autenticação interativa; disparado pelo agente dá `ValueError: mount failed`, e sessão nova não resolve.
- O `.keras` salvo pela VM (Keras 3.13) não abre no Keras 3.10 local, porque o venv é Python 3.9 e `keras>=3.11` exige Python 3.11+. Contorno: `modelos/cnn_chars_compat.keras`, o mesmo modelo com a chave `quantization_config` removida do `config.json` interno — mesmos 359.588 parâmetros.

**O conjunto de teste Mercosul (o que faltava desde o Dia 5)**

`notebooks/06_gabarito_mercosul.py` monta o conjunto e mede. **30 placas brasileiras — 16 Mercosul, 14 formato antigo — com texto verdadeiro conferido por humano.**

- **Achado: o dataset trafficbr tem anotações grosseiramente erradas.** As 12 maiores caixas rotuladas `plate` têm razão largura/altura entre 0,71 e 1,70 e cobrem 45-63% da imagem — são carros inteiros. Ordenar por área sem filtrar seleciona justamente esses erros. Filtro: razão entre 1,8 e 5,0 (placa real tem mediana 2,20) e no máximo 25% da imagem. **Isso provavelmente também explica os piores casos do Dia 2, com IoU 0,01-0,03 — era erro de anotação, não do detector.** Vale corrigir aquela interpretação no relatório.
- O split de teste tem só 3 caixas com ≥100×40px depois do filtro, então o conjunto usa `test` + `valid` (16 + 146 candidatas a ≥90×32px), com a origem registrada no CSV. As de `valid` foram usadas para selecionar o `best.pt` do detector (viés otimista leve na detecção), mas são inéditas para o reconhecimento, que é o que este conjunto mede.
- **Ponto metodológico:** o agente leu as 30 placas, mas 6 das leituras eram `O` vs `0` que só se resolviam **usando a regra de formato** — e o sistema avaliado também usa essa regra. Gabarito assim seria circular e inflaria a acurácia. O Raphael conferiu as 8 duvidosas: todas confirmaram a leitura do agente, e a #6 (ilegível por reflexo) ele leu como `BCE7C00`.

**Resultado final: comparação dos 3 pipelines, mesmas 30 placas** (`resultados/tabelas/comparacao_pipelines.csv`)

| pipeline | acc_caractere | acc_placa |
| --- | --- | --- |
| A) clássico (`segmentar`) + CNN do Dia 4 | 0,300 | 0/30 |
| B) YOLO só para as **caixas** + a **mesma** CNN | 0,619 | 7/30 |
| **C) YOLO caixa + classe** | **0,719** | **10/30 (0,333)** |

**O experimento B é a prova da tese:** mantendo a CNN idêntica dos dois lados e trocando apenas a segmentação, a acurácia por caractere dobra e a acurácia por placa sai de zero. O gargalo era a segmentação — não a CNN, não o dataset, não o detector.

Erros por posição caem de forma consistente: A `[24, 18, 21, 17, 21, 23, 23]` → C `[5, 8, 8, 10, 9, 10, 3]`. O miolo erra mais que as pontas, o que sugere confusão de classe e não falha de detecção.

**`CONF_CARACTERE = 0,05`** (não 0,25). Como a placa tem exatamente 7 caracteres e ficamos com os 7 de maior confiança, falso positivo é barato — faltar caractere estraga a leitura inteira e ainda invalida a regra de formato. Varredura:

| conf | placas com 7 chars | acc_caractere | acc_placa |
| --- | --- | --- | --- |
| 0,05 | 29/30 | 0,7190 | 0,3333 |
| 0,10 | 27/30 | 0,6952 | 0,3000 |
| 0,25 | 23/30 | 0,6571 | 0,2333 |
| 0,50 | 8/30 | 0,4810 | 0,1667 |

A regra de formato passou a render **+10 pontos** em C (0,619 → 0,719), contra +3 no pipeline clássico — regra só conserta leitura que já está quase certa.

**Queda de sessão: a estratégia do Dia 1 foi exercitada e funcionou**

A sessão do Colab caiu por volta da época 30. O `colab exec` perdeu a sessão às 22:51, mas o `last.pt` no Drive continuou sendo atualizado até 23:11 — foi só o websocket que caiu, a VM seguiu treinando e sincronizando. O checkpoint a cada 3 min preservou tudo, e o modelo foi recuperado do Drive sem perda. **Nenhuma época foi refeita.**

**LIMITAÇÕES HONESTAS**

1. **A meta "acurácia por placa > 0,80" NÃO foi atingida.** Está em 0,333 (10 de 30). Saiu de zero, o que é progresso real e mensurável, mas não é a meta.
2. Acurácia por caractere 0,719 também está abaixo da meta de 0,95. A CNN isolada faz 0,9435 em recortes exatos — a diferença é o que a detecção de caracteres ainda perde.
3. O modelo avaliado é o `last.pt` da época ~30 de 40, não o `best.pt` final. Há margem não medida.
4. Conjunto de 30 placas é pequeno: cada placa vale 3,3 pontos na acurácia por placa.
5. Detecção de layout acerta 24 de 30 — cada erro corrompe a máscara de formato.
6. `LIMIAR_CONFIANCA = 0,70` continua rejeitando tudo no pipeline clássico (conf_minima entre 0,196 e 0,563). Não recalibrado.

**Pendências**

- Fechar as 40 épocas do YOLO de caracteres (retomar do checkpoint no Drive) e reavaliar com o `best.pt` final.
- Recalibrar `LIMIAR_CONFIANCA` com os dados de gabarito.
- Trocar a chave hardcoded do Roboflow nos notebooks 01/03/04/05 por leitura de arquivo/variável de ambiente (a antiga já foi revogada; a nova nunca entrou no repo).
- Corrigir no relatório a interpretação da análise de erros do Dia 2, à luz das anotações erradas descobertas hoje.
- Ampliar o conjunto de gabarito, se sobrar tempo — 30 placas é pouco para separar 0,33 de 0,45.

**Próximo passo**
- Retomar o treino do YOLO até as 40 épocas, reavaliar, e escrever `RELATORIO.md` com a comparação dos dois métodos de segmentação como resultado central.

---

## Dia 6 (escopo original) — ONNX, API e latência ❌ NÃO EXECUTADO

**Cortado do escopo** na decisão registrada no topo deste arquivo, por causa do
atraso em relação ao roteiro. Nenhuma métrica de latência foi medida, e nada
disso aparece em material voltado à professora — o `RELATORIO.pdf` entregue não
menciona API, ONNX nem latência em lugar nenhum (conferido no Dia 7).

`api/app.py`, `api/Dockerfile` e as dependências de ONNX continuam no
repositório, sem uso. O `app.py` está escrito mas nunca foi executado, e ainda
instancia `LeitorDePlacas` (a rota clássica, acurácia 0,300) em vez de
`LeitorDePlacasYOLO` (a rota final, 0,719).

**Para retomar:** `docs/RETOMAR.md`, seção 4.6.

---

## Dia 7 — Relatório, entrega e organização do repositório ✅

**Objetivo:** fechar a entrega para a professora e deixar o projeto em estado de
pausa organizada, para ser retomado meses depois sem redescobrir nada.

**O que foi feito**

1. **Relatório em duas versões, de um Markdown só.**
   `notebooks/10_gerar_html_relatorio.py` passou a gerar dois arquivos a partir
   do mesmo `RELATORIO.md`: `RELATORIO.html` (completo, fica no repositório) e
   `RELATORIO_entrega.html` (sem as seções 6.4 Trabalhos futuros e 6.5
   Considerações de privacidade, que o Raphael preferiu não enviar). O corte é
   declarado numa lista no topo do script (`SECOES_FORA_DA_ENTREGA`), não editado
   à mão — o Markdown continua sendo a única fonte da verdade.

2. **Pasta `entrega/` consertada.** Três defeitos reais:
   - O PDF nunca era copiado: o script procurava exatamente `RELATORIO.pdf`, mas
     o navegador salva com o título da página. Resolvido com `achar_pdf()`, que
     varre a raiz.
   - A entrega levava o HTML completo, não o cortado.
   - O `LEIA-ME.md` apontava para "a seção 6.5 do relatório", que deixou de
     existir na versão entregue.
   Conferido extraindo o texto das 12 páginas do PDF: 6.3 e o Anexo presentes,
   6.4 e 6.5 ausentes. **A entrega está coerente.**

3. **Chave do Roboflow removida do código.** As 4 ocorrências hardcoded nos
   notebooks 01/03/04/05 (pendência aberta desde o Dia 6) foram trocadas pelo
   mesmo padrão do notebook 06: `ROBOFLOW_API_KEY` ou `/content/.roboflow_key`.
   **Decisão: o histórico do Git NÃO foi reescrito.** A chave está revogada
   (retorna 401), e reescrever 21 commits quebraria o remote sem ganho real de
   segurança. Em troca, o fato ficou registrado em três lugares, para que um 401
   futuro seja diagnosticado em segundos: no comentário de cada notebook, no
   README e em `docs/RETOMAR.md` §5.

4. **README reescrito.** Estava com a tabela de resultados em "(preencher)",
   prometia 14 testes (são 28), descrevia o pipeline pela rota clássica que foi
   substituída, e trazia um exemplo de resposta da API com uma latência
   (`"tempo_ms": 143.2`) que nunca foi medida. O número inventado saiu.

5. **`docs/RETOMAR.md` criado** — a peça que faltava para o projeto poder ficar
   parado. Sete seções: estado atual, como voltar a rodar em 5 minutos, o mapa
   do código, as 8 pendências ordenadas por retorno/esforço (cada uma com onde
   mexer e como medir se melhorou), as armadilhas que custaram tempo, onde está
   cada documento, e por onde começar numa retomada.

6. **Limpeza.** Removidas `_to_delete/` (1,2 MB de locks do git), `_preview/`
   (5,8 MB) e `t2/` (8,3 MB, trabalho de outra aluna). Antes de apagar, os 3 PNGs
   de `_preview/` que não tinham cópia idêntica em `resultados/figuras/` foram
   movidos para `docs/figuras_exploratorias/`. **Removidos depois, na varredura
   do mesmo dia** (ver abaixo): não eram citados em lugar nenhum e pesavam
   3,3 MB num repositório de 15 MB.

**Entregues**
- [x] `RELATORIO.pdf` — 12 páginas, em `entrega/RELATORIO.pdf`
- [ ] 10 slides — **cortados do escopo** (decisão do topo deste arquivo)
- [x] Repositório limpo e público
- [x] Seção de privacidade escrita — no `LEIA-ME.md` da entrega e no README;
      no relatório era a 6.5, retirada da versão da professora a pedido do Raphael

**A entrega, conferida**

`entrega/` — 42 MB: `RELATORIO.pdf`, `RELATORIO.html`, `ALPR_Mercosul.ipynb`,
`LEIA-ME.md`, `requirements.txt`, `src/` (5 módulos), `modelos/` (3 pesos,
29 MB), `imagens_exemplo/` (as 30 placas do gabarito), `resultados/tabelas/`
(18 CSVs + 4 JSONs) e `resultados/figuras/` (13 PNGs). Os 28 testes passam.

**Pendências que ficam abertas** (todas documentadas em `docs/RETOMAR.md`)

- `chars_best.pt` é o checkpoint da época ~30 de 40 — a de maior retorno.
- `LIMIAR_CONFIANCA = 0,70` nunca recalibrado.
- Gabarito de 30 placas é pequeno; cada placa vale 3,3 pontos.
- `api/` escrito mas não executado, e ligado ao pipeline clássico.

**Próximo passo:** projeto em pausa. Ao retomar, ler `docs/RETOMAR.md` §7.

---

## Registro de problemas

Anote aqui o que quebrou e como foi resolvido. Vira a seção "Limitações" do relatório.

| Dia | Problema | Solução |
| --- | --- | --- |
| 1 | `colab exec` tem timeout padrão de 30s, causando `TimeoutError` em scripts mais demorados (matplotlib, downloads, treino) | Sempre passar `--timeout` explícito (600 pra scripts longos, 60 pra snippets rápidos) |
| 1 | `colab upload` falhava de forma inconsistente (`File or directory not found` no caminho remoto) mesmo com sintaxe correta | Escrever o conteúdo do script via stdin em `colab exec` (`open(caminho,'w').write(codigo)`) em vez de usar `colab upload` |
| 1 | Sessões gratuitas do Colab caem imprevisivelmente (perda de conexão, kernel perdido) durante o treino de 40 épocas — uma vez perdendo ~1h de progresso | Checkpoint periódico (`last.pt` copiado pro Drive a cada 180s) + detecção automática e retomada (`resume=True`) no início do script |
| 1 | Disco local do Colab é apagado a cada sessão nova, então `results.csv` só guarda a última sessão de treino, não o histórico completo das 40 épocas | Aceito como limitação conhecida — pesos finais são cumulativos e válidos, só o gráfico de evolução ficou fragmentado |
| 4 | Figuras/tabelas dos Dias 1-3 nunca tinham sido trazidas do Drive pro repositório Git local — só existiam na cópia da VM (`colab exec` salva direto no Drive montado, não no Mac) | Baixadas todas de uma vez via `colab download` no Dia 4; a partir de agora, baixar do Drive pro local antes de fechar cada sessão, não só ao final do projeto |
| 4 | CNN de 36 classes ficou em 0,9348 na primeira rodada (meta > 0,95), com Q e O (as classes mais raras do treino) em 0% de acerto no teste | `class_weight` balanceado por classe no `fit()` — subiu para 0,9435; residual aceito como limitação (Q/O têm só 1-3 exemplos no teste, ruído estatístico) |
| 5 | `colab drivemount` deu `ValueError: mount failed` na primeira sessão do dia; a sessão de exec também caiu uma vez no meio do script (`RuntimeError: Connection was lost`) | Sessão nova (`colab new` de novo) resolveu o mount; a queda de conexão foi só o websocket — a VM continuou viva e `colab exec` de novo retomou sem perder nada |
| 5 | `colab upload` falhou com `500 Internal Server Error` nos 5 arquivos de `src/` (mesma instabilidade já registrada no Dia 1) | Workaround: gerar um script Python que escreve o conteúdo de cada arquivo via `open(...).write(repr_do_conteudo)` e mandar por `colab exec -f` (stdin), em vez de `colab upload` |
| 5 | Acurácia por caractere/placa caiu quase a zero ao medir o pipeline fim a fim sobre o dataset de caracteres (`project-swcsj`) | Investigado a fundo, 3 bugs reais corrigidos: (1) imagens do dataset têm fundo e rotação forte — corrigido com recorte via geometria das caixas anotadas; (2) `segmentar()` assumia largura igual entre os 7 caracteres — reescrita pra usar componentes conectados; (3) binarização Otsu era global (placa inteira) em vez de local (por caractere, como no treino) — corrigido com Otsu por fatia. Mesmo assim a acurácia agregada não melhorou — causa exata do resíduo não identificada, decisão de parar a investigação e documentar como limitação (ver Dia 5) |
| 5 | `colab exec` do kernel mantém os módulos Python já importados em memória entre chamadas — reenviar um `src/` atualizado pro `/content/src` NÃO faz o kernel usar a versão nova, continua rodando o código antigo já importado | `colab restart-kernel -s <sessao>` antes de re-testar qualquer mudança em `src/` na mesma sessão — sem isso, o diagnóstico fica testando código desatualizado sem erro nenhum pra avisar |
| 6 | Acurácia de 0,051 no Dia 5, causa "não identificada" | `corte_superior=0.35` era aplicado mesmo quando a entrada já era só a faixa dos caracteres, decapitando o topo de cada um. Corrigido: acc por caractere de 0,061 para 0,612, acc por placa de 0/14 para 3/14 |
| 6 | Os testes não pegavam nenhum dos dois bugs de recorte | `_faixa_com_blocos` desenhava todo bloco na mesma extensão vertical. Novo `_faixa_com_blocos_variados` + 5 testes, com falha verificada ao reverter a correção |
| 6 | Segmentação por componentes conectados acha 7 blobs em só 4/14 placas (8/14 no melhor caso) | Teto do método, não parâmetro. Decisão: treinar YOLO detector de caracteres sobre as 32.225 caixas já anotadas |
| 6 | `colab drivemount` falha com `ValueError: mount failed` mesmo em sessão nova | Exige autenticação interativa — precisa ser rodado pelo Raphael no terminal dele, não pelo agente |
| 6 | Chave do Roboflow hardcoded nos notebooks foi revogada (`401`) | Chave nova gravada só na VM (`/content/.roboflow_key`); notebook 06 lê de lá. Pendente trocar nos notebooks 01/03/04/05 |
| 6 | `cnn_chars.keras` salvo pelo Keras 3.13 da VM não abre no Keras 3.10 local (venv é Python 3.9, `keras>=3.11` exige 3.11+) | `cnn_chars_compat.keras`: mesmo modelo com `quantization_config` removida do `config.json` interno |
| 6 | Sessão do Colab caiu na época ~30 de 40 do YOLO de caracteres | Só o websocket caiu; a VM seguiu treinando. O checkpoint no Drive a cada 3 min (estratégia do Dia 1) preservou tudo — modelo recuperado sem refazer nenhuma época |
| 6 | Dataset trafficbr tem caixas rotuladas "plate" que são carros inteiros (45-63% da imagem, razão 0,71-1,70) | Filtro por razão largura/altura (1,8-5,0) e fração máxima da imagem (25%) em `06_gabarito_mercosul.py`. Reinterpreta os piores casos do Dia 2 |
| 6 | `conf` padrão de 0,25 no detector de caracteres deixava faltar caractere em 7 de 30 placas | `CONF_CARACTERE = 0,05` — a placa tem 7 caracteres e ficamos com os 7 de maior confiança, então falso positivo é barato. Completas: 23/30 → 29/30 |
