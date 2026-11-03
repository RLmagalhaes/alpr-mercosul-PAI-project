# ALPR Mercosul

Projeto acadêmico: recebe a foto de um veículo e devolve o texto da placa. O estado,
os resultados e como rodar estão no `README.md`.

Metas: mAP@0.5 > 0,90 na detecção, acurácia por caractere > 0,95, acurácia por placa > 0,80.

## Como trabalhar

O Raphael tem Python intermediário: lê e valida código, e quer entender o que foi feito.

- Escreva todo o código, comentado em português e pronto para rodar.
- Antes de um bloco, diga em 2 ou 3 linhas o que ele faz e por quê. Depois, diga qual
  saída esperar. Ao ver a saída real, explique em linguagem simples e diga se o
  resultado está bom ou ruim.
- Peça aprovação por issue e por PR. No meio do trabalho, pare só diante de uma
  surpresa ou antes de gastar cota de GPU em um treino longo.
- Trate erros direto: entregue a correção e explique a causa em 2 linhas.
- Reporte métricas com honestidade, mesmo quando forem ruins, e proponha o que tentar.
- Quando houver escolha, prefira o que funciona hoje ao que é elegante.

## Sessão

- **Início:** leia este arquivo e rode `gh issue list`.
- **Fim:** faça o commit, atualize as issues e rode `colab stop` se houver sessão aberta.

## Onde cada coisa fica

- O que falta fazer: GitHub Issues.
- Por que uma decisão foi tomada: `docs/adr/`.
- Registros do trabalho entregue: `docs/arquivo/`, consultados só quando a tarefa pedir
  esse histórico. A versão entregue é a tag `entrega-v1`.
- Modelos e trabalho com dados de licença restrita: repositório privado
  `RLmagalhaes/alpr-mercosul-modelos`. Este repositório é público e recebe só resultados
  de bases públicas.

## Execução

- Inferência, avaliação e testes rodam no `.venv` local, em CPU.
- Treino roda em GPU na nuvem. Antes de usar a Colab CLI, leia `docs/COLAB_SKILL.md`.
- Antes de treinar ou mexer em dados, leia `docs/armadilhas.md`.

## Convenções

- Use português nas explicações, nos comentários e nos nomes de variáveis.
- Todo número publicado precisa ter saído de código executado.
- Figuras vão para `resultados/figuras/` em PNG, com `dpi=140` e `bbox_inches='tight'`.
  Tabelas vão para `resultados/tabelas/` em CSV.
- `src/` é a fonte da verdade: os scripts importam de lá.

## Agent skills

- **Issue tracker:** GitHub Issues, com `gh`. Veja `docs/agents/issue-tracker.md`.
- **Triage labels:** `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`,
  `wontfix`. Veja `docs/agents/triage-labels.md`.
- **Domain docs:** single-context, com `GLOSSARY.md` e `docs/adr/` na raiz. Veja
  `docs/agents/domain.md`.
