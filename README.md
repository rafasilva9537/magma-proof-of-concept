# MAGMA Original + MAGMA Narrativa

Reimplementação do MAGMA (*Multi-Graph based Agentic Memory Architecture*, Jiang et al., ACL 2026) e extensão para narrativas multiagente (fato × declaração, visibilidade por personagem e contradição).

- Design, decisões e plano: [`docs/DESIGN.md`](docs/DESIGN.md)
- Regras de código: [`magma/AGENTS.md`](magma/AGENTS.md)

## Requisitos
- Python 3.12 (o `faiss-cpu` ainda não publica versão para Python mais novo)
- ~4 GB livres em disco para os modelos locais (embedding e classificador de intenção)
- ~5 GB de RAM por processo do runner (os dois modelos ficam carregados); vários runs em paralelo somam

## Dados

Nenhum conjunto de dados é versionado neste repositório. Cada um vem da fonte abaixo, e o caminho é passado ao runner por argumento.

| Conjunto | Uso | Fonte | Onde fica |
|---|---|---|---|
| LoCoMo (`locomo10.json`, 10 conversas, 1.986 perguntas) | Retenção e comparação com o artigo do MAGMA | Cópia incluída no repositório oficial do MAGMA, [`FredJiang0324/MAGMA`](https://github.com/FredJiang0324/MAGMA), em `data/locomo10.json`; o conjunto original é o do [LoCoMo](https://github.com/snap-research/locomo) (Maharana et al., 2024). SHA-256 da cópia usada: `553cd5a15e25f2ceccc6ed185221eba645080c93e5b91087560a91aa5961f365` | `MAGMA-original-repo/data/locomo10.json` (clone do repositório oficial, só leitura) |
| FANToM (`fantom_v1.json`, 870 conjuntos de perguntas em 253 conversas) | Visibilidade por personagem | Arquivo oficial dos autores, baixado pelo carregador do repositório [`skywalker023/fantom`](https://github.com/skywalker023/fantom): `fantom.tar.gz` (1,04 MB) em `storage.googleapis.com/ai2-mosaic-public/projects/fantom/`. Licença MIT; os autores restringem o uso dos dados à avaliação. As cópias no Hugging Face são de terceiros e não são usadas | `data/fantom/fantom_v1.json` (ignorado pelo git) |

Para obter os dois:

```bash
git clone https://github.com/FredJiang0324/MAGMA.git MAGMA-original-repo
python -m bench.fantom download
```

O segundo comando baixa o `fantom.tar.gz`, confere o SHA-256 publicado no carregador dos autores (`1d08dfa0ea474c7f83b9bc7e3a7b466eab25194043489dd618b4c5223e1253a4`) e extrai o `fantom_v1.json` em `data/fantom/`.

## Instalação

1. Crie e ative o ambiente virtual:

   ```bash
   # Windows
   py -3.12 -m venv .venv
   .venv\Scripts\activate

   # Linux / macOS
   python3.12 -m venv .venv
   source .venv/bin/activate
   ```

2. Instale as dependências:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Baixe o modelo de embedding (`Qwen/Qwen3-Embedding-0.6B`, ~1,2 GB, do Hugging Face). Ele fica no cache local (`~/.cache/huggingface`) e só é baixado uma vez:

   ```bash
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')"
   ```

   Baixe também o classificador de intenção do router (`fastino/GLiNER2.5-Decide`, ~2 GB, licença Apache 2.0), que roda localmente na CPU:

   ```bash
   python -c "from gliner2 import AutoExtractor; AutoExtractor.from_pretrained('fastino/GLiNER2.5-Decide')"
   ```

   No Windows, os avisos sobre *symlinks* e sobre `HF_TOKEN` podem ser ignorados: o cache funciona sem eles, só ocupa um pouco mais de disco.

4. Configure os modelos de linguagem: copie `.env.example` para `.env` e preencha chave, URL e modelo do LLM principal (`LLM_*`), do juiz (`JUDGE_*`) e do segundo juiz (`JUDGE2_*`). Qualquer endpoint compatível com a API da OpenAI funciona. O `.env` nunca é versionado.

   - `*_CONTEXT_WINDOW`: janela do modelo em tokens; prompts maiores são recusados antes do envio.
   - `*_EXTRA_BODY`: JSON enviado em toda requisição. No OpenRouter, fixe **um único provedor**, sem fallback, e exija que ele respeite os parâmetros (`response_format`, `seed`): `{"provider": {"order": ["<provedor>"], "allow_fallbacks": false, "require_parameters": true}, "reasoning": {"effort": "medium"}}`. Sem isso, o mesmo modelo pode ser servido por provedores com quantizações diferentes no mesmo run. O provedor que atendeu cada chamada fica no log (`llm_calls.jsonl`).

5. Teste com uma conversa curta antes de um run grande (piloto barato, ~50 turnos do LoCoMo; as chamadas ficam no cache e são reaproveitadas depois):

   ```bash
   python -m bench.measure_consolidation --locomo MAGMA-original-repo/data/locomo10.json --sample conv-26 --max-turns 50
   ```

## Benchmark (LoCoMo)

O runner constrói uma memória por conversa, faz as perguntas na data da última sessão, julga cada resposta com o prompt C.3 do artigo (juiz em `JUDGE_*`) e calcula F1, BLEU-1, tempo de construção, tokens e latência por pergunta. Tudo vai para `results/<run_id>/` (ignorado pelo git): configuração do run, um registro por construção e por resposta, resumo, memórias salvas e o log cru das chamadas.

Só construir a memória da menor conversa (`conv-30`, 369 turnos), sem perguntas:

```bash
python -m bench.runner --locomo MAGMA-original-repo/data/locomo10.json --build-only
```

Medir a recuperação (Stages 1–4, sem chamar o LLM) sobre a memória construída: onde cada turno de evidência fica (canal temporal, âncoras, nós visitados, contexto), quantas vezes o piso de similaridade da travessia descarta um vizinho e o tamanho do contexto. Resultado em `runs/measurements/retrieval/<conversa>/`:

```bash
python -m bench.measure_retrieval --locomo MAGMA-original-repo/data/locomo10.json --memory results/magma-sample-build-seed7/memories/conv-30 --sample conv-30
```

Amostra estratificada de 40 perguntas da menor conversa (modo `sample`, padrão):

```bash
python -m bench.runner --locomo MAGMA-original-repo/data/locomo10.json
```

LoCoMo inteiro (modo `report`), com teto de custo. Os preços são em US$ por milhão de tokens (entrada e saída) do LLM da memória e do juiz. Depois de um piloto de 20 perguntas, o runner projeta o custo do resto e para se a projeção passar do teto:

```bash
python -m bench.runner --locomo MAGMA-original-repo/data/locomo10.json --mode report --max-cost-usd 20 --llm-prices <entrada> <saída> --judge-prices <entrada> <saída>
```

Chamadas já feitas saem do cache, sem custo; rodar o mesmo comando de novo refaz o run de graça.

O sistema testado é escolhido por `--system` (`magma`, o padrão; `narrative`; `full_context`; `full_context_visible`), e `--ablation` tira um componente de uma memória de grafo (ex.: `--ablation no_causal`, Tabs. 4 e 5 do artigo). As perguntas e o juiz são os mesmos; uma ablação reaproveita do cache a construção do grafo:

```bash
python -m bench.runner --locomo MAGMA-original-repo/data/locomo10.json --system full_context --mode report --questions 500 --max-cost-usd 3 --llm-prices <entrada> <saída> --judge-prices <entrada> <saída>
```

## Benchmark (FANToM)

Cada conversa do FANToM é construída inteira e recebe as perguntas fechadas dos seus conjuntos (crença em múltipla escolha, listas e sim/não de quem sabe cada informação), com a correção oficial, sem juiz. A presença de cada personagem em cada fala é derivada das saídas e voltas; para conferi-la contra as listas oficiais de quem sabe (sem chamar o LLM):

```bash
python -m bench.fantom presence
```

Amostra de 30 conversas inteiras (seed 7), na Arquitetura B; `--system` aceita os mesmos valores do runner do LoCoMo, e `--conversations 0` usa todas:

```bash
python -m bench.fantom run --system narrative --conversations 30 --max-cost-usd 1 --llm-prices <entrada> <saída>
```

## Figuras

Os gráficos dos resultados saem dos `summary.json` dos runs terminados (A × C1 por categoria do LoCoMo, tokens por pergunta e ablações da A), em SVG, PDF e PNG (300 dpi), na pasta `runs/figures/`; o `figures.json` lista os runs de onde vem cada figura:

```bash
python -m bench.figures
```

Os diagramas das arquiteturas ficam em `docs/figures/` como `.drawio.svg`: abrem como imagem comum e são editáveis no [draw.io](https://www.drawio.com/) (desktop ou navegador), que salva no mesmo arquivo. Para exportar em PNG com o draw.io desktop:

```bash
draw.io --export --format png --scale 2 --border 20 --output runs/figures/arquitetura_a.png docs/figures/arquitetura_a.drawio.svg
```

## Verificação

```bash
python -m pytest
python -m mypy
```

Os testes não chamam nenhuma API. Os testes que usam os modelos reais (embedding e intenção) são pulados por padrão; para rodá-los depois do passo 3:

```bash
# Windows (PowerShell)
$env:RUN_MODEL_TESTS="1"; python -m pytest tests/test_embed.py tests/regression

# Linux / macOS
RUN_MODEL_TESTS=1 python -m pytest tests/test_embed.py tests/regression
```

`tests/regression/` guarda o snapshot da Arquitetura A (perfil `magma`) gravado na tag `v-original`: a saída das etapas de consulta para as 105 perguntas da menor conversa do LoCoMo, sobre a memória salva em `tests/regression/data/conv-30/`, e uma impressão digital dos prompts e das configurações da ingestão. A parte da ingestão roda sempre; a das consultas, com `RUN_MODEL_TESTS=1`. Uma mudança em `magma/` que altere o snapshot precisa ser intencional: registrar o motivo, gravar o snapshot de novo e medir a Arquitetura A de novo:

```bash
python -m bench.snapshot_retrieval --locomo MAGMA-original-repo/data/locomo10.json --memory tests/regression/data/conv-30 --sample conv-30 --output tests/regression/conv-30.snapshot.json
```
