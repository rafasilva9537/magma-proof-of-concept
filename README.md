# MAGMA Original + MAGMA Narrativa

Reimplementação do MAGMA (*Multi-Graph based Agentic Memory Architecture*, Jiang et al., ACL 2026) e extensão para narrativas multiagente (fato × declaração, visibilidade por personagem e contradição).

- Design, decisões e plano: [`docs/DESIGN.md`](docs/DESIGN.md)
- Regras de código: [`magma/AGENTS.md`](magma/AGENTS.md)

## Requisitos
- Python 3.12 (o `faiss-cpu` ainda não publica versão para Python mais novo)
- ~4 GB livres para os modelos locais (embedding e classificador de intenção)

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

## Verificação

```bash
python -m pytest
python -m mypy
```

Os testes não chamam nenhuma API. O teste que usa o modelo de embedding real é pulado por padrão; para rodá-lo depois do passo 3:

```bash
# Windows (PowerShell)
$env:RUN_MODEL_TESTS="1"; python -m pytest tests/test_embed.py

# Linux / macOS
RUN_MODEL_TESTS=1 python -m pytest tests/test_embed.py
```
