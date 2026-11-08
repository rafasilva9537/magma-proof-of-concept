# MAGMA Original + MAGMA Narrativa

Reimplementação do MAGMA (*Multi-Graph based Agentic Memory Architecture*, Jiang et al., ACL 2026) e extensão para narrativas multiagente (fato × declaração, visibilidade por personagem e contradição).

- Design, decisões e plano: [`docs/DESIGN.md`](docs/DESIGN.md)
- Regras de código: [`magma/AGENTS.md`](magma/AGENTS.md)

## Requisitos
- Python 3.12 (o `faiss-cpu` ainda não publica versão para Python mais novo)
- ~2 GB livres para o modelo de embedding

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

   No Windows, os avisos sobre *symlinks* e sobre `HF_TOKEN` podem ser ignorados: o cache funciona sem eles, só ocupa um pouco mais de disco.

4. Configure os modelos de linguagem: copie `.env.example` para `.env` e preencha chave, URL e modelo do LLM principal (`LLM_*`), do juiz (`JUDGE_*`) e do segundo juiz (`JUDGE2_*`). Qualquer endpoint compatível com a API da OpenAI funciona. O `.env` nunca é versionado.

   - `*_CONTEXT_WINDOW`: janela do modelo em tokens; prompts maiores são recusados antes do envio.
   - `*_EXTRA_BODY`: JSON enviado em toda requisição, ex.: `{"provider": {"order": ["..."], "allow_fallbacks": false}, "reasoning": {"enabled": false}}` no OpenRouter.

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
