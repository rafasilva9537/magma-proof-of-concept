# Regras de código

Vale para todo o código Python do projeto (`magma/`, `narrative/`, `bench/`, `baselines/`, `tests/`), escrito por pessoas ou por agentes. O design do sistema está em `docs/DESIGN.md`.

## 1. Legibilidade primeiro
O projeto precisa ser lido e explicado de ponta a ponta. Entre uma solução curta e esperta e uma mais longa e óbvia, escolha a óbvia.

- Funções curtas, com um propósito.

### Nomes
- **Descritivos e por extenso:** o nome diz o que a coisa é, sem abreviações (`anchor_scores`, não `s`; `semantic_edge_threshold`, não `sem_thr`; `Layer.TEMPORAL`, não `Layer.TEMP`).
- **Exceção:** símbolos e siglas do artigo, quando quem lê o código vai compará-lo com a equação (`rrf_k`, `lambda1`, `lambda2`, `w_tq`, `top_k`). A docstring diz a que equação ou tabela o nome corresponde.
- Membros de Enum em maiúsculas e com o valor igual ao nome (`CAUSAL = "CAUSAL"`), para que o texto gravado em JSON e o código digam a mesma coisa.
- Booleanos como pergunta (`is_anchor`, `adaptive`); funções como verbo (`select_anchors`, `linearize`); classes como substantivo (`AnchorSet`, `GraphStore`).
- Fluxo explícito: nada de metaprogramação, `getattr`/`setattr` dinâmicos, decoradores que mudam assinatura, registro automático por importação, `**kwargs` repassados às cegas ou monkey patching.
- Compreensões só quando cabem numa linha legível; senão, um `for`.
- Sem herança profunda. Composição e funções simples resolvem quase tudo.
- Comentários dizem **por quê**, não **o quê**. Nada de citar itens do design (D3, §6.4) no código.
- As equações e algoritmos do artigo ficam reconhecíveis: um passo do algoritmo vira uma função ou um bloco com nome, e as variáveis seguem os nomes do artigo quando ajudam (`w_tq`, `gamma`, `lambda1`).

## 2. Design desacoplado
Trocar a lógica de uma etapa (roteamento, seleção de âncoras, travessia, linearização) não pode exigir mudanças nas outras.

- Cada etapa expõe uma interface pequena (`Protocol` ou `ABC`, ver §4) e recebe suas dependências pelo construtor. Nenhuma etapa importa a implementação concreta de outra.
- Só a fachada (`memory.py`) e os pontos de entrada (consoles, runner) montam as peças concretas.
- Runner, métricas e consoles dependem só da interface `MemorySystem` (`add_turn`, `consolidate`, `query`), nunca do interior de `magma/` ou `narrative/`. Toda arquitetura e todo baseline a implementam.
- Etapas conversam por dataclasses de entrada e saída (ex.: `RouterOutput`, `AnchorSet`), não por estruturas soltas.
- Parâmetros vêm do `config.py`; nenhum número mágico espalhado pelo código.
- Um módulo novo ou uma variante (ex.: outro router) entra como outra implementação da mesma interface, não como `if` dentro da existente.
- `narrative/` importa e estende `magma/`; `magma/` nunca importa `narrative/`.
- Depois da tag `v-original`, o comportamento do perfil `magma` é fixo (ele é o grupo de controle da comparação). Editar `magma/` é permitido para ganchos e correções, desde que o gancho seja inoperante no perfil `magma` e o teste de regressão (`tests/regression/`) continue passando.

## 3. Armazenamento atrás de interfaces
O grafo e os índices são acessados só pelas interfaces (`GraphStore`, índices vetorial e lexical). A implementação atual (NetworkX em memória + JSON) precisa poder ser trocada por Neo4j ou outro backend sem tocar em ingestão, recuperação ou benchmark.

- Nenhum código fora da implementação concreta importa `networkx`, `faiss` ou `rank_bm25`.
- A interface fala em termos do domínio (`add_node`, `neighbors(node_id, layer)`), nunca em objetos da biblioteca.
- Os testes das interfaces rodam contra qualquer implementação.

## 4. Tipagem
Todo código é tipado e passa no `mypy` (configuração em `mypy.ini`).

- Dados internos: `@dataclass(frozen=True)` sempre que não precisarem mudar.
- Dados com formato de JSON que cruzam fronteiras (mensagens para o LLM, arquivos de cenário, linhas de log): `TypedDict`.
- Saída de LLM: modelo `pydantic`, que valida o que chega.
- Conjuntos fechados de valores do domínio (camadas, intenções, ablações, papéis, status): `StrEnum`, nunca `str` solto. O Enum é a fonte única dos valores (percorre-se com `for x in Layer`), valida texto vindo de arquivo, linha de comando ou LLM (`Layer("CAUSAL")`) e é gravado como texto no JSON.
- `Literal` só para campos que copiam um formato externo dentro de um `TypedDict` (ex.: `role` das mensagens da API do LLM, definido pelo provedor).
- Enum não pode ser estendido. Tipos que um pacote acrescenta a outro (ex.: tipos de nó e de aresta de um perfil) ficam num Enum por pacote, validados pelo registro de tipos.
- **Interfaces: `ABC` ou `Protocol`, conforme o papel.**
  - **`ABC`** para um contrato com vários métodos que o projeto implementa mais de uma vez e troca inteiro (backends de armazenamento como o `GraphStore`; as arquiteturas e os baselines do `MemorySystem`), sobretudo quando há uma regra que toda implementação precisa cumprir. A regra fica na base: o método público valida e chama um método abstrato que a implementação escreve, então nenhuma implementação consegue pulá-la. A implementação herda da base, o que deixa a relação explícita na leitura e na navegação da IDE.
  - **`Protocol`** para uma costura estreita (um a três métodos) com uma dependência externa ou entre etapas do pipeline (`ChatBackend`, `TextEncoder`, `Embedder`, router, travessia). Dublês de teste e adaptadores servem sem herdar.
  - Na dúvida, `Protocol`.
- **Evitar `dict` puro e `Any`.** `dict[str, X]` só quando a chave é de fato um identificador arbitrário (ex.: `node_id → Node`), nunca como registro com campos. `Any` só na borda com uma biblioteca sem tipos, isolado num único lugar e convertido para um tipo do projeto logo em seguida.
- Sem `# type: ignore` sem um comentário explicando o motivo.

## 5. Testes
- Cada módulo tem testes em `tests/`, sem chamadas reais a APIs (use dublês que implementam a mesma interface).
- Testes de algoritmo usam grafos pequenos montados à mão, com o resultado esperado calculado no próprio teste ou num comentário.
- `pytest` e `mypy` passam antes de cada commit.
