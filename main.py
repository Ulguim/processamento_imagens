from processador import ProcessadorMorfologico, OperacaoMorfologica, TipoRuido

# ── Caminhos ─────────────────────────────────────────────────────────────────
DATASET    = "./dataset/STI"
RESULTADOS = "./resultados"

# ── Parâmetros dos experimentos ───────────────────────────────────────────────
# Edite aqui para ajustar ou criar novos experimentos.
# Chave     : nome da análise (vira subpasta em resultados/)
# categorias: lista de categorias do dataset  (use OU categorias OU nomes)
# nomes     : lista de nomes de imagens
# operacoes : lista de OperacaoMorfologica  ou  "todas"
# kernels   : tamanhos dos elementos estruturantes (ímpares, ex: [3, 5, 7])
# ruido     : TipoRuido.SALT_PEPPER  ou  TipoRuido.GAUSSIANO
# intensidade: fração de pixels corrompidos (0.0 – 1.0)
# seed      : garante reprodutibilidade dos resultados

EXPERIMENTOS = {
    "fingerprint_todas_operacoes": dict(
        categorias  = ["Fingerprint"],
        operacoes   = "todas",
        kernels     = [3, 5, 7],
        ruido       = TipoRuido.SALT_PEPPER,
        intensidade = 0.05,
        seed        = 42,
    ),
    "medical_abertura_fechamento": dict(
        categorias  = ["Medical"],
        operacoes   = [OperacaoMorfologica.ABERTURA, OperacaoMorfologica.FECHAMENTO],
        kernels     = [3, 5],
        ruido       = TipoRuido.SALT_PEPPER,
        intensidade = 0.10,
        seed        = 42,
    ),
    "classica_erosao": dict(
        nomes       = ["cameraman"],
        operacoes   = [OperacaoMorfologica.EROSAO],
        kernels     = [3],
        ruido       = TipoRuido.SALT_PEPPER,
        intensidade = 0.05,
        seed        = 42,
    ),
}

# ── Execução ──────────────────────────────────────────────────────────────────
for nome, cfg in EXPERIMENTOS.items():
    p = ProcessadorMorfologico(
        diretorio_dataset = DATASET,
        diretorio_saida   = RESULTADOS,
        tamanhos_kernel   = cfg["kernels"],
        ruido             = cfg["ruido"],
        intensidade_ruido = cfg["intensidade"],
        seed              = cfg["seed"],
    )

    sel = p.selecionar_imagens(
        nomes      = cfg.get("nomes"),
        categorias = cfg.get("categorias"),
    )

    ops = cfg["operacoes"]
    if ops == "todas":
        sel.selecionar_operacoes(todas=True)
    else:
        sel.selecionar_operacoes(operacoes=ops)

    p.processar()
    p.visualizar(analise=nome, salvar=True)
    p.gerar_relatorio_metricas(analise=nome)

print(f"\nConcluído. Resultados em: {RESULTADOS}/")
