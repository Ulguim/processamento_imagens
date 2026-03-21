from processador import ProcessadorMorfologico, OperacaoMorfologica, TipoRuido

# ── Configuração global ──────────────────────────────────────────────────────
DATASET    = "./dataset/STI"
RESULTADOS = "./resultados"
KERNELS    = [3, 5, 7]

# ── Experimento A: Fingerprint — todas as operações ─────────────────────────
p = ProcessadorMorfologico(
    diretorio_dataset=DATASET,
    diretorio_saida=RESULTADOS,
    tamanhos_kernel=KERNELS,
)
(
    p.selecionar_imagens(categorias=["Fingerprint"])
     .selecionar_operacoes(todas=True)
)
p.processar()
p.visualizar(analise="fingerprint_todas_operacoes", salvar=True)
p.gerar_relatorio_metricas(analise="fingerprint_todas_operacoes")

# ── Experimento B: Médicas — abertura e fechamento ───────────────────────────
p.limpar_selecao()
(
    p.selecionar_imagens(categorias=["Medical"])
     .selecionar_operacoes(operacoes=[
         OperacaoMorfologica.ABERTURA,
         OperacaoMorfologica.FECHAMENTO,
     ])
)
p.processar()
p.visualizar(analise="medical_abertura_fechamento", salvar=True)
p.gerar_relatorio_metricas(analise="medical_abertura_fechamento")

# ── Experimento C: Clássica — erosão ────────────────────────────────────────
p2 = ProcessadorMorfologico(diretorio_dataset=DATASET, tamanhos_kernel=[3])
(
    p2.selecionar_imagens(nomes=["cameraman"])
      .selecionar_operacoes(operacoes=[OperacaoMorfologica.EROSAO])
)
p2.processar()
p2.visualizar(analise="classica_erosao", salvar=True)

print(f"\nConcluído. Resultados em: {RESULTADOS}/")
