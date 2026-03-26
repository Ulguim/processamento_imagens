from __future__ import annotations

import warnings
from pathlib import Path

import cv2

from .catalogo import CatalogoImagens
from .metricas import calcular_metricas
from .modelos import ImagemInfo, OperacaoMorfologica, TipoRuido
from .operacoes import aplicar, criar_kernel
from .ruido import adicionar_ruido
from .ranking import gerar_ranking
from .visualizacao import gerar_relatorio, visualizar, visualizar_combinado

warnings.filterwarnings("ignore")


class ProcessadorMorfologico:
    """
    Orquestra o pipeline de processamento morfológico.
    Delega toda a lógica para os módulos especializados do pacote.

    Uso:
        p = ProcessadorMorfologico(tamanhos_kernel=[3, 5, 7])
        p.selecionar_imagens(categorias=["Fingerprint"]).selecionar_operacoes(todas=True)
        p.processar()
        p.visualizar(analise="fingerprint_todas", salvar=True)
        p.gerar_relatorio_metricas(analise="fingerprint_todas")
    """

    def __init__(
        self,
        diretorio_dataset:  str | Path = "./dataset/STI",
        diretorio_saida:    str | Path = "./resultados",
        tamanhos_kernel:    list[int]  = None,
        ruido:              TipoRuido  = TipoRuido.SALT_PEPPER,
        intensidade_ruido:  float      = 0.05,
        seed:               int        = 42,
    ) -> None:
        self.diretorio_saida   = Path(diretorio_saida)
        self.tamanhos_kernel   = tamanhos_kernel or [3, 5]
        self.ruido             = ruido
        self.intensidade_ruido = intensidade_ruido
        self.seed              = seed

        self._catalogo               = CatalogoImagens(diretorio_dataset)
        self._imagens_selecionadas:  list[ImagemInfo]          = []
        self._operacoes_selecionadas: list[OperacaoMorfologica] = []
        self._resultados:            dict                       = {}

    # ------------------------------------------------------------------
    # Seleção — retornam self para encadeamento
    # ------------------------------------------------------------------

    def selecionar_imagens(
        self,
        nomes:      list[str] = None,
        categorias: list[str] = None,
        todos:      bool      = False,
    ) -> "ProcessadorMorfologico":
        self._imagens_selecionadas = self._catalogo.selecionar(
            nomes=nomes, categorias=categorias, todos=todos
        )
        return self

    def selecionar_operacoes(
        self,
        operacoes: list[OperacaoMorfologica] = None,
        todas:     bool                      = False,
    ) -> "ProcessadorMorfologico":
        if todas:
            self._operacoes_selecionadas = list(OperacaoMorfologica)
        elif operacoes:
            self._operacoes_selecionadas = operacoes
        else:
            raise ValueError(
                f"Forneça 'operacoes' ou use 'todas=True'. "
                f"Disponíveis: {[o.value for o in OperacaoMorfologica]}"
            )
        return self

    def limpar_selecao(self) -> "ProcessadorMorfologico":
        self._imagens_selecionadas   = []
        self._operacoes_selecionadas = []
        return self

    def listar_imagens(self, categoria: str = None, formato: str = None) -> None:
        self._catalogo.listar(categoria=categoria, formato=formato)

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def processar(self) -> dict:
        """Executa o pipeline completo e armazena resultados internamente."""
        self._validar_selecao()
        self._resultados = {}

        for info in self._imagens_selecionadas:
            print(f"\n{'='*60}")
            print(f"Processando: {info.categoria}/{info.nome}.{info.formato}")
            print("=" * 60)

            img = cv2.imread(str(info.caminho), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  [AVISO] Não foi possível carregar {info.caminho}. Pulando.")
                continue

            _, binaria = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ruidosa    = adicionar_ruido(binaria, self.ruido, self.intensidade_ruido, self.seed)

            chave = f"{info.nome}_{info.formato}"
            self._resultados[chave] = {
                "info":     info,
                "original": img,
                "binaria":  binaria,
                "ruidosa":  ruidosa,
            }

            for ksize in self.tamanhos_kernel:
                kernel  = criar_kernel(ksize)
                chave_k = f"kernel_{ksize}x{ksize}"
                self._resultados[chave][chave_k] = {}

                for op in self._operacoes_selecionadas:
                    resultado = aplicar(op, ruidosa, kernel)
                    metricas  = calcular_metricas(binaria, resultado)
                    self._resultados[chave][chave_k][op.value] = {
                        "imagem":   resultado,
                        "metricas": metricas,
                    }
                    print(
                        f"  {op.value:12s} kernel={ksize}x{ksize} | "
                        f"Acc={metricas['pixel_accuracy']:.4f} "
                        f"IoU={metricas['jaccard']:.4f} "
                        f"F1={metricas['f1']:.4f} "
                        f"SSIM={metricas['ssim']:.4f}"
                    )

        return self._resultados

    # ------------------------------------------------------------------
    # Saída
    # ------------------------------------------------------------------

    def visualizar(
        self,
        analise:     str  = None,
        imagem_nome: str  = None,
        salvar:      bool = True,
        mostrar:     bool = False,
    ) -> None:
        if not self._resultados:
            raise RuntimeError("Execute processar() antes de visualizar().")
        pasta = self.diretorio_saida / analise if analise else self.diretorio_saida
        visualizar(
            resultados=self._resultados,
            operacoes=self._operacoes_selecionadas,
            pasta_saida=pasta,
            imagem_nome=imagem_nome,
            mostrar=mostrar,
            salvar=salvar,
        )

    def visualizar_combinado(
        self,
        analise:     str  = None,
        imagem_nome: str  = None,
        salvar:      bool = True,
        mostrar:     bool = False,
    ) -> None:
        if not self._resultados:
            raise RuntimeError("Execute processar() antes de visualizar_combinado().")
        pasta = self.diretorio_saida / analise if analise else self.diretorio_saida
        visualizar_combinado(
            resultados=self._resultados,
            operacoes=self._operacoes_selecionadas,
            pasta_saida=pasta,
            imagem_nome=imagem_nome,
            mostrar=mostrar,
            salvar=salvar,
        )

    def gerar_relatorio_metricas(self, analise: str = None) -> None:
        if not self._resultados:
            raise RuntimeError("Execute processar() antes de gerar_relatorio_metricas().")
        gerar_relatorio(self._resultados, analise=analise)

    def gerar_ranking(self, analise: str = None) -> None:
        if not self._resultados:
            raise RuntimeError("Execute processar() antes de gerar_ranking().")
        pasta = self.diretorio_saida / analise if analise else self.diretorio_saida
        gerar_ranking(self._resultados, pasta_saida=pasta, analise=analise)

    # ------------------------------------------------------------------
    # Interno
    # ------------------------------------------------------------------

    def _validar_selecao(self) -> None:
        if not self._imagens_selecionadas:
            raise RuntimeError("Nenhuma imagem selecionada. Use selecionar_imagens() primeiro.")
        if not self._operacoes_selecionadas:
            raise RuntimeError("Nenhuma operação selecionada. Use selecionar_operacoes() primeiro.")
