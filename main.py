# Grupo 3: Processamento Morfológico
# • Utilizar erosão, dilatação, abertura e fechamento.
# • Aplicar as técnicas em imagens binárias com ruído.
# • Discutir como a morfologia matemática pode remover ruídos ou realçar estruturas.
# • Comparar os resultados em termos de métricas de acurácia.

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import f1_score, jaccard_score

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OperacaoMorfologica(Enum):
    EROSAO     = "erosao"
    DILATACAO  = "dilatacao"
    ABERTURA   = "abertura"
    FECHAMENTO = "fechamento"


class TipoRuido(Enum):
    SALT_PEPPER = "salt_pepper"
    GAUSSIANO   = "gaussiano"


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ImagemInfo:
    nome:      str
    caminho:   Path
    categoria: str
    formato:   str


# ---------------------------------------------------------------------------
# Classe principal
# ---------------------------------------------------------------------------

class ProcessadorMorfologico:
    """
    Processa imagens com operações morfológicas de forma modular e reutilizável.

    Uso típico:
        p = ProcessadorMorfologico(tamanhos_kernel=[3, 5])
        p.selecionar_imagens(categorias=["Fingerprint"]).selecionar_operacoes(todas=True)
        p.processar()
        p.visualizar(salvar=True)
        p.gerar_relatorio_metricas()
    """

    _FORMATOS_SUPORTADOS = {".bmp", ".pgm", ".ppm", ".png", ".jpg", ".jpeg"}

    _MAPA_OPERACOES: dict = {}  # preenchido após definição dos lambdas abaixo

    def __init__(
        self,
        diretorio_dataset: str | Path = "./dataset/STI",
        diretorio_saida:   str | Path = "./resultados",
        tamanhos_kernel:   list[int]  = None,
        ruido:             TipoRuido  = TipoRuido.SALT_PEPPER,
        intensidade_ruido: float      = 0.05,
        seed:              int        = 42,
    ) -> None:
        self.diretorio_dataset  = Path(diretorio_dataset)
        self.diretorio_saida    = Path(diretorio_saida)
        self.tamanhos_kernel    = tamanhos_kernel or [3, 5]
        self.ruido              = ruido
        self.intensidade_ruido  = intensidade_ruido
        self.seed               = seed

        self._catalogo:              dict[str, ImagemInfo] = {}
        self._imagens_selecionadas:  list[ImagemInfo]      = []
        self._operacoes_selecionadas: list[OperacaoMorfologica] = []
        self._resultados:            dict                  = {}

    # ------------------------------------------------------------------
    # Catálogo e seleção de imagens
    # ------------------------------------------------------------------

    def construir_catalogo(self) -> "ProcessadorMorfologico":
        """Escaneia dataset_STI recursivamente e popula o catálogo interno."""
        self._catalogo = {}
        for caminho in sorted(self.diretorio_dataset.rglob("*")):
            if caminho.is_file() and caminho.suffix.lower() in self._FORMATOS_SUPORTADOS:
                categoria = caminho.parent.name
                chave = f"{categoria}/{caminho.name}"
                self._catalogo[chave] = ImagemInfo(
                    nome=caminho.stem,
                    caminho=caminho,
                    categoria=categoria,
                    formato=caminho.suffix.lstrip(".").lower(),
                )
        return self

    def listar_imagens(self, categoria: str = None, formato: str = None) -> list[ImagemInfo]:
        """Retorna lista filtrada do catálogo. Filtros são opcionais e combinam."""
        if not self._catalogo:
            self.construir_catalogo()
        resultado = list(self._catalogo.values())
        if categoria:
            resultado = [i for i in resultado if categoria.lower() in i.categoria.lower()]
        if formato:
            resultado = [i for i in resultado if i.formato == formato.lower().lstrip(".")]
        for img in resultado:
            print(f"  [{img.categoria}] {img.nome}.{img.formato}")
        return resultado

    def selecionar_imagens(
        self,
        nomes:      list[str] = None,
        categorias: list[str] = None,
        todos:      bool      = False,
    ) -> "ProcessadorMorfologico":
        """
        Seleciona imagens para processar. Retorna self para encadeamento.
        Ao menos um argumento deve ser fornecido.
        """
        if not self._catalogo:
            self.construir_catalogo()
        if todos:
            self._imagens_selecionadas = list(self._catalogo.values())
            return self
        selecionadas: list[ImagemInfo] = []
        if categorias:
            for cat in categorias:
                selecionadas += [
                    i for i in self._catalogo.values()
                    if cat.lower() in i.categoria.lower()
                ]
        if nomes:
            nomes_lower = [n.lower() for n in nomes]
            selecionadas += [
                i for i in self._catalogo.values()
                if i.nome.lower() in nomes_lower
            ]
        if not selecionadas:
            raise ValueError("Nenhuma imagem encontrada com os critérios fornecidos.")
        # remove duplicatas preservando ordem
        vistos = set()
        for img in selecionadas:
            if str(img.caminho) not in vistos:
                self._imagens_selecionadas.append(img)
                vistos.add(str(img.caminho))
        return self

    def selecionar_operacoes(
        self,
        operacoes: list[OperacaoMorfologica] = None,
        todas:     bool                      = False,
    ) -> "ProcessadorMorfologico":
        """Seleciona operações a aplicar. Retorna self para encadeamento."""
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
        """Reseta seleções para reutilização do objeto."""
        self._imagens_selecionadas  = []
        self._operacoes_selecionadas = []
        return self

    def imprimir_catalogo(self, categoria: str = None) -> None:
        """Exibe o catálogo completo ou filtrado por categoria."""
        self.listar_imagens(categoria=categoria)

    # ------------------------------------------------------------------
    # Pipeline principal
    # ------------------------------------------------------------------

    def processar(self) -> dict:
        """
        Executa o pipeline completo para todas as imagens e operações selecionadas.
        Retorna o dict de resultados.
        """
        self._validar_selecao()
        self._resultados = {}

        for info in self._imagens_selecionadas:
            print(f"\n{'='*60}")
            print(f"Processando: {info.categoria}/{info.nome}.{info.formato}")
            print("=" * 60)

            img = self._carregar_imagem(info)
            if img is None:
                print(f"  [AVISO] Não foi possível carregar {info.caminho}. Pulando.")
                continue

            binaria = self._binarizar(img)
            ruidosa = self._adicionar_ruido(binaria)

            chave = f"{info.nome}_{info.formato}"
            self._resultados[chave] = {
                "info":     info,
                "original": img,
                "binaria":  binaria,
                "ruidosa":  ruidosa,
            }

            for ksize in self.tamanhos_kernel:
                kernel = self._criar_kernel(ksize)
                chave_k = f"kernel_{ksize}x{ksize}"
                self._resultados[chave][chave_k] = {}

                for op in self._operacoes_selecionadas:
                    resultado = self._MAPA_OPERACOES[op](ruidosa, kernel)
                    metricas  = self._calcular_metricas(binaria, resultado)
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
    # Visualização
    # ------------------------------------------------------------------

    def visualizar(
        self,
        imagem_nome: str  = None,
        analise:     str  = None,
        salvar:      bool = True,
        mostrar:     bool = False,
    ) -> None:
        """
        Gera figura de comparação por imagem.
        Linha 0: original | binarizada | ruidosa
        Linhas seguintes: resultados por kernel × operação.

        analise: nome do subdiretório dentro de diretorio_saida (ex: "fingerprint_todas").
                 Se None, salva direto em diretorio_saida.
        """
        if not self._resultados:
            raise RuntimeError("Execute processar() antes de visualizar().")

        pasta_saida = self.diretorio_saida / analise if analise else self.diretorio_saida
        pasta_saida.mkdir(parents=True, exist_ok=True)

        alvos = (
            {k: v for k, v in self._resultados.items() if imagem_nome in k}
            if imagem_nome
            else self._resultados
        )

        for chave, dados in alvos.items():
            info = dados["info"]
            kernels = [k for k in dados if k.startswith("kernel_")]
            n_ops   = len(self._operacoes_selecionadas)
            n_cols  = max(3, n_ops)
            n_rows  = 1 + len(kernels)

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
            axes = np.array(axes).reshape(n_rows, n_cols)

            fig.suptitle(
                f"{info.categoria} — {info.nome}.{info.formato}",
                fontsize=14, fontweight="bold",
            )

            # Linha 0: pipeline de entrada
            for ax in axes[0]:
                ax.axis("off")
            self._mostrar(axes[0, 0], dados["original"], "Original (Cinza)")
            self._mostrar(axes[0, 1], dados["binaria"],  "Binarizada (Otsu)")
            self._mostrar(axes[0, 2], dados["ruidosa"],  "Com Ruído (Sal-e-Pimenta)")

            # Linhas 1..N: resultados por kernel
            for row, chave_k in enumerate(kernels, start=1):
                ksize = chave_k.split("_")[1]
                for col, op in enumerate(self._operacoes_selecionadas):
                    ax = axes[row, col] if col < n_cols else None
                    if ax is None:
                        continue
                    entry  = dados[chave_k].get(op.value)
                    if entry is None:
                        ax.axis("off")
                        continue
                    m = entry["metricas"]
                    titulo = (
                        f"{op.value.capitalize()} {ksize}\n"
                        f"IoU={m['jaccard']:.3f}  F1={m['f1']:.3f}\n"
                        f"Acc={m['pixel_accuracy']:.3f}  SSIM={m['ssim']:.3f}"
                    )
                    self._mostrar(ax, entry["imagem"], titulo)

            # desliga eixos extras
            for row in range(1, n_rows):
                for col in range(len(self._operacoes_selecionadas), n_cols):
                    axes[row, col].axis("off")

            plt.tight_layout()

            if salvar:
                caminho_fig = pasta_saida / f"{chave}_comparacao.png"
                self._salvar_figura(fig, caminho_fig)
                print(f"  Figura salva: {caminho_fig}")
            if mostrar:
                plt.show()
            plt.close(fig)

    # ------------------------------------------------------------------
    # Relatório de métricas
    # ------------------------------------------------------------------

    def gerar_relatorio_metricas(self, analise: str = None) -> None:
        """
        Imprime tabela de métricas para todas as imagens e operações processadas.

        analise: rótulo opcional exibido no cabeçalho do relatório.
        """
        if not self._resultados:
            raise RuntimeError("Execute processar() antes de gerar_relatorio_metricas().")

        cabecalho = f"{'Imagem':<22} {'Kernel':<10} {'Operação':<12} {'Acc':>6} {'IoU':>6} {'F1':>6} {'SSIM':>6}"
        separador = "-" * len(cabecalho)
        titulo = f"RELATÓRIO DE MÉTRICAS" + (f" — {analise}" if analise else "")
        print(f"\n{separador}")
        print(titulo)
        print(separador)
        print(cabecalho)
        print(separador)

        for chave, dados in self._resultados.items():
            kernels = [k for k in dados if k.startswith("kernel_")]
            for chave_k in kernels:
                ksize = chave_k.split("_")[1]
                for op_nome, entry in dados[chave_k].items():
                    m = entry["metricas"]
                    print(
                        f"{chave:<22} {ksize:<10} {op_nome:<12} "
                        f"{m['pixel_accuracy']:>6.4f} "
                        f"{m['jaccard']:>6.4f} "
                        f"{m['f1']:>6.4f} "
                        f"{m['ssim']:>6.4f}"
                    )

        print(separador)
        print("\nDISCUSSÃO:")
        print(
            "  - Erosão: reduz objetos brancos, remove ruído de sal (pixels brancos isolados),\n"
            "    mas pode erodir estruturas finas como cristas de impressões digitais.\n"
            "  - Dilatação: expande objetos brancos, preenche buracos (ruído de pimenta),\n"
            "    mas pode fundir estruturas próximas.\n"
            "  - Abertura (erosão + dilatação): remove ruído de sal preservando a forma geral;\n"
            "    ideal para eliminar pequenos artefatos sem alterar as estruturas principais.\n"
            "  - Fechamento (dilatação + erosão): preenche lacunas internas e conecta regiões\n"
            "    próximas; eficaz contra ruído de pimenta.\n"
            "  - Kernels maiores produzem efeito mais forte, podendo melhorar métricas de\n"
            "    denoising mas prejudicar a preservação de detalhes (queda no SSIM/F1).\n"
        )

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _validar_selecao(self) -> None:
        if not self._imagens_selecionadas:
            raise RuntimeError("Nenhuma imagem selecionada. Use selecionar_imagens() primeiro.")
        if not self._operacoes_selecionadas:
            raise RuntimeError("Nenhuma operação selecionada. Use selecionar_operacoes() primeiro.")

    def _carregar_imagem(self, info: ImagemInfo) -> np.ndarray | None:
        img = cv2.imread(str(info.caminho), cv2.IMREAD_GRAYSCALE)
        return img

    def _binarizar(self, img: np.ndarray) -> np.ndarray:
        _, binaria = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binaria

    def _adicionar_ruido(self, img: np.ndarray) -> np.ndarray:
        if self.ruido == TipoRuido.SALT_PEPPER:
            return self._ruido_salt_pepper(img)
        return self._ruido_gaussiano(img)

    def _ruido_salt_pepper(self, img: np.ndarray) -> np.ndarray:
        rng = np.random.RandomState(self.seed)
        ruidosa = img.copy()
        h, w = img.shape
        n = int(self.intensidade_ruido * h * w)
        # sal (branco)
        ys, xs = rng.randint(0, h, n), rng.randint(0, w, n)
        ruidosa[ys, xs] = 255
        # pimenta (preto)
        ys, xs = rng.randint(0, h, n), rng.randint(0, w, n)
        ruidosa[ys, xs] = 0
        return ruidosa

    def _ruido_gaussiano(self, img: np.ndarray) -> np.ndarray:
        rng = np.random.RandomState(self.seed)
        std = self.intensidade_ruido * 255
        ruido = rng.normal(0, std, img.shape).astype(np.float32)
        ruidosa = np.clip(img.astype(np.float32) + ruido, 0, 255).astype(np.uint8)
        return ruidosa

    def _criar_kernel(self, tamanho: int) -> np.ndarray:
        return cv2.getStructuringElement(cv2.MORPH_RECT, (tamanho, tamanho))

    def _calcular_metricas(
        self,
        referencia: np.ndarray,
        resultado:  np.ndarray,
    ) -> dict:
        ref_flat = (referencia.flatten() / 255).astype(int)
        res_flat = (resultado.flatten()  / 255).astype(int)

        pixel_acc = float(np.mean(ref_flat == res_flat))
        jaccard   = float(jaccard_score(ref_flat, res_flat, zero_division=0))
        f1        = float(f1_score(ref_flat, res_flat, zero_division=0))
        ssim_val  = float(ssim(referencia, resultado, data_range=255))

        return {
            "pixel_accuracy": pixel_acc,
            "jaccard":        jaccard,
            "f1":             f1,
            "ssim":           ssim_val,
        }

    @staticmethod
    def _mostrar(ax: plt.Axes, img: np.ndarray, titulo: str) -> None:
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(titulo, fontsize=8)
        ax.axis("off")

    @staticmethod
    def _salvar_figura(fig: plt.Figure, caminho: Path) -> None:
        fig.savefig(str(caminho), dpi=120, bbox_inches="tight")


# preenche o mapa de operações (feito aqui para ficar próximo à classe)
ProcessadorMorfologico._MAPA_OPERACOES = {
    OperacaoMorfologica.EROSAO:     lambda img, k: cv2.erode(img, k),
    OperacaoMorfologica.DILATACAO:  lambda img, k: cv2.dilate(img, k),
    OperacaoMorfologica.ABERTURA:   lambda img, k: cv2.morphologyEx(img, cv2.MORPH_OPEN,  k),
    OperacaoMorfologica.FECHAMENTO: lambda img, k: cv2.morphologyEx(img, cv2.MORPH_CLOSE, k),
}


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Caso A: experimento principal do trabalho
    #   Fingerprint — todas as operações — kernels 3x3, 5x5, 7x7
    # ------------------------------------------------------------------
    print("\n### CASO A: Fingerprint — todas as operações ###")
    p = ProcessadorMorfologico(tamanhos_kernel=[3, 5, 7])
    (
        p.selecionar_imagens(categorias=["Fingerprint"])
         .selecionar_operacoes(todas=True)
    )
    p.processar()
    p.visualizar(analise="fingerprint_todas_operacoes", salvar=True)
    p.gerar_relatorio_metricas(analise="fingerprint_todas_operacoes")

    # ------------------------------------------------------------------
    # Caso B: imagens médicas — só abertura e fechamento
    # ------------------------------------------------------------------
    print("\n### CASO B: Médicas — abertura e fechamento ###")
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

    # ------------------------------------------------------------------
    # Caso C: imagem clássica isolada para demonstração rápida
    # ------------------------------------------------------------------
    print("\n### CASO C: Clássica (cameraman) — erosão ###")
    p2 = ProcessadorMorfologico(tamanhos_kernel=[3])
    (
        p2.selecionar_imagens(nomes=["cameraman"])
          .selecionar_operacoes(operacoes=[OperacaoMorfologica.EROSAO])
    )
    p2.processar()
    p2.visualizar(analise="classica_erosao", salvar=True)

    print(f"\nConcluído. Resultados em: ./resultados/")
