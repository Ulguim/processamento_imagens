from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .modelos import OperacaoMorfologica


def visualizar(
    resultados:  dict,
    operacoes:   list[OperacaoMorfologica],
    pasta_saida: Path,
    imagem_nome: str  = None,
    mostrar:     bool = False,
    salvar:      bool = True,
) -> None:
    """
    Gera grade de comparação por imagem e salva em pasta_saida.
    Linha 0: original | binarizada | ruidosa
    Linhas seguintes: resultado por kernel × operação (com métricas no título).
    """
    pasta_saida.mkdir(parents=True, exist_ok=True)

    alvos = (
        {k: v for k, v in resultados.items() if imagem_nome in k}
        if imagem_nome else resultados
    )

    for chave, dados in alvos.items():
        info    = dados["info"]
        kernels = [k for k in dados if k.startswith("kernel_")]
        n_cols  = max(3, len(operacoes))
        n_rows  = 1 + len(kernels)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = np.array(axes).reshape(n_rows, n_cols)

        fig.suptitle(
            f"{info.categoria} — {info.nome}.{info.formato}",
            fontsize=14, fontweight="bold",
        )

        for ax in axes[0]:
            ax.axis("off")
        _mostrar(axes[0, 0], dados["original"], "Original (Cinza)")
        _mostrar(axes[0, 1], dados["binaria"],  "Binarizada (Otsu)")
        _mostrar(axes[0, 2], dados["ruidosa"],  "Com Ruído (Sal-e-Pimenta)")

        for row, chave_k in enumerate(kernels, start=1):
            ksize = chave_k.split("_")[1]
            for col, op in enumerate(operacoes):
                ax    = axes[row, col] if col < n_cols else None
                entry = dados[chave_k].get(op.value) if ax is not None else None
                if entry is None:
                    if ax is not None:
                        ax.axis("off")
                    continue
                m = entry["metricas"]
                titulo = (
                    f"{op.value.capitalize()} {ksize}\n"
                    f"IoU={m['jaccard']:.3f}  F1={m['f1']:.3f}\n"
                    f"Acc={m['pixel_accuracy']:.3f}  SSIM={m['ssim']:.3f}"
                )
                _mostrar(ax, entry["imagem"], titulo)

        for row in range(1, n_rows):
            for col in range(len(operacoes), n_cols):
                axes[row, col].axis("off")

        plt.tight_layout()

        if salvar:
            caminho = pasta_saida / f"{chave}_comparacao.png"
            fig.savefig(str(caminho), dpi=120, bbox_inches="tight")
            print(f"  Figura salva: {caminho}")
        if mostrar:
            plt.show()
        plt.close(fig)


def gerar_relatorio(resultados: dict, analise: str = None) -> None:
    """Imprime tabela de métricas e discussão acadêmica no terminal."""
    cabecalho = (
        f"{'Imagem':<22} {'Kernel':<10} {'Operação':<12} "
        f"{'Acc':>6} {'IoU':>6} {'F1':>6} {'SSIM':>6}"
    )
    separador = "-" * len(cabecalho)
    titulo    = "RELATÓRIO DE MÉTRICAS" + (f" — {analise}" if analise else "")

    print(f"\n{separador}")
    print(titulo)
    print(separador)
    print(cabecalho)
    print(separador)

    for chave, dados in resultados.items():
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
    print(
        "\nDISCUSSÃO:\n"
        "  - Erosão: remove ruído de sal (pixels brancos isolados), mas erode estruturas finas.\n"
        "  - Dilatação: preenche ruído de pimenta (pixels pretos isolados), mas expande bordas.\n"
        "  - Abertura (erosão + dilatação): elimina artefatos pequenos sem alterar forma geral.\n"
        "  - Fechamento (dilatação + erosão): preenche lacunas internas e conecta regiões próximas.\n"
        "  - Kernels maiores: efeito mais forte — pode melhorar denoising mas reduz detalhes.\n"
    )


def _mostrar(ax: plt.Axes, img, titulo: str) -> None:
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title(titulo, fontsize=8)
    ax.axis("off")
