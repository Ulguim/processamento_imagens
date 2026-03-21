from __future__ import annotations

from pathlib import Path

from .modelos import ImagemInfo

_FORMATOS_SUPORTADOS = {".bmp", ".pgm", ".ppm", ".png", ".jpg", ".jpeg"}


class CatalogoImagens:
    def __init__(self, diretorio: str | Path) -> None:
        self.diretorio = Path(diretorio)
        self._itens: dict[str, ImagemInfo] = {}

    def construir(self) -> "CatalogoImagens":
        """Escaneia o diretório recursivamente e popula o catálogo."""
        self._itens = {}
        for caminho in sorted(self.diretorio.rglob("*")):
            if caminho.is_file() and caminho.suffix.lower() in _FORMATOS_SUPORTADOS:
                categoria = caminho.parent.name
                chave = f"{categoria}/{caminho.name}"
                self._itens[chave] = ImagemInfo(
                    nome=caminho.stem,
                    caminho=caminho,
                    categoria=categoria,
                    formato=caminho.suffix.lstrip(".").lower(),
                )
        return self

    def listar(self, categoria: str = None, formato: str = None) -> list[ImagemInfo]:
        """Retorna lista filtrada. Imprime cada item encontrado."""
        if not self._itens:
            self.construir()
        itens = list(self._itens.values())
        if categoria:
            itens = [i for i in itens if categoria.lower() in i.categoria.lower()]
        if formato:
            itens = [i for i in itens if i.formato == formato.lower().lstrip(".")]
        for img in itens:
            print(f"  [{img.categoria}] {img.nome}.{img.formato}")
        return itens

    def selecionar(
        self,
        nomes:      list[str] = None,
        categorias: list[str] = None,
        todos:      bool      = False,
    ) -> list[ImagemInfo]:
        """Retorna imagens selecionadas sem duplicatas, preservando ordem."""
        if not self._itens:
            self.construir()
        if todos:
            return list(self._itens.values())

        selecionadas: list[ImagemInfo] = []
        if categorias:
            for cat in categorias:
                selecionadas += [
                    i for i in self._itens.values()
                    if cat.lower() in i.categoria.lower()
                ]
        if nomes:
            nomes_lower = [n.lower() for n in nomes]
            selecionadas += [
                i for i in self._itens.values()
                if i.nome.lower() in nomes_lower
            ]
        if not selecionadas:
            raise ValueError("Nenhuma imagem encontrada com os critérios fornecidos.")

        vistos: set[str] = set()
        resultado: list[ImagemInfo] = []
        for img in selecionadas:
            key = str(img.caminho)
            if key not in vistos:
                resultado.append(img)
                vistos.add(key)
        return resultado
