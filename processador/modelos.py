from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class OperacaoMorfologica(Enum):
    EROSAO     = "erosao"
    DILATACAO  = "dilatacao"
    ABERTURA   = "abertura"
    FECHAMENTO = "fechamento"


class TipoRuido(Enum):
    SALT_PEPPER = "salt_pepper"
    GAUSSIANO   = "gaussiano"


@dataclass
class ImagemInfo:
    nome:      str
    caminho:   Path
    categoria: str
    formato:   str
