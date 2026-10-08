from enum import Enum
from typing import List, Literal
from pydantic import BaseModel, Field, field_validator

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ModelConfig(BaseModel):
    model_name: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, gt=0)

class ModelResponse(BaseModel):
    content: str
    provider: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

# =============================================================
# MODELOS PRE-ENTREGA 2 (Agregar a partir de la línea 18)
# =============================================================

class NivelCriticidad(str, Enum):
    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class EntidadesTecnicas(BaseModel):
    tecnologias: List[str] = Field(
        ...,
        min_length=1,
        description="Lista de tecnologías, frameworks, bases de datos o herramientas identificadas en el texto."
    )
    nivel_de_criticidad: NivelCriticidad = Field(
        ...,
        description="Gravedad del problema expuesto o criticidad del componente arquitectónico."
    )
    resumen_tecnico: str = Field(
        ...,
        min_length=10,
        description="Resumen técnico conciso (1 a 2 oraciones) sobre el contenido analizado."
    )

    @field_validator("tecnologias")
    @classmethod
    def validar_tecnologias(cls, v: List[str]) -> List[str]:
        limpio = [t.strip() for t in v if t.strip()]
        if not limpio:
            raise ValueError("La lista de tecnologías no puede quedar vacía tras la limpieza.")
        return list(dict.fromkeys(limpio))