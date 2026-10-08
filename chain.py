import os
import logging
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from schemas import EntidadesTecnicas


load_dotenv()


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("pipeline_extraccion")


prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Sos un ingeniero especialista en arquitectura de software y análisis de logs. "
     "Tu tarea es analizar el texto provisto y extraer las entidades técnicas con precisión. "
     "Identificá tecnologías mencionadas, evaluá la criticidad (baja, media, alta) "
     "y genera un resumen técnico breve."),
    ("human", "{texto}")
])


def get_model(provider: str = "openai"):
    """Fábrica para instanciar el cliente con temperature=0 para extracción determinista."""
    if provider == "openai":
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    elif provider == "anthropic":
        return ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0)
    else:
        raise ValueError(f"Proveedor no soportado: {provider}")


def build_chain(provider: str = "openai"):
    """
    Construye la Cadena LCEL: prompt | model.with_structured_output(Schema) + .with_retry()
    """
    model = get_model(provider)
    
    
    structured_model = model.with_structured_output(EntidadesTecnicas)
    
    
    chain = (prompt | structured_model).with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=True,
    )
    return chain


async def process_text(text: str, provider: str = "openai") -> EntidadesTecnicas:
    """
    Función asíncrona principal requerida. Ejecuta la cadena usando .ainvoke() e incluye logs del proceso.
    """
    chain = build_chain(provider)
    logger.info(f"[{provider.upper()}] Ejecutando cadena asíncrona mediante .ainvoke() ({len(text)} caracteres)...")

    try:
        resultado = await chain.ainvoke({"texto": text})
        logger.info(f"[{provider.upper()}] ✅ Validación Pydantic exitosa.")
        return resultado
    except Exception as e:
        logger.error(f"[{provider.upper()}] ❌ Falló tras agotar reintentos (.with_retry): {e}")
        raise
