import asyncio
from dotenv import load_dotenv

# Importaciones de la Pre-entrega 1
from schemas import ChatMessage, ModelConfig
from clients import AsyncLLMManager

# Importación de la Pre-entrega 2 (Cadena LCEL)
from chain import process_text

load_dotenv()


# =============================================================
#       (Cliente LLM Unificado)
# =============================================================
async def probar_entrega_1():
    """Ejecuta la prueba de clientes unificados (OpenAI y Anthropic en modo normal y streaming)."""
    prompt = [ChatMessage(role="user", content="¿Qué es la entropía? Responde en 2 oraciones.")]
    
    # --- PRUEBA OPENAI ---
    print("========================================")
    print("PROBANDO OPENAI (ENTREGA 1)")
    print("========================================")
    try:
        openai_config = ModelConfig(model_name="gpt-4o-mini", temperature=0.5, max_tokens=150)
        openai_manager = AsyncLLMManager(provider="openai")

        print("\n--- Modo Normal ---")
        response_openai = await openai_manager.generate(prompt, openai_config)
        print(f"Respuesta ({response_openai.provider} - {response_openai.model}):")
        print(response_openai.content)

        print("\n--- Modo Streaming ---")
        print("Respuesta: ", end="", flush=True)
        async for chunk in openai_manager.generate_stream(prompt, openai_config):
            print(chunk, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"Error en OpenAI: {e}\n")

    # --- PRUEBA ANTHROPIC ---
    print("========================================")
    print("PROBANDO ANTHROPIC (ENTREGA 1)")
    print("========================================")
    try:
        anthropic_config = ModelConfig(model_name="claude-3-5-haiku-20241022", temperature=0.5, max_tokens=150)
        anthropic_manager = AsyncLLMManager(provider="anthropic")

        print("\n--- Modo Normal ---")
        response_anthropic = await anthropic_manager.generate(prompt, anthropic_config)
        print(f"Respuesta ({response_anthropic.provider} - {response_anthropic.model}):")
        print(response_anthropic.content)

        print("\n--- Modo Streaming ---")
        print("Respuesta: ", end="", flush=True)
        async for chunk in anthropic_manager.generate_stream(prompt, anthropic_config):
            print(chunk, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"Error en Anthropic: {e}\n")


# =============================================================
#       (Pipeline LCEL con Pydantic y Retry)
# =============================================================
async def probar_entrega_2():
    """Ejecuta el pipeline de extracción de entidades técnicas con validación y resiliencia."""
    texto_tecnico = """
    Nuestra API en FastAPI está devolviendo timeouts intermitentes. El caché en Redis
    parece saturarse en picos de tráfico y las conexiones a PostgreSQL se agotan
    porque el pool está mal dimensionado. Esto está afectando a usuarios en producción.
    """
    
    texto_ambiguo = """
    El sistema anda medio raro últimamente, parece que cuando entra mucha gente
    se cae una de las bases pero no estoy muy seguro de qué versión estamos usando.
    """

    print("========================================")
    print("PROBANDO PIPELINE LCEL (ENTREGA 2)")
    print("========================================")
    
    # 1. Caso Normal: Prueba con log técnico estructurado
    for provider in ["openai", "anthropic"]:
        print(f"\n--- Procesando con: {provider.upper()} ---")
        try:
            resultado = await process_text(texto_tecnico, provider=provider)
            print(resultado.model_dump_json(indent=2))
        except Exception as e:
            print(f"Error procesando con {provider}: {e}")

    # 2. Prueba de Estrés: Texto ambiguo para verificar respuesta y fallback
    print("\n--- Prueba de Estrés (Texto ambiguo) ---")
    try:
        resultado_ambiguo = await process_text(texto_ambiguo, provider="openai")
        print(resultado_ambiguo.model_dump_json(indent=2))
    except Exception as e:
        print(f"Error en prueba de estrés: {e}")


 
async def main():
    # Ejecuta por defecto la entrega actual
    await probar_entrega_2()

    # Si deseas volver a probar la entrega 1, simplemente descomentá la siguiente línea:
    # await probar_entrega_1()


if __name__ == "__main__":
    asyncio.run(main())