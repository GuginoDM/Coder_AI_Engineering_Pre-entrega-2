# Coder_AI_Engineering_Pre-entrega-2

Pre-entrega 2: Pipeline de procesamiento validado
Qué construir
Debes desarrollar un Pipeline de Extracción de Entidades Técnicas. El sistema debe recibir un párrafo de texto sin procesar (por ejemplo, una descripción de arquitectura de software o un log de error) y devolver un objeto validado.

Los componentes requeridos:
Esquema Pydantic: Un modelo que defina campos como tecnologias (lista de strings), nivel_de_criticidad (enum: baja, media, alta), y resumen_tecnico (string).
Prompt Template: Un template modular que acepte el texto de entrada y las instrucciones de formato.
Cadena LCEL: Una composición que una el Prompt + LLM + Output Parser.
Lógica de Resiliencia: Configuración de al menos un reintento automático si el LLM devuelve un JSON mal formado o incompleto.
Pasos sugeridos
Define tu Contrato: Crea la clase Pydantic. Piensa en qué restricciones quieres poner (ej. que la lista de tecnologías no esté vacía).
Prepara el Parser: Utiliza PydanticOutputParser o el método .with_structured_output() de LangChain (preferido para OpenAI/Anthropic).
Ensambla la Cadena:
python


chain = prompt | model.with_structured_output(TuEsquema)
Añade Resiliencia: Envuelve la llamada con una estrategia de reintento (.with_retry()).
Prueba de Estrés: Pasa un texto ambiguo y verifica si el validador lanza excepciones o si el modelo se recupera.
Errores comunes a evitar
Ignorar el finish_reason: A veces el LLM corta la respuesta por falta de tokens. Tu pipeline debe detectar si el objeto está incompleto antes de intentar transformarlo.
Hardcoding de Prompts: Evita las F-strings de Python dentro de la cadena. Usa ChatPromptTemplate para mantener la modularidad y permitir que LangChain gestione las variables de entrada.
​
