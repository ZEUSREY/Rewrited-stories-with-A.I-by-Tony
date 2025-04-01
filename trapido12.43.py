import os
import sys
import time
import logging
import gc
import re
import json
import numpy as np
import signal
import torch
import requests
import csv
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm

# -----------------------------------------------------------
# CONFIGURACIÓN GENERAL (CUDA, Logging, etc.)
# -----------------------------------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.6,expandable_segments:True"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler("historia_reescrita.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def flush_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# -----------------------------------------------------------
# MENÚ DE SELECCIÓN DE SUPERVISOR (Doble opción: Allenai y BigBird)
# -----------------------------------------------------------
SUPERVISOR_OPTIONS = {
    "1": {"name": "allenai/longformer-base-4096", "description": "Allenai Longformer (4096 tokens)"},
    "2": {"name": "google/bigbird-roberta-base", "description": "BigBird Roberta Base (4096 tokens)"}
}

def seleccionar_modelo_supervisor():
    print("Elige el modelo supervisor para evaluar la perplejidad:")
    for key, info in SUPERVISOR_OPTIONS.items():
        print(f"{key}. {info['description']}")
    opcion = input("Ingresa el número del modelo (por defecto: 1 - Allenai Longformer): ").strip()
    if opcion not in SUPERVISOR_OPTIONS or opcion == "":
        print("Opción inválida o vacía. Se usará Allenai Longformer por defecto.")
        opcion = "1"
    return SUPERVISOR_OPTIONS[opcion]["name"]

CURRENT_SUPERVISOR_MODEL_NAME = seleccionar_modelo_supervisor()
logger.info(f"Supervisor seleccionado: {CURRENT_SUPERVISOR_MODEL_NAME}")
CURRENT_SUPERVISOR_TOKENIZER = AutoTokenizer.from_pretrained(CURRENT_SUPERVISOR_MODEL_NAME)
CURRENT_SUPERVISOR_MODEL = AutoModelForMaskedLM.from_pretrained(CURRENT_SUPERVISOR_MODEL_NAME)
model_max_length = CURRENT_SUPERVISOR_TOKENIZER.model_max_length
if model_max_length < 1000 or model_max_length > 100000:
    model_max_length = 4096
CURRENT_SUPERVISOR_TOKENIZER.model_max_length = model_max_length
logger.info(f"SUPERVISOR_TOKENIZER.model_max_length ajustado a {model_max_length} tokens.")

# -----------------------------------------------------------
# PARÁMETROS GLOBALES Y CONFIGURACIÓN DEL SISTEMA
# -----------------------------------------------------------
MODEL_SOURCE = "local"  # Opciones: "local" (MSTY), "qwen", "ollama"
LOCAL_MODEL_NAME = "vicgalle-Configurable-Janus-7B-Q4_K_S-1741902058230:latest"
DASHSCOPE_API_KEY = ""
DASHSCOPE_API_URL = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
API_SERVICE = "msty"

INPUT_FILE = "historia.txt"
OUTPUT_FILE = "historia_reescrita_final.txt"

# Parámetros para la división en partes y generación
CONTENT_SIZE = 3500      # Caracteres de contenido real por bloque
SUMMARY_SIZE = 500       # Caracteres para resumen del bloque anterior
FRAGMENTACION_MODO = "escenas"
CHUNK_SIZE = 3000
OVERLAP_SIZE = 300       # CHUNK_SIZE > OVERLAP_SIZE
TOP_K = 5
MIN_CONTEXT_SIM = 0.1
SIM_THRESHOLD = 0.70
MAX_ATTEMPTS = 3
BATCH_SIZE = 1024
EMBEDDING_MINI_BATCH = 4
MAX_WORKERS = 6
MEMORY_THRESHOLD_MB = 100
EMBEDDING_MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"
MAX_GROUP_LENGTH = 2000  # Máximo de caracteres por grupo

# Actualizamos el número de frases a agrupar para supervisión
PHRASE_GROUP_COUNT = 17  # Se agruparán 17 frases para evaluación
# Umbral para la supervisión de bloque (más flexible para bloques largos)
BLOCK_SUPERVISION_THRESHOLD = 150

LISTA_MODELOS = [
    "vicgalle-Configurable-Janus-7B-Q4_K_S-1741902058230:latest",
    "Configurable-Janus-7B-Q4_K_M-1741902233387:latest",
    "vicgalle-Configurable-Janus-7B-Q4_K_S-1741902058230:latest",
    "qwen:14b-chat-v1.5-q2_K",
    "llama2:13b",
    "llama3:latest"
]
EVALUATOR_MODELOS = [
    "oh-dcft-v3.1-claude-3-5-sonnet-20241022.Q5_K_M-1740593547298:latest"
]

# -----------------------------------------------------------
# PROMPTS FINALES CON DIRECTRICES (Primera Persona y Narrativa Clara)
# -----------------------------------------------------------
PROMPT_FINAL_ES = """
[Consejos y Directrices Generales]
Contexto General:
La historia abarca abusos escolares, venganzas, batallas personales, conflictos familiares y episodios laborales.
La re-narración debe preservar al menos el 70% del contenido esencial del original (entre 70 000 y 80 000 caracteres) y reflejar la riqueza emocional y la diversidad de episodios, pero de forma clara y sencilla.

Estructura y División:
- Divide la narración en 22 partes (o según la longitud necesaria).
- Cada parte debe iniciarse con “Parte X de Y – [Título descriptivo]” y terminar con “Fin de la Parte X”.
- La narrativa debe ser completa y no un resumen; se deben enlazar los episodios de forma coherente.

Tono y Estilo:
- Combina la crudeza de los hechos con la ironía y la reflexión personal.
- Usa un lenguaje claro, sencillo y descriptivo, respetando la complejidad de los escenarios.
- **Narra la historia en primera persona, como si estos hechos te hubieran sucedido a ti.**
- No inventes nuevos episodios; re-narra el contenido original, simplificando sin perder los detalles esenciales.

Detalles y Transiciones:
- Incluye descripciones detalladas de ambientes, sensaciones y emociones.
- Desarrolla diálogos, acciones y reflexiones para capturar la densidad del relato, pero de forma directa.
- Al inicio de cada parte, incluye un breve resumen de la parte anterior y anuncia lo que se abordará en la parte actual.

Formato estándar:
Parte X de Y – [Título descriptivo]
[Breve conexión con la parte anterior]
[Cuerpo narrativo desarrollado en primera persona, claro y directo]
Fin de la Parte X.

Texto original:
{texto}

Genera la re-narración completa y detallada:
"""

PROMPT_FINAL_EN = """
[General Guidelines]
General Context:
The story covers school abuse, revenge, personal battles, family conflicts, and work-related episodes.
The rewriting must preserve at least 70% of the essential content of the original (between 70,000 and 80,000 characters)
and reflect the emotional richness and diversity of the episodes, but in a clear and simple manner.

Structure and Division:
- Divide the narrative into 22 parts (or as many as necessary based on length).
- Each part must begin with "Part X of Y – [Descriptive Title]" and end with "End of Part X."
- The narrative should be complete and not a summary; the episodes must be coherently linked.

Tone and Style:
- Combine harsh facts with irony and reflective insight.
- Use clear, simple, and descriptive language, respecting the complexity of different settings.
- **Narrate the story in the first person, as if these events happened to you.**
- Do not invent new episodes; re-narrate the original content, simplifying while preserving essential details.

Details and Transitions:
- Include detailed descriptions of environments, feelings, and emotions.
- Develop dialogues, actions, and reflections to capture the richness of the story, in a direct manner.
- At the beginning of each part, include a brief summary of the previous part and state what will be covered next.

Standard Format:
Part X of Y – [Descriptive Title]
[Brief connection with the previous part]
[Developed narrative body in the first person]
End of Part X.
Original text:
{texto}
Generate the complete and detailed rewriting:
"""

def seleccionar_prompt(text):
    try:
        import langdetect
        lang = langdetect.detect(text)
    except Exception as e:
        logger.error(f"Error en detección de idioma: {e}")
        lang = 'es'
    return PROMPT_FINAL_ES if lang == 'es' else PROMPT_FINAL_EN

def generar_header_prompt(parte_num, total_partes, resumen_anterior, bloque_actual):
    header = (
        f"Parte {parte_num} de {total_partes} – [Título descriptivo]\n\n"
        f"Recuerda: narra en primera persona y no inventes nuevos episodios. En la Parte {parte_num - 1} se abordaron: {resumen_anterior}\n\n"
        f"Texto original:\n{bloque_actual}\n\n"
        "Genera la re-narración completa y detallada:"
    )
    return header

# -----------------------------------------------------------
# FUNCIONES DE FRAGMENTACIÓN Y PREPROCESAMIENTO
# -----------------------------------------------------------
def dividir_en_chunks(texto, chunk_size=CHUNK_SIZE, overlap=OVERLAP_SIZE):
    if chunk_size <= overlap:
        logger.warning("CHUNK_SIZE debe ser mayor que OVERLAP_SIZE. Ajustando OVERLAP_SIZE a CHUNK_SIZE//2.")
        overlap = chunk_size // 2
    chunks = []
    start = 0
    while start < len(texto):
        end = min(start + chunk_size, len(texto))
        chunks.append(texto[start:end])
        if end == len(texto):
            break
        start = end - overlap
    return chunks

def dividir_texto(texto, modo=FRAGMENTACION_MODO):
    if modo == "escenas":
        escenas = texto.split("\n\n")
        if len(escenas) == 1 or any(len(escena) > CHUNK_SIZE for escena in escenas):
            logger.info("El texto tiene pocas separaciones o escenas muy largas; usando chunking con ventana deslizante...")
            escenas = dividir_en_chunks(texto, CHUNK_SIZE, OVERLAP_SIZE)
        return escenas
    elif modo == "parrafos":
        return texto.split("\n")
    elif modo == "oraciones":
        return re.split('[.!?]+', texto)
    else:
        return texto.split("\n\n")

def fusionar_fragmentos(escenas, min_length=50):
    escenas_fusionadas = []
    buffer_escena = ""
    for escena in escenas:
        escena = escena.strip()
        if len(escena) < min_length or not re.search(r"[.,;:!?]", escena):
            buffer_escena += " " + escena
        else:
            if buffer_escena:
                escena = (buffer_escena + " " + escena).strip()
                buffer_escena = ""
            escenas_fusionadas.append(escena)
    if buffer_escena and escenas_fusionadas:
        escenas_fusionadas[-1] += " " + buffer_escena.strip()
    return escenas_fusionadas

def validar_escena(escena):
    return len(escena) >= 50 and bool(re.search(r"[.,;:!?]", escena))

def preprocess_spanish_text(text):
    text = re.sub('[\u4e00-\u9fff]', '', text)
    text = re.sub(r'(?m)^.*翻译.*$', '', text)
    return text.strip()

def load_russian_text(filename):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return f.read().strip()
    else:
        logger.error(f"No se encontró el archivo ruso '{filename}'")
        return ""

def check_language(text, expected_lang='es'):
    try:
        import langdetect
        return langdetect.detect(text) == expected_lang
    except:
        return False

def contains_chinese_characters(text):
    return bool(re.search('[\u4e00-\u9fff]', text))

# Función para evitar cortar oraciones a medias
def dividir_texto_por_frases(texto, max_block_size=3000):
    oraciones = re.split(r'(?<=[.!?])\s+', texto)
    bloques = []
    bloque_actual = ""
    for oracion in oraciones:
        if len(bloque_actual) + len(oracion) + 1 > max_block_size:
            # Si el bloque no termina en puntuación, se intenta mover la última oración completa al siguiente bloque
            if not re.search(r'[.!?]$', bloque_actual.strip()):
                partes = re.split(r'(?<=[.!?])\s+', bloque_actual.strip())
                if len(partes) > 1:
                    bloque_actual = " ".join(partes[:-1])
                    oracion = partes[-1] + " " + oracion
            bloques.append(bloque_actual.strip())
            bloque_actual = oracion
        else:
            bloque_actual += " " + oracion if bloque_actual else oracion
    if bloque_actual:
        bloques.append(bloque_actual.strip())
    return bloques

# -----------------------------------------------------------
# FUNCIÓN NUEVA: DETECCIÓN DE PENSAMIENTOS INTERNOS (CHAIN-OF-THOUGHT)
# -----------------------------------------------------------
def es_pensamiento_interno(texto):
    # Detecta patrones comunes de chain-of-thought
    patrones = [r"^\s*pienso", r"^\s*reflexiono", r"^\s*considero", r"^\s*me pregunto"]
    for patron in patrones:
        if re.search(patron, texto, flags=re.IGNORECASE):
            return True
    return False

# -----------------------------------------------------------
# SUPERVISOR: PERPLEJIDAD (usando el supervisor seleccionado)
# -----------------------------------------------------------
def calcular_perplejidad(frase):
    inputs = CURRENT_SUPERVISOR_TOKENIZER(frase, return_tensors="pt").to(CURRENT_SUPERVISOR_MODEL.device)
    input_ids = inputs.input_ids
    seq_length = input_ids.size(1)
    log_likelihood = 0.0
    with torch.no_grad():
        for i in range(seq_length):
            input_ids_masked = input_ids.clone()
            input_ids_masked[0, i] = CURRENT_SUPERVISOR_TOKENIZER.mask_token_id
            outputs = CURRENT_SUPERVISOR_MODEL(input_ids_masked)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits[0, i], dim=-1)
            token_id = input_ids[0, i]
            token_prob = probs[token_id]
            log_likelihood += torch.log(token_prob + 1e-10)
    perplexity = torch.exp(-log_likelihood / seq_length)
    return perplexity.item()

def supervisar_frase(frase, threshold=BLOCK_SUPERVISION_THRESHOLD):
    # Si se detecta que el bloque es un pensamiento interno, omite la supervisión
    if es_pensamiento_interno(frase):
        logger.info("Detectado pensamiento interno; omitiendo supervisión de perplejidad.")
        return True
    p = calcular_perplejidad(frase)
    logger.info(f"Perplejidad del bloque: {p:.2f}")
    return p < threshold

# -----------------------------------------------------------
# MEMORIA RAG Y MEMORIA EXTERNA (para contexto)
# -----------------------------------------------------------
external_memory = []  # Para contexto adicional
rag_memory = []       # Fragmentos generados
rag_memory_embeddings = None

def update_rag_memory(segment):
    global rag_memory, rag_memory_embeddings
    rag_memory.append(segment)
    rag_memory_embeddings = safe_encode(rag_memory)

def retrieve_context(query):
    global rag_memory, rag_memory_embeddings
    if rag_memory and rag_memory_embeddings is not None:
        try:
            query_embedding = obtener_embedding(query)
            cos_scores = util.cos_sim(query_embedding, rag_memory_embeddings)[0]
            top_results = torch.topk(cos_scores, k=min(3, len(rag_memory)))
            retrieved_lines = [rag_memory[idx] for idx in top_results[1].tolist()]
            return "\n".join(retrieved_lines)
        except Exception as e:
            logger.error(f"Error en la recuperación de contexto de rag_memory: {e}")
    return ("Información adicional relevante: la narrativa se desarrolla en un universo expansivo, "
            "con múltiples líneas temporales, personajes complejos y conflictos épicos que se interconectan.")

def augment_prompt(prompt):
    memory_context = external_memory[-1][-200:] if external_memory else ""
    retrieved = retrieve_context(prompt)
    augmented = (
        f"{prompt}\n\n[Contexto extraído]:\n{memory_context}\n\n"
        f"[Información adicional]:\n{retrieved}\n\nContinúa la re-narración sin inventar nuevos episodios:"
    )
    return augmented

# Función para cargar contexto en ruso y generar embeddings para él
def obtener_contexto_ruso():
    texto_ruso = load_russian_text("Historia_ruso.txt")
    if texto_ruso:
        escenas_ruso = dividir_texto(texto_ruso)
        escenas_ruso_fusionadas = fusionar_fragmentos(escenas_ruso)
        escenas_ruso_validas = [escena for escena in escenas_ruso_fusionadas if validar_escena(escena)]
        return escenas_ruso_validas
    return []

# -----------------------------------------------------------
# MÓDULO DE REINTENTOS
# -----------------------------------------------------------
def retry_operation(operation, max_attempts=MAX_ATTEMPTS, delay=1, allowed_exceptions=(Exception,)):
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except allowed_exceptions as e:
            logger.error(f"Intento {attempt}/{max_attempts} fallido: {e}")
            if attempt < max_attempts:
                time.sleep(delay)
            else:
                logger.error("Máximo de reintentos alcanzado.")
                raise

# -----------------------------------------------------------
# FUNCIONES DE GENERACIÓN DE TEXTO
# -----------------------------------------------------------
def generar_con_msty_con_supervision(prompt, modelo=None, max_tokens=8000, temperature=0.80):
    if not modelo:
        modelo = LOCAL_MODEL_NAME
    MSTY_API_URL = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": modelo,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    full_response = ""
    # Agrupar 17 frases para la supervisión
    for attempt in range(1, MAX_ATTEMPTS+1):
        try:
            resp = retry_operation(lambda: requests.post(MSTY_API_URL, json=data, headers=headers, stream=True, timeout=120),
                                   allowed_exceptions=(requests.exceptions.Timeout, requests.exceptions.ConnectionError, Exception))
            if resp.status_code != 200:
                logger.error(f"[LOCAL] Error en MSTY (intento {attempt}): {resp.status_code}, {resp.text}")
                continue
            buffer_frase = ""
            for line in resp.iter_lines():
                if line:
                    try:
                        obj = json.loads(line.decode("utf-8", errors='ignore'))
                        token = obj.get("response", "")
                        full_response += token
                        buffer_frase += token
                        print(token, end='', flush=True)
                        # Dividir en frases y agrupar 17 para supervisión del bloque
                        frases = re.split(r'(?<=[.!?])\s+', buffer_frase)
                        if len(frases) >= PHRASE_GROUP_COUNT:
                            bloque = " ".join(frases[:PHRASE_GROUP_COUNT])
                            if not supervisar_frase(bloque, threshold=BLOCK_SUPERVISION_THRESHOLD):
                                logger.warning("Bloque de 17 frases rechazado por el supervisor (perplejidad alta).")
                                return full_response.strip()
                            buffer_frase = " ".join(frases[PHRASE_GROUP_COUNT:])
                        if obj.get("done", False):
                            if buffer_frase and not supervisar_frase(buffer_frase, threshold=BLOCK_SUPERVISION_THRESHOLD):
                                logger.warning("Fragmento final rechazado por el supervisor (perplejidad alta).")
                                return full_response.strip()
                            logger.debug(f"[LOCAL] Respuesta final: {full_response}")
                            external_memory.append(full_response[-200:])
                            return full_response.strip()
                    except json.JSONDecodeError as e:
                        logger.error(f"Error decodificando JSON: {e}")
                        continue
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            logger.error(f"[LOCAL] Error de red en MSTY (intento {attempt}): {e}")
        except Exception as e:
            logger.error(f"[LOCAL] Error inesperado en MSTY (intento {attempt}): {e}")
    logger.warning("No se obtuvo respuesta válida de MSTY; usando texto original.")
    return None

def generar_con_msty(prompt, modelo=None, max_tokens=8000, temperature=0.85):
    return generar_con_msty_con_supervision(prompt, modelo=modelo, max_tokens=max_tokens, temperature=temperature)

def generar_con_qwen(prompt, max_tokens=8000, temperature=0.85):
    if not DASHSCOPE_API_KEY:
        logger.error("API key de Qwen no configurada.")
        return None
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DASHSCOPE_API_KEY}"}
    data = {
        "model": "qwen-turbo",
        "input": {"prompt": prompt},
        "parameters": {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.90,
            "top_k": 40,
            "result_format": "text"
        }
    }
    try:
        response = retry_operation(lambda: requests.post(DASHSCOPE_API_URL, headers=headers, json=data, timeout=30),
                                   allowed_exceptions=(requests.exceptions.Timeout, requests.exceptions.ConnectionError, Exception))
        response.raise_for_status()
        result = response.json()
        logger.debug(f"[QWEN] Respuesta API: {result}")
        if 'output' in result and 'text' in result['output']:
            return result['output']['text'].strip()
        logger.error("[QWEN] Respuesta inesperada")
        return None
    except Exception as e:
        logger.error(f"[QWEN] Error en la API: {e}")
        return None

def generar_con_ollama(prompt, modelo=None, max_tokens=8000, temperature=0.85):
    if not modelo:
        modelo = LOCAL_MODEL_NAME
    command = ["ollama", "run", modelo]
    try:
        result = retry_operation(lambda: subprocess.run(command, input=prompt, capture_output=True, text=True,
                                                         encoding="utf-8", errors="replace", timeout=60),
                                 allowed_exceptions=(subprocess.SubprocessError, Exception))
        if result.returncode != 0:
            logger.error(f"Error en Ollama: {result.stderr.strip()}")
            return None
        return result.stdout.strip()
    except Exception as e:
        logger.error(f"Error llamando a Ollama: {e}")
        return None

def generar_texto(prompt, max_tokens=8000, temperature=0.85):
    augmented_prompt = augment_prompt(prompt)
    if API_SERVICE.lower() == "qwen":
        logger.info("[QWEN] Generando con Qwen API...")
        return generar_con_qwen(augmented_prompt, max_tokens, temperature)
    elif API_SERVICE.lower() == "ollama":
        logger.info("[OLLAMA] Generando con Ollama...")
        return generar_con_ollama(augmented_prompt, modelo=LOCAL_MODEL_NAME, max_tokens=max_tokens, temperature=temperature)
    else:
        logger.info(f"[LOCAL] Generando con MSTY (Modelo={LOCAL_MODEL_NAME})...")
        return generar_con_msty(augmented_prompt, modelo=LOCAL_MODEL_NAME, max_tokens=max_tokens, temperature=temperature)

# -----------------------------------------------------------
# EMBEDDINGS Y SIMILITUD
# -----------------------------------------------------------
embedding_cache = {}

def get_embedding_cached(text, model=None):
    global embedding_cache
    if model is None:
        model = embedding_model
    if text in embedding_cache:
        return embedding_cache[text]
    emb = model.encode([text], convert_to_tensor=True)[0]
    embedding_cache[text] = emb
    return emb

def safe_encode(texts, mini_batch_size=BATCH_SIZE, threshold_mb=MEMORY_THRESHOLD_MB, model=None):
    if model is None:
        model = embedding_model
    device_type = str(model.device)
    final_dtype = torch.float16 if "cuda" in device_type else torch.float32
    embeddings_list = []
    with torch.no_grad():
        for i in range(0, len(texts), mini_batch_size):
            batch_texts = texts[i:i+mini_batch_size]
            emb = get_embedding_cached(batch_texts[0], model=model)
            embeddings_list.append(emb.unsqueeze(0))
    flush_gpu()
    return torch.cat(embeddings_list, dim=0).to(main_device, dtype=final_dtype)

def obtener_embeddings_batch(textos):
    return safe_encode(textos, model=embedding_model)

def obtener_embedding(texto):
    return get_embedding_cached(texto)

def calcular_similitud_gpu(emb1, emb2):
    emb1_norm = torch.nn.functional.normalize(emb1, dim=0)
    emb2_norm = torch.nn.functional.normalize(emb2, dim=0)
    return torch.dot(emb1_norm, emb2_norm)

def similitud_textos(t1, t2):
    emb1 = obtener_embedding(t1)
    emb2 = obtener_embedding(t2)
    emb1_norm = torch.nn.functional.normalize(emb1, dim=0)
    emb2_norm = torch.nn.functional.normalize(emb2, dim=0)
    return float(torch.dot(emb1_norm, emb2_norm))

def consultar_memoria_gpu(query_emb, memoria_tensor, current_index, escenas_validas):
    if len(escenas_validas) <= 1:
        return ""
    query_norm = torch.nn.functional.normalize(query_emb.unsqueeze(0), dim=1)
    memoria_norm = torch.nn.functional.normalize(memoria_tensor, dim=1)
    sims = torch.matmul(memoria_norm, query_norm.T).squeeze()
    if sims.numel() <= 1:
        logger.warning("El tensor de similitud es vacío o tiene solo un elemento, no se puede indexar.")
        return ""
    try:
        if current_index < sims.numel():
            sims[current_index] = -1
    except Exception as e:
        logger.error(f"Error al indexar sims: {e}")
        return ""
    mask = sims >= MIN_CONTEXT_SIM
    if not mask.any():
        return ""
    sims_filtrados = sims[mask]
    indices_filtrados = torch.nonzero(mask, as_tuple=False).squeeze()
    if indices_filtrados.dim() == 0:
        indices_filtrados = indices_filtrados.unsqueeze(0)
    sorted_indices = indices_filtrados[torch.argsort(sims_filtrados, descending=True)]
    top_indices = sorted_indices[:TOP_K].tolist()
    context_fragments = [escenas_validas[i] for i in top_indices]
    return "\n".join(context_fragments)

# -----------------------------------------------------------
# CONFIGURACIÓN DE DISPOSITIVOS PARA EMBEDDINGS
# -----------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
embedding_model = None
embedding_model_0 = None
embedding_model_1 = None
multi_gpu = False
main_device = device
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
if num_gpus >= 2:
    logger.info(f"Se detectaron {num_gpus} GPUs; se usarán 2 para embeddings.")
    embedding_model_0 = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cuda:0")
    embedding_model_1 = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cuda:1")
    multi_gpu = True
    main_device = torch.device("cuda:0")
    embedding_model = embedding_model_0
elif num_gpus == 1:
    logger.info("Se detectó 1 GPU; se usará 'cuda:0' para embeddings.")
    multi_gpu = False
    main_device = torch.device("cuda:0")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cuda:0")
else:
    logger.info("No se detectaron GPUs; se usará CPU para embeddings.")
    multi_gpu = False
    main_device = torch.device("cpu")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")

def test_gpu():
    if torch.cuda.is_available():
        logger.info(f"Usando GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Usando CPU")

def set_embedding_device(device_str):
    global embedding_model, main_device
    device = torch.device(device_str)
    main_device = device
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    logger.info(f"Modelo de embeddings configurado en: {device_str}")

# -----------------------------------------------------------
# GENERACIÓN ITERATIVA DE TEXTO CON AJUSTE DINÁMICO Y FEEDBACK
# -----------------------------------------------------------
def remove_overlap(text_total, new_text, min_overlap=50):
    max_overlap = min(len(text_total), len(new_text))
    for i in range(max_overlap, min_overlap - 1, -1):
        if text_total.endswith(new_text[:i]):
            return new_text[i:]
    return new_text

def reformular_frases(frases):
    if not frases or len(frases) < 3:
        return " ".join(frases)
    frases_reformuladas = []
    for frase in frases:
        # Se busca simplificar la redacción y añadir un detalle adicional
        frases_reformuladas.append(frase.strip() + " (detalle adicional).")
    return " ".join(frases_reformuladas)

def summarize_text(text, max_length=SUMMARY_SIZE):
    if len(text) <= max_length:
        return text.strip()
    return text[-max_length:].strip()

def generar_texto_iterativo(prompt, modelo=None, max_tokens=8000, temperature=0.85, target_length=3100):
    texto_total = ""
    prompt_actual = prompt
    dynamic_max_tokens = max_tokens
    dynamic_temperature = temperature
    prev_segment = ""
    similarity_threshold = 0.9

    while len(texto_total) < target_length:
        respuesta = generar_texto(prompt_actual, max_tokens=dynamic_max_tokens, temperature=dynamic_temperature)
        if not respuesta or respuesta.strip() == "":
            logger.warning("No se obtuvo respuesta; terminando la generación iterativa.")
            break
        respuesta = respuesta.strip()
        respuesta = remove_overlap(texto_total, respuesta, min_overlap=50)
        # Dividir en frases y agrupar 17 frases para supervisión conjunta
        frases = re.split(r'(?<=[.!?])\s+', respuesta)
        if len(frases) >= PHRASE_GROUP_COUNT:
            bloque = " ".join(frases[:PHRASE_GROUP_COUNT])
            if not supervisar_frase(bloque, threshold=BLOCK_SUPERVISION_THRESHOLD):
                logger.warning("Bloque de 17 frases rechazado por el supervisor (perplejidad alta).")
                return texto_total.strip()
            respuesta = " ".join(frases[PHRASE_GROUP_COUNT:])
        if prev_segment:
            sim = similitud_textos(prev_segment, respuesta)
            logger.info(f"Similitud con el segmento anterior: {sim:.2f}")
            if sim > similarity_threshold:
                dynamic_temperature = min(dynamic_temperature + 0.05, 1.0)
                logger.info(f"Ajustando temperatura a: {dynamic_temperature}")
        prev_segment = respuesta
        texto_total += "\n" + respuesta
        update_rag_memory(respuesta)
        if len(respuesta) < 0.5 * dynamic_max_tokens:
            dynamic_max_tokens = min(dynamic_max_tokens + 500, 15000)
        else:
            dynamic_max_tokens = max(dynamic_max_tokens - 200, 2000)
        chapter_summary = summarize_text(texto_total, max_length=SUMMARY_SIZE)
        prompt_actual = ("Resumen del capítulo anterior:\n" + chapter_summary +
                         "\n\nContinúa la re-narración sin inventar nuevos episodios:\n" + texto_total[-500:])
        logger.info(f"Longitud acumulada: {len(texto_total)} caracteres")
    return texto_total.strip()

# -----------------------------------------------------------
# GENERACIÓN DE LA HISTORIA FINAL DIVIDIDA EN BLOQUES
# -----------------------------------------------------------
def dividir_texto_por_frases(texto, max_block_size=3000):
    oraciones = re.split(r'(?<=[.!?])\s+', texto)
    bloques = []
    bloque_actual = ""
    for oracion in oraciones:
        if len(bloque_actual) + len(oracion) + 1 <= max_block_size:
            bloque_actual += " " + oracion if bloque_actual else oracion
        else:
            # Si el bloque no termina en puntuación, mover la última oración al siguiente bloque
            if not re.search(r'[.!?]$', bloque_actual.strip()):
                partes = re.split(r'(?<=[.!?])\s+', bloque_actual.strip())
                if len(partes) > 1:
                    bloque_actual = " ".join(partes[:-1])
                    oracion = partes[-1] + " " + oracion
            bloques.append(bloque_actual.strip())
            bloque_actual = oracion
    if bloque_actual:
        bloques.append(bloque_actual.strip())
    return bloques

def extraer_resumen(bloque_anterior, max_chars=SUMMARY_SIZE):
    if not bloque_anterior:
        return ""
    if len(bloque_anterior) <= max_chars:
        return bloque_anterior.strip()
    else:
        return bloque_anterior[-max_chars:].strip()

def generar_re_narracion_22_partes(texto_original):
    texto_original = preprocess_spanish_text(texto_original)
    bloques = dividir_texto_por_frases(texto_original, max_block_size=3000)
    logger.info(f"Se dividió el texto en {len(bloques)} bloques.")

    total_partes = len(bloques)
    partes_generadas = []

    for idx, bloque in enumerate(bloques):
        parte_num = idx + 1
        if idx > 0:
            resumen = extraer_resumen(bloques[idx - 1], max_chars=SUMMARY_SIZE)
            header = generar_header_prompt(parte_num, total_partes, resumen, bloque)
            prompt = header
        else:
            prompt = f"Re-narra la Parte 1 de {total_partes} de la historia:\n\n{bloque}\n\nContinúa la re-narración sin inventar nuevos episodios:"
        logger.info(f"Generando la Parte {parte_num}...")
        resultado = generar_texto(prompt, max_tokens=1500, temperature=0.7)
        if not resultado:
            logger.warning(f"La Parte {parte_num} no se generó correctamente; se usará el bloque original.")
            resultado = bloque
        sim = similitud_textos(bloque, resultado)
        logger.info(f"[Parte {parte_num}] Similitud = {sim:.2f}")
        if sim < SIM_THRESHOLD:
            logger.warning(f"[Parte {parte_num}] Similitud < {SIM_THRESHOLD:.2f} => {sim:.2f}")
        parte_final = (
            f"Parte {parte_num} de {total_partes} – [Título descriptivo]\n\n"
            f"{resultado}\n\n"
            f"Fin de la Parte {parte_num}"
        )
        partes_generadas.append(parte_final)
    texto_final = "\n\n".join(partes_generadas)
    return texto_final

# -----------------------------------------------------------
# CREAR ARCHIVO TXT FINAL
# -----------------------------------------------------------
def crear_archivo_txt(texto_completo, archivo_salida):
    with open(archivo_salida, "w", encoding="utf-8") as f:
        f.write(texto_completo)
    logger.info(f"Archivo TXT guardado como: {archivo_salida}")

# -----------------------------------------------------------
# FUNCIÓN PRINCIPAL: GENERAR HISTORIA FINAL (TXT)
# -----------------------------------------------------------
def generar_historia_final(input_file, output_txt_path, modelo=LOCAL_MODEL_NAME):
    with open(input_file, "r", encoding="utf-8") as f:
        texto_original = f.read()
    texto_original = preprocess_spanish_text(texto_original)
    logger.info("Generando la re-narración en bloques...")
    historia_final = generar_re_narracion_22_partes(texto_original)
    logger.info("Creando archivo TXT final...")
    crear_archivo_txt(historia_final, output_txt_path)
    logger.info("Proceso de generación final completado.")
    return historia_final

# -----------------------------------------------------------
# FUNCIONES OPCIONALES PARA MULTI-MODELO Y EVALUACIÓN (NO OBLIGATORIAS)
# -----------------------------------------------------------
def guardar_resultados_csv(csv_filename, modelo, prompt_label, archivo_texto, similitud, tiempo, otros_datos=""):
    existe = os.path.exists(csv_filename)
    with open(csv_filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not existe:
            writer.writerow(["modelo", "prompt", "archivo", "similitud", "tiempo", "otros"])
        writer.writerow([modelo, prompt_label, archivo_texto, similitud, tiempo, otros_datos])

def main_for_scenario(prompt_label, prompt_base, model_label):
    start_time = time.time()
    if not os.path.exists(INPUT_FILE):
        logger.error(f"No se encontró {INPUT_FILE}")
        return
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        texto_original = f.read()
    if len(texto_original.strip()) < 100:
        logger.error("Texto demasiado corto.")
        return
    texto_preprocesado = preprocess_spanish_text(texto_original)
    safe_model = model_label.replace('/', '_').replace(':','_')
    output_prefix = f"historia_{safe_model}_{prompt_label}"
    texto_final_22 = generar_re_narracion_22_partes(texto_preprocesado)
    out_file = f"{output_prefix}_variada.txt"
    with open(out_file, "w", encoding="utf-8") as fw:
        fw.write(texto_final_22)
    elapsed_time = time.time() - start_time
    similitud_prom = 0.65  # Valor de ejemplo
    tiempo_inferencia = elapsed_time
    guardar_resultados_csv("resultados.csv", model_label, prompt_label, out_file, similitud_prom, tiempo_inferencia)
    logger.info(f"Flujo finalizado para {model_label} + {prompt_label}. Archivo guardado en: {out_file}")

def archivos_generados_existen(safe_model, prompt_label, min_size=8024):
    prefix = f"historia_{safe_model}_{prompt_label}"
    txt_file = f"{prefix}_variada.txt"
    emb_orig = f"{prefix}_original_embs.npy"
    emb_var = f"{prefix}_variada_embs.npy"
    files = [txt_file, emb_orig, emb_var]
    for fpath in files:
        if not os.path.exists(fpath):
            return False
        if os.path.getsize(fpath) < min_size:
            return False
    with open(txt_file, "r", encoding="utf-8") as f:
        content = f.read().strip()
    return len(content) >= 200

def run_all_models_in_order():
    global LOCAL_MODEL_NAME
    original_model = LOCAL_MODEL_NAME
    for modelo in LISTA_MODELOS:
        safe_model = modelo.replace('/', '_').replace(':','_')
        logger.info("\n" + "="*60)
        logger.info(f"=== PROCESANDO MODELO: {modelo} ===")
        logger.info("="*60 + "\n")
        LOCAL_MODEL_NAME = modelo
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            sample_text = f.read(500)
        selected_prompt = seleccionar_prompt(sample_text)
        prompt_label = "A_ES"  # Usamos el prompt en español
        if archivos_generados_existen(safe_model, prompt_label):
            logger.info(f"SKIP: El modelo {modelo} + prompt {prompt_label} ya tiene archivos válidos. Saltando.")
            continue
        main_for_scenario(prompt_label, selected_prompt, model_label=modelo)
    LOCAL_MODEL_NAME = original_model
    logger.info("\n*** TODOS LOS MODELOS Y PROMPTS FUERON PROCESADOS ***\n")

def evaluar_con_claude(csv_filename="resultados.csv"):
    if not os.path.exists(csv_filename):
        logger.error("No se encontró el archivo de resultados para evaluar con Claude.")
        return
    with open(csv_filename, "r", encoding="utf-8") as f:
        reporte = f.read()
    prompt_eval = f"""
Eres un crítico literario experto. A continuación se te presenta un reporte global con resultados de diversas variantes de re-narraciones,
generadas por distintos modelos utilizando un único prompt seleccionado según el idioma.
Analiza los siguientes datos y determina cuál variante tiene la mejor calidad narrativa y lógica.
Debes evaluar la coherencia, creatividad, fidelidad al significado original y calidad narrativa.

Reporte:
{reporte}

Proporciona un ranking final ordenado (del mejor al peor) indicando:
- Modelo
- Observaciones generales
"""
    global LOCAL_MODEL_NAME
    original_local_model = LOCAL_MODEL_NAME
    respuestas = []
    for evaluator in EVALUATOR_MODELOS:
        LOCAL_MODEL_NAME = evaluator
        logger.info(f"Evaluando con modelo evaluador Claude: {evaluator}")
        respuesta = generar_texto(prompt_eval, max_tokens=1024, temperature=0.5)
        respuestas.append((evaluator, respuesta))
    LOCAL_MODEL_NAME = original_local_model
    for eval_model, resp in respuestas:
        logger.info(f"Respuesta de {eval_model}:\n{resp}\n")

# -----------------------------------------------------------
# FUNCIÓN PRINCIPAL PARA GENERAR LA HISTORIA FINAL (TXT)
# -----------------------------------------------------------
def generar_historia_final(input_file, output_txt_path, modelo=LOCAL_MODEL_NAME):
    with open(input_file, "r", encoding="utf-8") as f:
        texto_original = f.read()
    texto_original = preprocess_spanish_text(texto_original)
    logger.info("Generando la re-narración en bloques...")
    historia_final = generar_re_narracion_22_partes(texto_original)
    logger.info("Creando archivo TXT final...")
    crear_archivo_txt(historia_final, output_txt_path)
    logger.info("Proceso de generación final completado.")
    return historia_final

# -----------------------------------------------------------
# EJECUCIÓN FINAL
# -----------------------------------------------------------
if __name__ == "__main__":
    test_gpu()
    INPUT_FILE = "historia.txt"
    OUTPUT_FILE = "historia_reescrita_final.txt"
    if os.path.exists(INPUT_FILE):
        generar_historia_final(INPUT_FILE, OUTPUT_FILE)
    else:
        logger.error(f"El archivo de entrada {INPUT_FILE} no existe.")
