from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sqlalchemy import create_engine, text
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="BIODIVERSITY-GUARD-Predict API",
    description="API REST para conectar la Base de Datos PostgreSQL (Supabase) con los Modelos de IA y el Dashboard de Monitoreo.",
    version="1.0.0"
)

# CORS: configurar los orígenes permitidos mediante CORS_ORIGINS.
# Por defecto se permite cualquier origen, sin envío de credenciales.
origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# La URL de conexión se configura en el entorno (Render o archivo .env local).
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "Falta configurar DATABASE_URL en las variables de entorno."
    )

engine = create_engine(DATABASE_URL, pool_pre_ping=True)


@app.get("/", tags=["Estado del Sistema"])
def verificar_estado():
    """Verifica la conexión activa entre la API y PostgreSQL en Supabase."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {
            "estado": "activo",
            "servicio": "BIODIVERSITY-GUARD-Predict API",
            "base_de_datos": "Conectada a Supabase (PostgreSQL)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error de conexión a BD: {str(e)}")


@app.get("/api/anp", tags=["Catálogo y Datos Históricos"])
def listar_anp():
    """Retorna las 31 Áreas Naturales Protegidas registradas."""
    df = pd.read_sql("SELECT * FROM anp ORDER BY anp_codi", engine)
    return df.to_dict(orient="records")


@app.get("/api/historico", tags=["Catálogo y Datos Históricos"])
def obtener_datos_historicos(
    anp_codi: Optional[str] = Query(None, description="Filtrar por código de ANP (ej. BP01)"),
    anio: Optional[int] = Query(None, description="Filtrar por año de reporte (2019-2023)")
):
    """Retorna el historial consolidado (SERNANP + SENAMHI) para gráficos de evolución en el Dashboard."""
    query = "SELECT * FROM datos_historicos WHERE 1=1"
    params = {}
    if anp_codi:
        query += " AND anp_codi = :anp_codi"
        params["anp_codi"] = anp_codi
    if anio:
        query += " AND anio_reporte = :anio"
        params["anio"] = anio
    query += " ORDER BY fecha DESC, anp_codi ASC"

    df = pd.read_sql(text(query), engine, params=params)
    return df.to_dict(orient="records")


@app.get("/api/features/frente1", tags=["Módulo de IA (Feature Store)"])
def obtener_features_frente1(anp_codi: Optional[str] = None):
    """Provee al Módulo de IA los datos preprocesados para el Frente 1 (Regresión)."""
    query = "SELECT * FROM datos_frente1"
    params = {}
    if anp_codi:
        query += " WHERE anp_codi = :anp_codi"
        params["anp_codi"] = anp_codi
    query += " ORDER BY fecha_objetivo_f1, anp_codi"

    df = pd.read_sql(text(query), engine, params=params)
    return df.to_dict(orient="records")


@app.get("/api/features/frente2", tags=["Módulo de IA (Feature Store)"])
def obtener_features_frente2(anp_codi: Optional[str] = None):
    """Provee al Módulo de IA los datos preprocesados para el Frente 2 (Clasificación)."""
    query = "SELECT * FROM datos_frente2"
    params = {}
    if anp_codi:
        query += " WHERE anp_codi = :anp_codi"
        params["anp_codi"] = anp_codi
    query += " ORDER BY fecha, anp_codi"

    df = pd.read_sql(text(query), engine, params=params)
    return df.to_dict(orient="records")


@app.get("/api/predicciones/frente1", tags=["Predicciones para Dashboard"])
def obtener_predicciones_frente1(
    anp_codi: Optional[str] = Query(None, description="Código del ANP (ej. BP01)"),
    mes: Optional[str] = Query(None, description="Fecha objetivo YYYY-MM-DD (ej. 2023-08-01)")
):
    """Entrega las hectáreas de deforestación estimadas por el modelo Random Forest ajustado."""
    query = "SELECT * FROM predicciones_frente1 WHERE 1=1"
    params = {}
    if anp_codi:
        query += " AND anp_codi = :anp_codi"
        params["anp_codi"] = anp_codi
    if mes:
        query += " AND fecha_objetivo = :mes"
        params["mes"] = mes
    query += " ORDER BY fecha_objetivo DESC, hectareas_estimadas DESC"

    df = pd.read_sql(text(query), engine, params=params)
    return df.to_dict(orient="records")


@app.get("/api/predicciones/frente2", tags=["Predicciones para Dashboard"])
def obtener_predicciones_frente2(
    anp_codi: Optional[str] = Query(None, description="Código del ANP (ej. BP01)"),
    nivel_riesgo: Optional[str] = Query(None, description="Filtrar por nivel (BAJO, MODERADO, ALTO, CRÍTICO)")
):
    """Entrega la clasificación de nivel de riesgo de incursión ilegal estimada por Random Forest."""
    query = "SELECT * FROM predicciones_frente2 WHERE 1=1"
    params = {}
    if anp_codi:
        query += " AND anp_codi = :anp_codi"
        params["anp_codi"] = anp_codi
    if nivel_riesgo:
        query += " AND nivel_riesgo_estimado = :nivel_riesgo"
        params["nivel_riesgo"] = nivel_riesgo.upper()
    query += " ORDER BY fecha_observacion DESC, anp_codi ASC"

    df = pd.read_sql(text(query), engine, params=params)
    return df.to_dict(orient="records")


@app.get("/api/dashboard/alertas", tags=["Predicciones para Dashboard"])
def obtener_alertas_consolidadas(
    anp_codi: Optional[str] = Query(None, description="Filtrar por código de ANP"),
    nivel_riesgo: Optional[str] = Query(None, description="Filtrar por riesgo (ej. CRÍTICO)")
):
    """Endpoint principal del Dashboard: une predicciones de Frente 1 y Frente 2 por ANP y mes."""
    query = "SELECT * FROM vista_alertas_dashboard WHERE 1=1"
    params = {}
    if anp_codi:
        query += " AND anp_codi = :anp_codi"
        params["anp_codi"] = anp_codi
    if nivel_riesgo:
        query += " AND nivel_riesgo_estimado = :nivel_riesgo"
        params["nivel_riesgo"] = nivel_riesgo.upper()

    df = pd.read_sql(text(query), engine, params=params)
    return df.to_dict(orient="records")
