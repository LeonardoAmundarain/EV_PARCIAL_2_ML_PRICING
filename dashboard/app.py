"""
dashboard/app.py - Dashboard interactivo LLM Price Performance
Consume la FastAPI y muestra visualizaciones Plotly interactivas
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="LLM Price Performance Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Estilos ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
    padding: 1rem; border-radius: 12px; color: white;
    text-align: center; margin: 0.3rem;
}
.metric-value { font-size: 2rem; font-weight: bold; }
.metric-label { font-size: 0.85rem; opacity: 0.85; }
h1 { color: #1e3a5f; }
</style>
""", unsafe_allow_html=True)

# ── Helpers API ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def get_stats():
    try:
        return requests.get(f"{API_URL}/stats", timeout=10).json()
    except:
        return {}

@st.cache_data(ttl=60)
def get_benchmarks():
    try:
        return requests.get(f"{API_URL}/stats/benchmarks", timeout=10).json()
    except:
        return []

@st.cache_data(ttl=60)
def get_providers():
    try:
        return requests.get(f"{API_URL}/providers", timeout=10).json()
    except:
        return []

@st.cache_data(ttl=60)
def get_modelos(provider=None, tier=None, open_source=None, limit=200):
    params = {"limit": limit}
    if provider: params["provider"] = provider
    if tier:     params["tier"]     = tier
    if open_source is not None: params["open_source"] = open_source
    try:
        r = requests.get(f"{API_URL}/modelos", params=params, timeout=30)
        return r.json().get("data", [])
    except:
        return []

@st.cache_data(ttl=60)
def get_top_valor(n=20, tier=None):
    params = {"n": n}
    if tier: params["tier"] = tier
    try:
        return requests.get(f"{API_URL}/top-valor", params=params, timeout=10).json()
    except:
        return []

def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.status_code == 200, r.json()
    except:
        return False, {}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.title("🤖 LLM Dashboard")
    st.markdown("---")

    api_ok, api_info = check_api()
    if api_ok:
        st.success(f" API conectada\n\n{api_info.get('modelos_en_db', 0):,} modelos en BD")
    else:
        st.error(" API no disponible")

    st.markdown("---")
    pagina = st.radio("📊 Sección", [
        "📈 Resumen General",
        "🏆 Ranking de Modelos",
        "💰 Análisis de Precios",
        "🧠 Benchmarks Técnicos",
        "🔍 Explorador de Modelos",
        "🤖 Predictor de Tier ML"
    ])

# ── Página 1: Resumen General ─────────────────────────────────────────────────
if pagina == "📈 Resumen General":
    st.title("📈 Resumen General del Dataset")

    stats = get_stats()
    if not stats:
        st.error("No se pudieron cargar las estadísticas. Verifica que la API esté corriendo.")
        st.stop()

    total = stats.get("total_modelos", 0)
    tiers = stats.get("por_tier", {})
    providers = stats.get("top_10_providers", {})
    open_src  = stats.get("open_source", {})

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🤖 Total Modelos", f"{total:,}")
    col2.metric("🏢 Providers", len(providers))
    col3.metric("🆓 Open Source", open_src.get("True", 0))
    col4.metric("💼 Propietarios", open_src.get("False", 0))

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Distribución por Pricing Tier")
        tier_df = pd.DataFrame(list(tiers.items()), columns=["Tier", "Cantidad"])
        colores = {"Free":"#2ecc71","Budget":"#3498db","Mid":"#f39c12",
                   "Premium":"#e74c3c","Ultra":"#9b59b6","Unknown":"#95a5a6"}
        fig = px.pie(tier_df, values="Cantidad", names="Tier",
                     color="Tier", color_discrete_map=colores,
                     hole=0.4, title="Modelos por Tier de Precio")
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Top 10 Providers por cantidad de modelos")
        prov_df = pd.DataFrame(list(providers.items()), columns=["Provider", "Modelos"])
        prov_df = prov_df.sort_values("Modelos", ascending=True)
        fig2 = px.bar(prov_df, x="Modelos", y="Provider", orientation="h",
                      color="Modelos", color_continuous_scale="blues",
                      title="Modelos por Provider")
        fig2.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Open Source vs Propietario
    st.subheader("Open Source vs Propietario")
    os_df = pd.DataFrame([
        {"Tipo": "Open Source", "Cantidad": open_src.get("True", 0)},
        {"Tipo": "Propietario", "Cantidad": open_src.get("False", 0)}
    ])
    fig3 = px.bar(os_df, x="Tipo", y="Cantidad", color="Tipo",
                  color_discrete_map={"Open Source":"#2ecc71","Propietario":"#e74c3c"},
                  title="Distribución Open Source vs Propietario", text="Cantidad")
    fig3.update_traces(textposition="outside")
    st.plotly_chart(fig3, use_container_width=True)

# ── Página 2: Ranking ─────────────────────────────────────────────────────────
elif pagina == "🏆 Ranking de Modelos":
    st.title("🏆 Ranking de Modelos")

    tab1, tab2 = st.tabs(["💡 Mejor Valor (Inteligencia/Dólar)", "🧠 Más Inteligentes"])

    with tab1:
        n = st.slider("Cantidad de modelos", 5, 50, 20)
        tier_f = st.selectbox("Filtrar por tier", ["Todos","Budget","Mid","Premium","Ultra","Free"])
        datos = get_top_valor(n, tier_f if tier_f != "Todos" else None)
        if datos:
            df = pd.DataFrame(datos)
            df = df[["model_name","provider","pricing_tier","aa_intelligence_index",
                     "blended_cost_usd_per_1m","intelligence_per_dollar"]].dropna()
            df.columns = ["Modelo","Provider","Tier","Inteligencia","Costo Blended","Intel/Dólar"]
            df = df.round(2)

            fig = px.bar(df.head(20), x="Intel/Dólar", y="Modelo", orientation="h",
                         color="Tier", title="Top modelos por Inteligencia/Dólar",
                         hover_data=["Provider","Inteligencia","Costo Blended"])
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df, use_container_width=True)

    with tab2:
        try:
            top_int = requests.get(f"{API_URL}/top-inteligencia?n=30", timeout=10).json()
            if top_int:
                df2 = pd.DataFrame(top_int)
                df2 = df2[["model_name","provider","pricing_tier","aa_intelligence_index",
                            "aa_coding_index","aa_math_index"]].dropna(subset=["aa_intelligence_index"])
                df2.columns = ["Modelo","Provider","Tier","Inteligencia","Coding","Math"]
                df2 = df2.round(2)

                fig = px.scatter(df2, x="Coding", y="Inteligencia", size="Inteligencia",
                                 color="Tier", hover_name="Modelo", hover_data=["Provider"],
                                 title="Inteligencia vs Coding Index", size_max=30)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df2, use_container_width=True)
        except:
            st.error("Error cargando datos")

# ── Página 3: Análisis de Precios ─────────────────────────────────────────────
elif pagina == "💰 Análisis de Precios":
    st.title("💰 Análisis de Precios")

    datos = get_modelos(limit=200)
    if not datos:
        st.error("Sin datos disponibles"); st.stop()

    df = pd.DataFrame(datos)
    df_precio = df[df["pricing_tier"] != "Unknown"].dropna(subset=["blended_cost_usd_per_1m"])

    c1, c2 = st.columns(2)

    with c1:
        fig = px.box(df_precio, x="pricing_tier", y="blended_cost_usd_per_1m",
                     color="pricing_tier", title="Distribución de Costos por Tier",
                     category_orders={"pricing_tier":["Free","Budget","Mid","Premium","Ultra"]},
                     log_y=True)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        df_scatter = df.dropna(subset=["aa_intelligence_index","blended_cost_usd_per_1m"])
        fig2 = px.scatter(df_scatter, x="blended_cost_usd_per_1m", y="aa_intelligence_index",
                          color="pricing_tier", hover_name="model_name",
                          hover_data=["provider"],
                          title="Inteligencia vs Costo (USD/1M tokens)",
                          log_x=True, opacity=0.7)
        st.plotly_chart(fig2, use_container_width=True)

    # Costo promedio por provider
    st.subheader("Costo promedio por Provider (Top 15)")
    prov_cost = df.dropna(subset=["blended_cost_usd_per_1m"])
    prov_cost = prov_cost.groupby("provider")["blended_cost_usd_per_1m"].mean().reset_index()
    prov_cost.columns = ["Provider","Costo Promedio"]
    prov_cost = prov_cost.sort_values("Costo Promedio", ascending=False).head(15)
    fig3 = px.bar(prov_cost, x="Provider", y="Costo Promedio",
                  color="Costo Promedio", color_continuous_scale="reds",
                  title="Costo Blended Promedio por Provider (USD/1M tokens)")
    fig3.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig3, use_container_width=True)

# ── Página 4: Benchmarks Técnicos ────────────────────────────────────────────
elif pagina == "🧠 Benchmarks Técnicos":
    st.title("🧠 Benchmarks Técnicos por Tier")

    bench = get_benchmarks()
    if not bench:
        st.error("Sin datos"); st.stop()

    df_b = pd.DataFrame(bench).dropna(subset=["avg_intelligence"])
    df_b = df_b[df_b["tier"] != "Unknown"]
    orden = ["Free","Budget","Mid","Premium","Ultra"]
    df_b["tier"] = pd.Categorical(df_b["tier"], categories=orden, ordered=True)
    df_b = df_b.sort_values("tier")

    
    # Heatmap de benchmarks por tier
    st.subheader("Heatmap: Benchmarks promedio por Tier")

    heat_data = df_b[["tier","avg_intelligence","avg_coding","avg_math","avg_speed"]].copy()
    heat_data.columns = ["Tier","Inteligencia","Coding","Math","Velocidad (tok/s)"]
    heat_data = heat_data.set_index("Tier")
    heat_data = heat_data.round(2)

    fig = px.imshow(
    heat_data,
    text_auto=True,
    color_continuous_scale="Blues",
    title="Capacidades técnicas promedio por Pricing Tier",
    aspect="auto",
    labels=dict(color="Score")
    )
    fig.update_layout(
    xaxis_title="Benchmark",
    yaxis_title="Pricing Tier",
    height=350
    )
    st.plotly_chart(fig, use_container_width=True)

    # Barras agrupadas
    df_melt = df_b.melt(id_vars=["tier"], value_vars=["avg_intelligence","avg_coding","avg_math"],
                         var_name="Benchmark", value_name="Score")
    df_melt["Benchmark"] = df_melt["Benchmark"].map({
        "avg_intelligence":"Inteligencia","avg_coding":"Coding","avg_math":"Math"
    })
    fig2 = px.bar(df_melt, x="tier", y="Score", color="Benchmark", barmode="group",
                  title="Scores promedio de benchmarks por Tier",
                  category_orders={"tier": orden})
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Tabla resumen")
    df_b_display = df_b.copy()
    df_b_display.columns = ["Tier","Avg Inteligencia","Avg Coding","Avg Math","Avg Velocidad","N Modelos"]
    df_b_display = df_b_display.round(2)
    st.dataframe(df_b_display, use_container_width=True)

# ── Página 5: Explorador ──────────────────────────────────────────────────────
elif pagina == "🔍 Explorador de Modelos":
    st.title("🔍 Explorador de Modelos")

    providers_data = get_providers()
    provider_list  = ["Todos"] + sorted([p["provider"] for p in providers_data if p.get("provider")])
    tiers_list     = ["Todos","Free","Budget","Mid","Premium","Ultra","Unknown"]

    c1, c2, c3 = st.columns(3)
    sel_prov  = c1.selectbox("Provider", provider_list)
    sel_tier  = c2.selectbox("Pricing Tier", tiers_list)
    sel_os    = c3.selectbox("Open Source", ["Todos","Sí","No"])

    os_map = {"Todos": None, "Sí": True, "No": False}
    datos = get_modelos(
        provider=None if sel_prov=="Todos" else sel_prov,
        tier=None if sel_tier=="Todos" else sel_tier,
        open_source=os_map[sel_os],
        limit=200
    )

    if not datos:
        st.warning("No se encontraron modelos con esos filtros"); st.stop()

    df = pd.DataFrame(datos)
    st.markdown(f"**{len(df)} modelos encontrados**")

    # Scatter interactivo
    cols_num = ["aa_intelligence_index","aa_coding_index","aa_math_index",
                "blended_cost_usd_per_1m","output_tokens_per_second","intelligence_per_dollar"]
    cols_disp = {c: c for c in cols_num if c in df.columns}

    c1, c2 = st.columns(2)
    eje_x = c1.selectbox("Eje X", list(cols_disp.keys()), index=3)
    eje_y = c2.selectbox("Eje Y", list(cols_disp.keys()), index=0)

    df_plot = df.dropna(subset=[eje_x, eje_y])
    fig = px.scatter(df_plot, x=eje_x, y=eje_y,
                     color="pricing_tier", hover_name="model_name",
                     hover_data=["provider","is_open_source"],
                     title=f"{eje_y} vs {eje_x}", opacity=0.75,
                     size_max=15)
    st.plotly_chart(fig, use_container_width=True)

    # Tabla
    cols_tabla = ["model_name","provider","pricing_tier","aa_intelligence_index",
                  "blended_cost_usd_per_1m","output_tokens_per_second","is_open_source"]
    cols_tabla = [c for c in cols_tabla if c in df.columns]
    st.dataframe(df[cols_tabla].round(2), use_container_width=True)

# ── Página 6: Predictor ML ────────────────────────────────────────────────────
elif pagina == "🤖 Predictor de Tier ML":
    st.title("🤖 Predictor de Pricing Tier con ML")
    st.markdown("Ingresa las características de un modelo LLM para predecir su tier de precio usando los modelos Random Forest y Gradient Boosting entrenados.")

    with st.form("prediccion_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("Benchmarks")
            intel  = st.slider("Inteligencia (aa_intelligence_index)", 0.0, 60.0, 25.0, 0.5)
            coding = st.slider("Coding Index", 0.0, 60.0, 22.0, 0.5)
            math   = st.slider("Math Index",   0.0, 65.0, 20.0, 0.5)

        with c2:
            st.subheader("Precios (USD/1M tokens)")
            inp_cost = st.number_input("Costo Input",  min_value=0.0, max_value=100.0, value=2.0, step=0.5)
            out_cost = st.number_input("Costo Output", min_value=0.0, max_value=200.0, value=8.0, step=0.5)

        with c3:
            st.subheader("Rendimiento")
            tps     = st.number_input("Tokens/segundo", min_value=0.0, max_value=600.0, value=80.0, step=5.0)
            ttft    = st.number_input("Tiempo 1er token (s)", min_value=0.0, max_value=300.0, value=1.5, step=0.5)
            elo     = st.number_input("Arena ELO", min_value=0.0, max_value=1500.0, value=1150.0, step=10.0)
            year    = st.selectbox("Año de lanzamiento", [2023,2024,2025,2026], index=2)

        submitted = st.form_submit_button("🔮 Predecir Tier", use_container_width=True)

    if submitted:
        payload = {
            "aa_intelligence_index": intel,
            "aa_coding_index":       coding,
            "aa_math_index":         math,
            "input_cost_usd_per_1m": inp_cost,
            "output_cost_usd_per_1m":out_cost,
            "output_tokens_per_second": tps,
            "time_to_first_token_s": ttft,
            "chatbot_arena_elo":     elo,
            "release_year":          float(year),
            "cost_avg":              (inp_cost + out_cost) / 2
        }
        try:
            r   = requests.post(f"{API_URL}/prediccion", json=payload, timeout=10)
            res = r.json()

            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("🌲 Random Forest", res.get("prediccion_rf","—"))
            c2.metric("🚀 Gradient Boosting", res.get("prediccion_gb","—"))
            c3.metric("🤝 Consenso", res.get("consenso","—"))

            # Gráfico de confianza
            confianza = res.get("confianza_rf", {})
            if confianza:
                df_conf = pd.DataFrame(list(confianza.items()), columns=["Tier","Probabilidad"])
                df_conf = df_conf.sort_values("Probabilidad", ascending=True)
                fig = px.bar(df_conf, x="Probabilidad", y="Tier", orientation="h",
                             color="Probabilidad", color_continuous_scale="blues",
                             title="Probabilidad por Tier (Random Forest)",
                             text=df_conf["Probabilidad"].apply(lambda x: f"{x:.1%}"))
                fig.update_traces(textposition="outside")
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error al predecir: {e}")
