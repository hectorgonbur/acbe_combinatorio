import streamlit as st
import itertools
import pandas as pd
import json
from datetime import datetime

# --- CONFIGURACIÓN E INTERFAZ ---
st.set_page_config(page_title="Betsson Pro: Master Suite", layout="wide")
st.title("⚽ Betsson Pro: Sistema Profesional (2-10 Partidos)")

# --- INICIALIZACIÓN DE ESTADO (MEMORIA) ---
if "biblioteca" not in st.session_state:
    st.session_state["biblioteca"] = {} 

# --- FUNCIONES DE GESTIÓN DE DATOS ---
def cargar_datos_en_session_state(datos):
    st.session_state["num_p_slider"] = datos.get("num_p", 6)
    st.session_state["solo_ganador_check"] = datos.get("solo_ganador", False)
    st.session_state["err_max_slider"] = datos.get("err_max", 2)
    st.session_state["apuesta_input"] = datos.get("apuesta_col", 1.0)
    
    if "biblioteca" in datos:
        st.session_state["biblioteca"] = datos["biblioteca"]

    num = datos.get("num_p", 6)
    for i in range(num):
        try:
            st.session_state[f"c_{i}"] = datos["competiciones"][i]
            st.session_state[f"l_{i}"] = datos["local"][i]
            st.session_state[f"v_{i}"] = datos["visit"][i]
            st.session_state[f"b_{i}"] = datos["base"][i]
            
            opciones_carga = ["1", "2"] if datos.get("solo_ganador") else ["1", "X", "2"]
            for op in opciones_carga:
                st.session_state[f"q_{op}_{i}"] = datos["cuotas"][i].get(op, 1.0)
        except IndexError:
            pass

def genera_sistema(base_user, err_maxima, quotes, costo, ops):
    combinazioni = itertools.product(ops, repeat=len(base_user))
    sistema = []
    
    for c in combinazioni:
        diff = sum(1 for i in range(len(base_user)) if c[i] != base_user[i])
        
        # Filtro de errores
        if diff <= err_maxima:
            quota_tot = 1.0
            for idx, s in enumerate(c):
                quota_tot *= quotes[idx][s]
            
            # --- CAMBIO: ELIMINADA LA COLUMNA "QUOTA TOTAL" ---
            # Nos quedamos solo con el dinero, que es lo que importa.
            sistema.append({
                "Columna": "-".join(c),
                "Fallos": diff,
                "Ganancia Bruta (€)": round(quota_tot * costo, 2)
            })
    return pd.DataFrame(sistema)

# --- BARRA LATERAL: ARCHIVOS ---
st.sidebar.header("📂 Gestión de Archivos")
archivo_subido = st.sidebar.file_uploader("Cargar Jugada/Historial", type=["json"])

if archivo_subido is not None:
    try:
        datos_cargados = json.load(archivo_subido)
        if st.sidebar.button("🔄 Restaurar Todo"):
            cargar_datos_en_session_state(datos_cargados)
            st.toast("Datos restaurados.", icon="✅")
            st.rerun()
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

st.sidebar.divider()

# --- CONFIGURACIÓN PARAMETROS ---
st.sidebar.header("⚙️ Configuración")
num_p = st.sidebar.slider("Partidos", 2, 10, key="num_p_slider", value=6)
solo_ganador = st.sidebar.checkbox("Modo 2 Resultados", key="solo_ganador_check", value=False)
err_max = st.sidebar.select_slider("Errores permitidos", options=[0, 1, 2, 3, 4, 5], key="err_max_slider", value=2)
apuesta_col = st.sidebar.number_input("Inversión por columna (€)", min_value=0.1, key="apuesta_input", value=1.0)

opciones = ["1", "2"] if solo_ganador else ["1", "X", "2"]

# --- GRID DE ENTRADA ---
st.subheader("1. Definir Partidos y Cuotas")
matriz_cuotas, col_base, equipos_local, equipos_visit, competiciones = [], [], [], [], []

grid = st.columns(2)
for i in range(num_p):
    with grid[i % 2]:
        with st.expander(f"PARTIDO {i+1}", expanded=True):
            comp = st.text_input("Torneo", key=f"c_{i}", value="Liga")
            competiciones.append(comp)
            
            c_l, c_v = st.columns(2)
            loc = c_l.text_input("Local", key=f"l_{i}", value=f"Local {i+1}")
            vis = c_v.text_input("Visitante", key=f"v_{i}", value=f"Visitante {i+1}")
            equipos_local.append(loc); equipos_visit.append(vis)

            c_b, c_qs = st.columns([1, 3])
            b = c_b.selectbox("Base", opciones, key=f"b_{i}")
            col_base.append(b)
            
            q_cols = c_qs.columns(len(opciones))
            d_q = {}
            for j, op in enumerate(opciones):
                val_q = q_cols[j].number_input(f"Q{op}", min_value=1.01, key=f"q_{op}_{i}", value=2.0)
                d_q[op] = val_q
            matriz_cuotas.append(d_q)

# --- CÁLCULO Y VISUALIZACIÓN DEL SISTEMA ---
st.divider()
st.subheader("2. Tabla de Combinaciones y Ganancias")

if st.button("📊 Calcular Rentabilidad", type="primary"):
    with st.spinner('Procesando matemáticas...'):
        # 1. Generamos el sistema
        df_sistema = genera_sistema(col_base, err_max, matriz_cuotas, apuesta_col, opciones)
        
        # 2. Calculamos el Costo Total
        spesa_totale = len(df_sistema) * apuesta_col
        
        # 3. Calculamos Ganancia Neta
        if not df_sistema.empty:
            df_sistema["Ganancia Neta (€)"] = df_sistema["Ganancia Bruta (€)"] - spesa_totale
        
        # Métricas
        c1, c2, c3 = st.columns(3)
        c1.metric("Columnas", len(df_sistema))
        c2.metric("Costo del Sistema", f"{spesa_totale:.2f} €")
        
        if not df_sistema.empty:
            max_neto = df_sistema["Ganancia Neta (€)"].max()
            c3.metric("Mejor Ganancia Neta", f"{max_neto:.2f} €", delta_color="normal" if max_neto > 0 else "inverse")
        
        # TABLA (LIMPIA, SIN QUOTA TOTAL)
        if not df_sistema.empty:
            st.dataframe(
                df_sistema.style.format({
                    "Ganancia Bruta (€)": "{:.2f} €",
                    "Ganancia Neta (€)": "{:.2f} €"
                }),
                use_container_width=True,
                height=400,
                hide_index=True # Ocultamos el índice numérico para ganar más espacio
            )
        else:
            st.warning("No se generaron columnas con esa configuración.")
            
        st.session_state["ultimo_sistema"] = df_sistema
        st.session_state["spesa_totale"] = spesa_totale

elif "ultimo_sistema" in st.session_state:
    df = st.session_state["ultimo_sistema"]
    spesa = st.session_state.get("spesa_totale", 0)
    
    st.info(f"Mostrando último cálculo (Costo: {spesa:.2f} €)")
    st.dataframe(
        df.style.format({
            "Ganancia Bruta (€)": "{:.2f} €",
            "Ganancia Neta (€)": "{:.2f} €"
        }),
        use_container_width=True,
        hide_index=True
    )

# --- SIMULADOR DE RESULTADOS ---
st.divider()
st.subheader("3. Simulador de Resultados Reales")
st.info("Ingresa los resultados para ver si ganaste dinero real.")

res_sim, goles_l, goles_v = [], [], []
cols_sim = st.columns(num_p)

for i in range(num_p):
    with cols_sim[i]: 
        st.caption(f"{equipos_local[i]} vs {equipos_visit[i]}")
        c_gl, c_gv = st.columns(2)
        gl = c_gl.number_input("L", min_value=0, step=1, key=f"sim_gl_{i}")
        gv = c_gv.number_input("V", min_value=0, step=1, key=f"sim_gv_{i}")
        goles_l.append(gl); goles_v.append(gv)
        
        if solo_ganador: s_auto = "1" if gl > gv else "2"
        else: s_auto = "1" if gl > gv else ("2" if gv > gl else "X")
        res_sim.append(s_auto)
        
        color = "🟢" if s_auto == col_base[i] else "🔴"
        st.markdown(f"**{s_auto}** {color}")

# --- BIBLIOTECA DE ESTADÍSTICAS ---
st.divider()
st.header("📚 Biblioteca de Equipos")

if st.button("💾 Registrar en Historial"):
    for i in range(num_p):
        nom_l = equipos_local[i]
        if nom_l not in st.session_state["biblioteca"]:
            st.session_state["biblioteca"][nom_l] = {"pj": 0, "gf": 0, "gc": 0, "wins": 0}
        
        st.session_state["biblioteca"][nom_l]["pj"] += 1
        st.session_state["biblioteca"][nom_l]["gf"] += goles_l[i]
        st.session_state["biblioteca"][nom_l]["gc"] += goles_v[i]
        if res_sim[i] == "1": st.session_state["biblioteca"][nom_l]["wins"] += 1

        nom_v = equipos_visit[i]
        if nom_v not in st.session_state["biblioteca"]:
            st.session_state["biblioteca"][nom_v] = {"pj": 0, "gf": 0, "gc": 0, "wins": 0}
        
        st.session_state["biblioteca"][nom_v]["pj"] += 1
        st.session_state["biblioteca"][nom_v]["gf"] += goles_v[i]
        st.session_state["biblioteca"][nom_v]["gc"] += goles_l[i]
        if res_sim[i] == "2": st.session_state["biblioteca"][nom_v]["wins"] += 1
    
    st.success("✅ Historial actualizado.")

if st.session_state["biblioteca"]:
    lista_stats = []
    for equipo, stats in st.session_state["biblioteca"].items():
        prom_gf = stats["gf"] / stats["pj"] if stats["pj"] > 0 else 0
        lista_stats.append({
            "Equipo": equipo,
            "Jugados": stats["pj"],
            "Goles Favor": stats["gf"],
            "Goles Contra": stats["gc"],
            "Prom. Goles": round(prom_gf, 2),
            "Victorias": stats["wins"]
        })
    
    df_biblio = pd.DataFrame(lista_stats)
    st.dataframe(df_biblio, use_container_width=True, hide_index=True)

# --- DESCARGA ---
st.divider()
st.subheader("💾 Guardar Datos")

col_d1, col_d2 = st.columns([3, 1])
with col_d1:
    nombre_archivo = st.text_input("Nombre:", value=f"backup_{datetime.now().strftime('%Y%m%d')}")

datos_export = {
    "num_p": num_p,
    "solo_ganador": solo_ganador,
    "err_max": err_max,
    "apuesta_col": apuesta_col,
    "local": equipos_local,
    "visit": equipos_visit,
    "competiciones": competiciones,
    "base": col_base,
    "cuotas": matriz_cuotas,
    "biblioteca": st.session_state["biblioteca"], 
    "fecha_creacion": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

json_str = json.dumps(datos_export, indent=4)

with col_d2:
    st.write("##")
    st.download_button("⬇️ Descargar JSON", json_str, f"{nombre_archivo}.json", "application/json")
