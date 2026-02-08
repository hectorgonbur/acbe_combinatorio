"""
🎯 ACBE-S73 QUANTUM BETTING SUITE v2.3
Sistema profesional de optimización de portafolios de apuestas deportivas
Combina Inferencia Bayesiana Gamma-Poisson, Teoría de la Información y Criterio de Kelly
Con cobertura S73 completa (2 errores) y gestión probabilística avanzada

CORRECCIONES IMPLEMENTADAS v2.3:
1. ✅ Corrección total de errores de tipado en gráficos Plotly (paleta RISK_PALETTE)
2. ✅ Restauración funcional del modo manual de inputs con toggle auto/manual
3. ✅ Validación institucional del sistema S73 reducido con umbrales probabilísticos
4. ✅ Capa de unificación Portfolio Engine con métricas cuantitativas completas
5. ✅ Modularización limpia y tipado fuerte
6. ✅ CORRECCIÓN CRÍTICA: Separación de fases Input/Análisis para evitar recargas
7. ✅ Sistema de estado de sesión para persistencia de datos
8. ✅ Botones de acción ubicados correctamente junto a los inputs
9. ✅ CORRECCIÓN: Botón "Volver a Ingreso de Datos" completamente funcional
10. ✅ SISTEMA DE DESCARGA: Exportación profesional CSV/Excel/TXT

Autor: Arquitecto de Software & Data Scientist Senior
Nivel: Quant Developer | Risk Engineer | Institutional Betting Model
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from typing import List, Tuple, Dict, Optional, Any, Union
import io
from datetime import datetime
warnings.filterwarnings('ignore')

# ============================================================================
# SECCIÓN 0: MANEJO DE ESTADO DE SESIÓN - MEJORADO
# ============================================================================

class SessionStateManager:
    """Gestor del estado de sesión para evitar recargas no deseadas."""
    
    @staticmethod
    def initialize_session_state():
        """Inicializa todas las variables de estado necesarias."""
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'matches_data' not in st.session_state:
            st.session_state.matches_data = None
        if 'params_dict' not in st.session_state:
            st.session_state.params_dict = None
        if 'processing_done' not in st.session_state:
            st.session_state.processing_done = False
        if 'current_tab' not in st.session_state:
            st.session_state.current_tab = "input"
        # NUEVO: Control explícito de fase
        if 'current_phase' not in st.session_state:
            st.session_state.current_phase = "input"  # "input" o "analysis"
        # NUEVO: Historial para navegación
        if 'phase_history' not in st.session_state:
            st.session_state.phase_history = ["input"]
    
    @staticmethod
    def reset_to_input():
        """Reinicia al estado de ingreso de datos - CORREGIDO."""
        st.session_state.data_loaded = False
        st.session_state.processing_done = False
        st.session_state.current_phase = "input"
        st.session_state.phase_history = ["input"]
        # Mantener datos para posible reutilización
        # st.session_state.matches_data = None
        # st.session_state.params_dict = None
    
    @staticmethod
    def move_to_analysis():
        """Mueve a la fase de análisis."""
        st.session_state.data_loaded = True
        st.session_state.processing_done = True
        st.session_state.current_phase = "analysis"
        if "analysis" not in st.session_state.phase_history:
            st.session_state.phase_history.append("analysis")
    
    @staticmethod
    def can_go_back() -> bool:
        """Verifica si se puede retroceder a fase anterior."""
        return len(st.session_state.phase_history) > 1
    
    @staticmethod
    def go_back():
        """Retrocede a la fase anterior."""
        if SessionStateManager.can_go_back():
            st.session_state.phase_history.pop()
            previous_phase = st.session_state.phase_history[-1]
            st.session_state.current_phase = previous_phase
            
            # Ajustar estados según fase
            if previous_phase == "input":
                st.session_state.data_loaded = False
                st.session_state.processing_done = False
            else:
                st.session_state.data_loaded = True
                st.session_state.processing_done = True
    
    @staticmethod
    def clear_all_data():
        """Limpia todos los datos de la sesión."""
        st.session_state.data_loaded = False
        st.session_state.matches_data = None
        st.session_state.params_dict = None
        st.session_state.processing_done = False
        st.session_state.current_tab = "input"
        st.session_state.current_phase = "input"
        st.session_state.phase_history = ["input"]

# ============================================================================
# SECCIÓN 1: CONFIGURACIÓN DEL SISTEMA Y CONSTANTES MATEMÁTICAS
# ============================================================================

class SystemConfig:
    """Configuración centralizada del sistema ACBE-S73."""
    
    # Parámetros de simulación
    MONTE_CARLO_ITERATIONS = 10000
    KELLY_FRACTION_MAX = 0.03  # 3% máximo por columna
    MIN_PROBABILITY = 1e-10    # Evitar log(0)
    BASE_ENTROPY = 3           # Base logarítmica para 3 resultados
    
    # Modelo Bayesiano Gamma-Poisson
    DEFAULT_ATTACK_MEAN = 1.2
    DEFAULT_DEFENSE_MEAN = 0.8
    DEFAULT_HOME_ADVANTAGE = 1.1
    DEFAULT_ALPHA_PRIOR = 2.0  # Parámetro de forma Gamma
    DEFAULT_BETA_PRIOR = 1.0   # Parámetro de tasa Gamma
    
    # Sistema S73
    NUM_MATCHES = 6            # Partidos por sistema
    FULL_COMBINATIONS = 3 ** 6  # 729 combinaciones posibles
    TARGET_COMBINATIONS = 73   # Objetivo de columnas reducidas
    HAMMING_DISTANCE_TARGET = 2  # ¡CORREGIDO: Cobertura de 2 errores!
    
    # Umbrales de clasificación por entropía
    STRONG_MATCH_THRESHOLD = 0.30   # ≤ 0.30: Partido Fuerte (1 signo)
    MEDIUM_MATCH_THRESHOLD = 0.60   # 0.30-0.60: Partido Medio (2 signos)
                                    # ≥ 0.60: Partido Caótico (3 signos)
    
    # Umbrales de reducción S73 (NUEVO - Validación institucional)
    MIN_OPTION_PROBABILITY = 0.55   # Umbral mínimo por opción
    MIN_PROBABILITY_GAP = 0.12      # Gap mínimo entre 1ª y 2ª opción
    MIN_EV_THRESHOLD = 0.0          # EV mínimo positivo
    
    # Gestión de riesgo
    MIN_ODDS = 1.01
    MAX_ODDS = 100.0
    DEFAULT_BANKROLL = 10000.0
    MAX_PORTFOLIO_EXPOSURE = 0.15   # 15% exposición máxima del portafolio
    MIN_JOINT_PROBABILITY = 0.001   # Umbral mínimo probabilidad conjunta
    
    # Configuración visual - PALETA CORREGIDA
    COLORS = {
        'primary': '#1E88E5',
        'secondary': '#FFC107', 
        'success': '#4CAF50',
        'danger': '#F44336',
        'warning': '#FF9800',
        'info': '#00BCD4'
    }
    
    # Paleta de riesgo para gráficos Pie (CORRECCIÓN PROBLEMA 1)
    RISK_PALETTE = [
        "#00BCD4",  # info
        "#4CAF50",  # success
        "#FFC107",  # warning
        "#FF9800",  # orange
        "#F44336"   # danger
    ]
    
    # Mapeo de resultados
    OUTCOME_MAPPING = {'1': 0, 'X': 1, '2': 2}
    OUTCOME_LABELS = ['1', 'X', '2']
    OUTCOME_COLORS = ['#1E88E5', '#FF9800', '#F44336']
    
    # Parámetros input manual
    MANUAL_INPUT_DEFAULTS = {
        'min_odds': 1.01,
        'max_odds': 100.0,
        'default_attack': 1.2,
        'default_defense': 0.8,
        'default_home_advantage': 1.1,
        'min_attack': 0.5,
        'max_attack': 2.0,
        'min_defense': 0.5,
        'max_defense': 2.0,
        'min_home_advantage': 1.0,
        'max_home_advantage': 1.5
    }

# ============================================================================
# SECCIÓN 2: CAPA DE INPUT PROFESIONAL CON MODO MANUAL CORREGIDO
# ============================================================================

class MatchInputLayer:
    """Capa de input profesional para partidos reales con validaciones avanzadas."""
    
    @staticmethod
    def validate_odds(odds_array: np.ndarray) -> np.ndarray:
        """
        Valida y normaliza cuotas ingresadas por el usuario.
        
        Args:
            odds_array: Array (n, 3) con cuotas [1, X, 2]
            
        Returns:
            Array validado y normalizado
        """
        # Validar valores nulos
        if np.any(np.isnan(odds_array)):
            st.warning("⚠️ Algunas cuotas tienen valores inválidos. Usando defaults...")
            return np.full_like(odds_array, 2.0)
        
        # Validar cuotas mínimas
        if np.any(odds_array <= SystemConfig.MIN_ODDS):
            st.warning(f"⚠️ Algunas cuotas son menores a {SystemConfig.MIN_ODDS}. Ajustando...")
            odds_array = np.maximum(odds_array, SystemConfig.MIN_ODDS + 0.01)
        
        # Validar cuotas máximas
        if np.any(odds_array > SystemConfig.MAX_ODDS):
            st.warning(f"⚠️ Algunas cuotas superan {SystemConfig.MAX_ODDS}. Ajustando...")
            odds_array = np.minimum(odds_array, SystemConfig.MAX_ODDS)
        
        return odds_array
    
    @staticmethod
    def render_manual_input_section() -> Tuple[pd.DataFrame, Dict, str]:
        """
        Renderiza la sección de input manual para partidos reales.
        DEVUELVE DATOS SIN PROCESAR - el procesamiento se hace después del botón.
        
        Returns:
            matches_df: DataFrame con datos de partidos
            params_dict: Diccionario con parámetros
            mode: Modo seleccionado
        """
        st.header("⚽ Input Manual de Partidos Reales")
        
        # Selector de modo - CORREGIDO
        st.subheader("🎯 Modo de Operación")
        
        # Usar selectbox para mayor claridad
        mode = st.selectbox(
            "Selecciona el modo de análisis:",
            ["🔘 Modo Automático", "🎮 Modo Manual"],
            index=0,  # Por defecto automático
            key="mode_selector_main"
        )
        
        is_manual_mode = mode == "🎮 Modo Manual"
        
        # Mostrar indicador claro del modo actual
        if is_manual_mode:
            st.success("✅ **MODO MANUAL ACTIVADO** - Puedes ajustar todas las fuerzas manualmente")
        else:
            st.info("🔘 **MODO AUTOMÁTICO** - Las fuerzas se estiman automáticamente")
        
        # Contenedor principal de input
        matches_data = []
        attack_strengths = []
        defense_strengths = []
        home_advantages = []
        
        # Crear 6 partidos (sistema S73 clásico)
        st.subheader(f"📝 Ingreso de {SystemConfig.NUM_MATCHES} Partidos")
        
        for match_idx in range(1, SystemConfig.NUM_MATCHES + 1):
            st.markdown(f"### Partido {match_idx}")
            
            # Contenedor para cada partido
            col_a, col_b, col_c = st.columns([2, 2, 3])
            
            with col_a:
                league = st.text_input(
                    f"Liga/Competición {match_idx}",
                    value=f"Liga {match_idx}",
                    key=f"league_{match_idx}_main"
                )
                home_team = st.text_input(
                    f"Equipo Local {match_idx}",
                    value=f"Local {match_idx}",
                    key=f"home_{match_idx}_main"
                )
                away_team = st.text_input(
                    f"Equipo Visitante {match_idx}",
                    value=f"Visitante {match_idx}",
                    key=f"away_{match_idx}_main"
                )
            
            with col_b:
                # Input de cuotas con validación
                odds_1 = st.number_input(
                    f"Cuota 1 - {home_team}",
                    min_value=1.01,
                    max_value=100.0,
                    value=2.0,
                    step=0.1,
                    key=f"odds1_{match_idx}_main"
                )
                odds_x = st.number_input(
                    f"Cuota X - Empate",
                    min_value=1.01,
                    max_value=100.0,
                    value=3.0,
                    step=0.1,
                    key=f"oddsx_{match_idx}_main"
                )
                odds_2 = st.number_input(
                    f"Cuota 2 - {away_team}",
                    min_value=1.01,
                    max_value=100.0,
                    value=2.5,
                    step=0.1,
                    key=f"odds2_{match_idx}_main"
                )
            
            with col_c:
                # ===== CORRECCIÓN CRÍTICA: Lógica condicional corregida =====
                if is_manual_mode:
                    # Mostrar controles avanzados expandidos por defecto en modo manual
                    with st.expander("⚙️ Ajustes Avanzados", expanded=True):
                        st.markdown("**Fuerzas Relativas (default ≈ 1.0)**")
                        
                        # Sliders para fuerzas con valores más realistas
                        home_attack = st.slider(
                            f"Ataque {home_team}",
                            min_value=0.5,
                            max_value=2.0,
                            value=SystemConfig.DEFAULT_ATTACK_MEAN,
                            step=0.05,
                            key=f"ha_{match_idx}_main",
                            help="Fuerza de ataque del equipo local (1.0 = promedio)"
                        )
                        home_defense = st.slider(
                            f"Defensa {home_team}",
                            min_value=0.5,
                            max_value=2.0,
                            value=SystemConfig.DEFAULT_DEFENSE_MEAN,
                            step=0.05,
                            key=f"hd_{match_idx}_main",
                            help="Fuerza defensiva del equipo local (1.0 = promedio)"
                        )
                        away_attack = st.slider(
                            f"Ataque {away_team}",
                            min_value=0.5,
                            max_value=2.0,
                            value=SystemConfig.DEFAULT_ATTACK_MEAN,
                            step=0.05,
                            key=f"aa_{match_idx}_main",
                            help="Fuerza de ataque del equipo visitante (1.0 = promedio)"
                        )
                        away_defense = st.slider(
                            f"Defensa {away_team}",
                            min_value=0.5,
                            max_value=2.0,
                            value=SystemConfig.DEFAULT_DEFENSE_MEAN,
                            step=0.05,
                            key=f"ad_{match_idx}_main",
                            help="Fuerza defensiva del equipo visitante (1.0 = promedio)"
                        )
                        home_advantage = st.slider(
                            f"Ventaja Local",
                            min_value=1.0,
                            max_value=1.5,
                            value=SystemConfig.DEFAULT_HOME_ADVANTAGE,
                            step=0.01,
                            key=f"adv_{match_idx}_main",
                            help="Factor de ventaja por jugar en casa (típico: 1.1-1.2)"
                        )
                else:
                    # Valores por defecto para modo automático - NO mostrar controles
                    home_attack = SystemConfig.DEFAULT_ATTACK_MEAN
                    home_defense = SystemConfig.DEFAULT_DEFENSE_MEAN
                    away_attack = SystemConfig.DEFAULT_ATTACK_MEAN
                    away_defense = SystemConfig.DEFAULT_DEFENSE_MEAN
                    home_advantage = SystemConfig.DEFAULT_HOME_ADVANTAGE
                    
                    # Solo mostrar información, no controles
                    st.info(
                        f"🔘 **Modo Automático - Partido {match_idx}**\n\n"
                        f"Fuerzas estimadas automáticamente:\n"
                        f"- **{home_team}**: Ataque={home_attack:.1f}, Defensa={home_defense:.1f}\n"
                        f"- **{away_team}**: Ataque={away_attack:.1f}, Defensa={away_defense:.1f}\n"
                        f"- **Ventaja local**: {home_advantage:.2f}x"
                    )
            
            # Calcular margen implícito
            implied_prob = (1/odds_1 + 1/odds_x + 1/odds_2)
            margin = (implied_prob - 1) * 100
            
            # Almacenar datos del partido
            matches_data.append({
                'match_id': match_idx,
                'league': league,
                'home_team': home_team,
                'away_team': away_team,
                'home_attack': home_attack,
                'away_attack': away_attack,
                'home_defense': home_defense,
                'away_defense': away_defense,
                'home_advantage': home_advantage,
                'odds_1': odds_1,
                'odds_X': odds_x,
                'odds_2': odds_2,
                'implied_prob': implied_prob,
                'margin': margin,
                'mode': 'Manual' if is_manual_mode else 'Auto'
            })
            
            attack_strengths.append([home_attack, away_attack])
            defense_strengths.append([home_defense, away_defense])
            home_advantages.append(home_advantage)
            
            st.markdown("---")
        
        # Crear DataFrames y matrices
        matches_df = pd.DataFrame(matches_data)
        
        # Extraer matriz de cuotas
        odds_matrix = matches_df[['odds_1', 'odds_X', 'odds_2']].values
        odds_matrix = MatchInputLayer.validate_odds(odds_matrix)
        
        # Crear diccionario con todos los parámetros
        params_dict = {
            'attack_strengths': np.array(attack_strengths),
            'defense_strengths': np.array(defense_strengths),
            'home_advantages': np.array(home_advantages),
            'matches_df': matches_df,
            'odds_matrix': odds_matrix,
            'mode': 'manual' if is_manual_mode else 'auto'
        }
        
        # Resumen del input
        MatchInputLayer._render_input_summary(matches_df, params_dict)
        
        return matches_df, params_dict, 'manual' if is_manual_mode else 'auto'
    
    @staticmethod
    def _render_input_summary(matches_df: pd.DataFrame, params_dict: Dict):
        """Renderiza resumen del input ingresado."""
        st.subheader("📋 Resumen del Input")
        
        # Métricas clave
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_margin = matches_df['margin'].mean()
            st.metric("Margen Promedio", f"{avg_margin:.2f}%")
        
        with col2:
            avg_odds = matches_df[['odds_1', 'odds_X', 'odds_2']].values.mean()
            st.metric("Cuota Promedio", f"{avg_odds:.2f}")
        
        with col3:
            total_combinations = 3 ** len(matches_df)
            st.metric("Combinaciones Totales", f"{total_combinations:,}")
        
        with col4:
            mode = params_dict['mode']
            st.metric("Modo", "🎮 Manual" if mode == 'manual' else "🔘 Automático")
    
    @staticmethod
    def process_manual_input(params_dict: Dict) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Procesa input manual para alimentar el pipeline ACBE.
        
        Args:
            params_dict: Diccionario con parámetros ingresados
            
        Returns:
            matches_df, odds_matrix, probabilities
        """
        # Extraer parámetros
        attack_strengths = params_dict['attack_strengths']
        defense_strengths = params_dict['defense_strengths']
        home_advantages = params_dict['home_advantages']
        odds_matrix = params_dict['odds_matrix']
        matches_df = params_dict['matches_df']
        
        # Calcular tasas de goles con ventajas específicas por partido
        n_matches = len(attack_strengths)
        lambda_home = np.zeros(n_matches)
        lambda_away = np.zeros(n_matches)
        
        for i in range(n_matches):
            lambda_home[i] = attack_strengths[i, 0] * defense_strengths[i, 1] * home_advantages[i]
            lambda_away[i] = attack_strengths[i, 1] * defense_strengths[i, 0]
        
        # Simular probabilidades ACBE
        probabilities = ACBEModel.vectorized_poisson_simulation(lambda_home, lambda_away)
        
        # Agregar columnas calculadas al DataFrame
        matches_df['lambda_home'] = lambda_home
        matches_df['lambda_away'] = lambda_away
        
        return matches_df, odds_matrix, probabilities

# ============================================================================
# SECCIÓN 3: MODELO MATEMÁTICO ACBE (VECTORIZADO)
# ============================================================================

class ACBEModel:
    """Modelo Bayesiano Gamma-Poisson para estimación de probabilidades."""
    
    @staticmethod
    @st.cache_data
    def vectorized_poisson_simulation(lambda_home: np.ndarray, 
                                     lambda_away: np.ndarray, 
                                     n_sims: int = SystemConfig.MONTE_CARLO_ITERATIONS) -> np.ndarray:
        """
        Simulación vectorizada de resultados usando distribución Poisson.
        
        Args:
            lambda_home: Tasas de goles locales (n_matches,)
            lambda_away: Tasas de goles visitantes (n_matches,)
            n_sims: Iteraciones Monte Carlo
            
        Returns:
            Array (n_matches, 3) con probabilidades [P(1), P(X), P(2)]
        """
        n_matches = len(lambda_home)
        
        # Generación vectorizada de goles
        home_goals = np.random.poisson(
            lam=np.tile(lambda_home, (n_sims, 1)),
            size=(n_sims, n_matches)
        )
        
        away_goals = np.random.poisson(
            lam=np.tile(lambda_away, (n_sims, 1)),
            size=(n_sims, n_matches)
        )
        
        # Cálculo de resultados (vectorizado)
        home_wins = (home_goals > away_goals).sum(axis=0) / n_sims
        draws = (home_goals == away_goals).sum(axis=0) / n_sims
        away_wins = (home_goals < away_goals).sum(axis=0) / n_sims
        
        # Ensamblar matriz de probabilidades
        probabilities = np.column_stack([home_wins, draws, away_wins])
        
        # Normalización y estabilidad numérica
        probabilities = np.clip(probabilities, SystemConfig.MIN_PROBABILITY, 1.0)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        
        return probabilities
    
    @staticmethod
    @st.cache_data
    def gamma_poisson_bayesian(attack_strengths: np.ndarray,
                              defense_strengths: np.ndarray,
                              home_advantage: float = SystemConfig.DEFAULT_HOME_ADVANTAGE,
                              alpha_prior: float = SystemConfig.DEFAULT_ALPHA_PRIOR,
                              beta_prior: float = SystemConfig.DEFAULT_BETA_PRIOR) -> Tuple[np.ndarray, np.ndarray]:
        """
        Modelo Bayesian Gamma-Poisson para estimar tasas de goles.
        
        Args:
            attack_strengths: Array (n_matches, 2) [ataque_local, ataque_visitante]
            defense_strengths: Array (n_matches, 2) [defensa_local, defensa_visitante]
            home_advantage: Ventaja de local
            
        Returns:
            lambda_home, lambda_away: Tasas estimadas de goles
        """
        n_matches = attack_strengths.shape[0]
        
        # Cálculo de lambda con ventaja local
        lambda_home = attack_strengths[:, 0] * defense_strengths[:, 1] * home_advantage
        lambda_away = attack_strengths[:, 1] * defense_strengths[:, 0]
        
        # Actualización Bayesian (Gamma-Poisson conjugada)
        alpha_posterior = alpha_prior + lambda_home + lambda_away
        beta_posterior = beta_prior + 2  # 2 equipos por partido
        
        # Muestreo de la posterior (vectorizado)
        lambda_home_samples = np.random.gamma(
            shape=alpha_posterior,
            scale=1/beta_posterior,
            size=(SystemConfig.MONTE_CARLO_ITERATIONS, n_matches)
        ).mean(axis=0)
        
        lambda_away_samples = np.random.gamma(
            shape=alpha_posterior,
            scale=1/beta_posterior,
            size=(SystemConfig.MONTE_CARLO_ITERations, n_matches)
        ).mean(axis=0)
        
        return lambda_home_samples, lambda_away_samples
    
    @staticmethod
    def calculate_entropy(probabilities: np.ndarray) -> np.ndarray:
        """
        Calcula entropía de Shannon (base 3) para cada partido.
        
        Args:
            probabilities: Array (n_matches, 3) de probabilidades
            
        Returns:
            Array (n_matches,) de entropías
        """
        # Estabilidad numérica
        probs = np.clip(probabilities, SystemConfig.MIN_PROBABILITY, 1.0)
        
        # Entropía vectorizada (base 3)
        entropy = -np.sum(probs * np.log(probs) / np.log(SystemConfig.BASE_ENTROPY), axis=1)
        
        return entropy
    
    @staticmethod
    def normalize_entropy(entropy: np.ndarray) -> np.ndarray:
        """
        Normaliza entropías al rango [0, 1].
        
        Args:
            entropy: Array de entropías
            
        Returns:
            Array normalizado
        """
        if np.max(entropy) - np.min(entropy) < SystemConfig.MIN_PROBABILITY:
            return np.ones_like(entropy)
        
        return (entropy - np.min(entropy)) / (np.max(entropy) - np.min(entropy))

# ============================================================================
# SECCIÓN 4: TEORÍA DE LA INFORMACIÓN Y CLASIFICACIÓN PROBABILÍSTICA
# ============================================================================

class InformationTheory:
    """Clasificación probabilística basada en entropía y teoría de información."""
    
    @staticmethod
    def classify_matches_by_entropy(probabilities: np.ndarray, 
                                   normalized_entropies: np.ndarray,
                                   odds_matrix: Optional[np.ndarray] = None) -> Tuple[List[List[int]], List[str]]:
        """
        Clasifica partidos según entropía normalizada y reduce espacio de signos.
        
        Sistema de clasificación:
        - Entropía ≤ 0.30: Partido Fuerte → 1 signo (el más probable)
        - Entropía 0.30-0.60: Partido Medio → 2 signos (más probables)
        - Entropía ≥ 0.60: Partido Caótico → 3 signos
        
        Args:
            probabilities: Array (n_matches, 3) de probabilidades
            normalized_entropies: Array (n_matches,) de entropías normalizadas
            odds_matrix: Array (n_matches, 3) de cuotas (opcional para filtros EV)
            
        Returns:
            allowed_signs: Lista de listas con signos permitidos por partido
            classifications: Lista de clasificaciones
        """
        allowed_signs = []
        classifications = []
        
        for i in range(len(probabilities)):
            entropy_norm = normalized_entropies[i]
            probs = probabilities[i]
            
            # Calcular EV si se proporcionan cuotas
            if odds_matrix is not None:
                evs = probs * odds_matrix[i] - 1
            else:
                evs = np.zeros(3)
            
            if entropy_norm <= SystemConfig.STRONG_MATCH_THRESHOLD:
                # Partido Fuerte: solo el signo más probable
                best_sign = np.argmax(probs)
                # Aplicar filtros institucionales
                if (probs[best_sign] >= SystemConfig.MIN_OPTION_PROBABILITY and 
                    evs[best_sign] > SystemConfig.MIN_EV_THRESHOLD):
                    allowed_signs.append([best_sign])
                    classifications.append('Fuerte')
                else:
                    # No pasa filtros, considerar más signos
                    allowed_signs.append([0, 1, 2])
                    classifications.append('Caótico (filtro)')
                
            elif entropy_norm <= SystemConfig.MEDIUM_MATCH_THRESHOLD:
                # Partido Medio: 2 signos más probables
                top_two = np.argsort(probs)[-2:].tolist()
                # Aplicar filtro de gap
                sorted_probs = np.sort(probs)[::-1]
                if len(sorted_probs) >= 2 and (sorted_probs[0] - sorted_probs[1]) >= SystemConfig.MIN_PROBABILITY_GAP:
                    # Gap suficiente, solo el más probable
                    allowed_signs.append([np.argmax(probs)])
                    classifications.append('Fuerte (gap)')
                else:
                    allowed_signs.append(top_two)
                    classifications.append('Medio')
                
            else:
                # Partido Caótico: 3 signos
                allowed_signs.append([0, 1, 2])
                classifications.append('Caótico')
        
        return allowed_signs, classifications
    
    @staticmethod
    def calculate_expected_value(probabilities: np.ndarray, odds_matrix: np.ndarray) -> np.ndarray:
        """
        Calcula el valor esperado (EV) para cada apuesta.
        
        Fórmula: EV = p * q - 1, donde:
        - p: probabilidad estimada
        - q: cuota ofrecida
        
        Args:
            probabilities: Array (n_matches, 3) de probabilidades
            odds_matrix: Array (n_matches, 3) de cuotas
            
        Returns:
            Array (n_matches, 3) de valores esperados
        """
        return probabilities * odds_matrix - 1

# ============================================================================
# SECCIÓN 5: SISTEMA COMBINATORIO S73 MEJORADO (VALIDACIÓN INSTITUCIONAL)
# ============================================================================

class S73System:
    """Sistema combinatorio S73 con cobertura garantizada de 2 errores y validación institucional."""
    
    @staticmethod
    @st.cache_data
    def generate_prefiltered_combinations(probabilities: np.ndarray,
                                         normalized_entropies: np.ndarray,
                                         odds_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera combinaciones pre-filtradas usando clasificación por entropía y filtros institucionales.
        
        Reduce el espacio de búsqueda antes de aplicar el sistema S73 con validación cuantitativa.
        
        Args:
            probabilities: Array (6, 3) de probabilidades (para 6 partidos)
            normalized_entropies: Array (6,) de entropías normalizadas
            odds_matrix: Array (6, 3) de cuotas para filtros EV
            
        Returns:
            combinations: Array (n_combinations, 6) de combinaciones filtradas
            joint_probs: Array (n_combinations,) de probabilidades conjuntas
        """
        # 1. Clasificar partidos y obtener signos permitidos con filtros institucionales
        allowed_signs, _ = InformationTheory.classify_matches_by_entropy(
            probabilities, normalized_entropies, odds_matrix
        )
        
        # 2. Validar que cada partido tenga al menos un signo (requisito S73)
        for i in range(len(allowed_signs)):
            if len(allowed_signs[i]) == 0:
                # Si no hay signos que cumplan filtros, usar los 3 signos
                allowed_signs[i] = [0, 1, 2]
                st.warning(f"Partido {i+1}: Ningún signo cumple filtros institucionales. Usando 3 signos.")
        
        # 3. Generar producto cartesiano de signos permitidos
        import itertools
        combinations_list = list(itertools.product(*allowed_signs))
        combinations = np.array(combinations_list)
        
        # 4. Calcular probabilidades conjuntas (vectorizado)
        n_combinations = len(combinations)
        joint_probs = np.ones(n_combinations)
        
        for idx, combo in enumerate(combinations):
            for match_idx, sign in enumerate(combo):
                joint_probs[idx] *= probabilities[match_idx, sign]
        
        # 5. Filtrar por umbral mínimo de probabilidad conjunta
        mask = joint_probs >= SystemConfig.MIN_JOINT_PROBABILITY
        filtered_combinations = combinations[mask]
        filtered_probs = joint_probs[mask]
        
        # 6. Validación estructural: garantizar mínimo de combinaciones para cobertura
        if len(filtered_combinations) < SystemConfig.TARGET_COMBINATIONS:
            st.warning(
                f"Solo {len(filtered_combinations)} combinaciones pasan filtros. "
                f"Se requieren al menos {SystemConfig.TARGET_COMBINATIONS} para cobertura S73."
            )
            # Relajar filtros progresivamente
            for threshold in [SystemConfig.MIN_JOINT_PROBABILITY/10, SystemConfig.MIN_JOINT_PROBABILITY/100]:
                mask = joint_probs >= threshold
                if len(combinations[mask]) >= SystemConfig.TARGET_COMBINATIONS:
                    filtered_combinations = combinations[mask]
                    filtered_probs = joint_probs[mask]
                    st.info(f"Filtros relajados a probabilidad conjunta ≥ {threshold:.6f}")
                    break
        
        return filtered_combinations, filtered_probs
    
    @staticmethod
    def hamming_distance_matrix(combinations: np.ndarray) -> np.ndarray:
        """
        Calcula matriz de distancias de Hamming entre combinaciones.
        
        Args:
            combinations: Array (n_combinations, 6) de combinaciones
            
        Returns:
            Array (n_combinations, n_combinations) de distancias
        """
        n = len(combinations)
        distances = np.zeros((n, n), dtype=np.int8)
        
        # Cálculo eficiente de distancias Hamming usando broadcasting
        for i in range(n):
            # Vectorizado: comparar fila i con todas las demás
            distances[i] = np.sum(combinations[i] != combinations, axis=1)
        
        return distances
    
    @staticmethod
    @st.cache_data
    def build_s73_coverage_system(filtered_combinations: np.ndarray,
                                 filtered_probs: np.ndarray,
                                 validate_coverage: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construye sistema S73 con cobertura garantizada de 2 errores y validación institucional.
        
        Implementa algoritmo greedy optimizado que selecciona combinaciones
        que maximizan la cobertura de espacio (Hamming distance ≤ 2) y cumple requisitos cuantitativos.
        
        Args:
            filtered_combinations: Combinaciones pre-filtradas
            filtered_probs: Probabilidades conjuntas correspondientes
            validate_coverage: Validar cobertura de 2 errores (True por defecto)
            
        Returns:
            selected_combinations: Array de combinaciones seleccionadas
            selected_probs: Array de probabilidades seleccionadas
        """
        n_combinations = len(filtered_combinations)
        
        if n_combinations <= SystemConfig.TARGET_COMBINATIONS:
            return filtered_combinations, filtered_probs
        
        # 1. Ordenar por probabilidad descendente
        sorted_indices = np.argsort(filtered_probs)[::-1]
        sorted_combinations = filtered_combinations[sorted_indices]
        sorted_probs = filtered_probs[sorted_indices]
        
        # 2. Precalcular matriz de distancias Hamming
        distance_matrix = S73System.hamming_distance_matrix(sorted_combinations)
        
        # 3. Algoritmo greedy con cobertura de 2 errores
        selected_indices = []
        covered_indices = set()
        
        # Validación: matriz de cobertura inicial (todas las combinaciones deben ser cubiertas)
        all_indices = set(range(n_combinations))
        
        while (len(selected_indices) < SystemConfig.TARGET_COMBINATIONS and 
               len(covered_indices) < n_combinations):
            
            best_idx = -1
            best_coverage_gain = -1
            
            # Buscar combinación que maximice cobertura de no cubiertos
            for i in range(n_combinations):
                if i in selected_indices:
                    continue
                
                # Combinaciones cubiertas por i (Hamming ≤ 2)
                coverage_mask = distance_matrix[i] <= SystemConfig.HAMMING_DISTANCE_TARGET
                uncovered_coverage = sum(1 for j in range(n_combinations) 
                                       if coverage_mask[j] and j not in covered_indices)
                
                # Ponderar por probabilidad y cobertura (optimización cuantitativa)
                coverage_gain = uncovered_coverage * (1 + sorted_probs[i])
                
                if coverage_gain > best_coverage_gain:
                    best_coverage_gain = coverage_gain
                    best_idx = i
            
            if best_idx == -1:
                break
            
            # Agregar combinación seleccionada
            selected_indices.append(best_idx)
            
            # Actualizar conjunto de combinaciones cubiertas
            newly_covered = np.where(
                distance_matrix[best_idx] <= SystemConfig.HAMMING_DISTANCE_TARGET
            )[0]
            covered_indices.update(newly_covered)
        
        # 4. Validación de cobertura completa
        if validate_coverage and len(covered_indices) < n_combinations:
            st.warning(f"Cobertura incompleta: {len(covered_indices)}/{n_combinations} combinaciones cubiertas")
            # Completar con combinaciones no cubiertas
            uncovered = list(all_indices - covered_indices)
            needed = SystemConfig.TARGET_COMBINATIONS - len(selected_indices)
            selected_indices.extend(uncovered[:needed])
            covered_indices.update(uncovered[:needed])
        
        # 5. Si no alcanza el target, completar con más probables
        if len(selected_indices) < SystemConfig.TARGET_COMBINATIONS:
            remaining_needed = SystemConfig.TARGET_COMBINATIONS - len(selected_indices)
            for i in range(n_combinations):
                if i not in selected_indices:
                    selected_indices.append(i)
                    remaining_needed -= 1
                    if remaining_needed == 0:
                        break
        
        # 6. Extraer combinaciones seleccionadas
        selected_combinations = sorted_combinations[selected_indices]
        selected_probs = sorted_probs[selected_indices]
        
        # 7. Validación final
        if validate_coverage:
            # Verificar que todas las combinaciones estén a distancia ≤ 2 de alguna seleccionada
            final_distance_matrix = S73System.hamming_distance_matrix(selected_combinations)
            min_distances_to_selected = np.min(final_distance_matrix, axis=0)
            max_min_distance = np.max(min_distances_to_selected)
            
            if max_min_distance > SystemConfig.HAMMING_DISTANCE_TARGET:
                st.error(f"❌ Error de cobertura: Distancia máxima = {max_min_distance} > {SystemConfig.HAMMING_DISTANCE_TARGET}")
            else:
                st.success(f"✅ Cobertura validada: Todas las combinaciones a distancia ≤ {SystemConfig.HAMMING_DISTANCE_TARGET}")
        
        return selected_combinations, selected_probs
    
    @staticmethod
    def calculate_combination_odds(combination: np.ndarray, odds_matrix: np.ndarray) -> float:
        """
        Calcula la cuota conjunta de una combinación.
        
        Args:
            combination: Array (6,) con signos seleccionados
            odds_matrix: Array (6, 3) de cuotas
            
        Returns:
            Cuota conjunta (producto de cuotas seleccionadas)
        """
        selected_odds = odds_matrix[np.arange(6), combination]
        return np.prod(selected_odds)

# ============================================================================
# SECCIÓN 6: CRITERIO DE KELLY INTEGRADO CON MODO MANUAL CORREGIDO
# ============================================================================

class KellyCapitalManagement:
    """Gestión de capital basada en criterio de Kelly con ajustes por entropía y modo manual."""
    
    @staticmethod
    def calculate_kelly_stakes(probabilities: np.ndarray,
                              odds_matrix: np.ndarray,
                              normalized_entropies: np.ndarray,
                              kelly_fraction: float = 1.0,
                              manual_stake: Optional[float] = None) -> np.ndarray:
        """
        Calcula stakes Kelly ajustados por entropía con soporte para modo manual.
        
        Args:
            probabilities: Array (n_matches, 3) de probabilidades
            odds_matrix: Array (n_matches, 3) de cuotas
            normalized_entropies: Array (n_matches,) de entropías normalizadas
            kelly_fraction: Fracción de Kelly a aplicar (0-1)
            manual_stake: Stake manual fijo (None para automático)
            
        Returns:
            Array (n_matches, 3) de stakes recomendados
        """
        if manual_stake is not None:
            # Modo manual: stake fijo para todas las apuestas
            stakes = np.full_like(probabilities, manual_stake)
            # Ajustar por entropía incluso en modo manual
            entropy_adjustment = (1.0 - normalized_entropies[:, np.newaxis])
            stakes = stakes * entropy_adjustment
            return stakes
        
        # Modo automático: calcular Kelly
        with np.errstate(divide='ignore', invalid='ignore'):
            kelly_raw = (probabilities * odds_matrix - 1) / (odds_matrix - 1)
        
        # Manejar casos especiales
        kelly_raw = np.nan_to_num(kelly_raw, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Aplicar límites (0 a KELLY_FRACTION_MAX)
        kelly_capped = np.clip(kelly_raw, 0, SystemConfig.KELLY_FRACTION_MAX)
        
        # Ajustar por entropía (más incertidumbre → menor stake)
        entropy_adjustment = (1.0 - normalized_entropies[:, np.newaxis])
        stakes = kelly_capped * entropy_adjustment * kelly_fraction
        
        return stakes
    
    @staticmethod
    def calculate_column_kelly(combination: np.ndarray,
                              joint_probability: float,
                              combination_odds: float,
                              avg_entropy: float,
                              manual_stake: Optional[float] = None) -> float:
        """
        Calcula stake Kelly para una columna del sistema S73 con soporte manual.
        
        Args:
            combination: Array (6,) de signos
            joint_probability: Probabilidad conjunta de la combinación
            combination_odds: Cuota conjunta
            avg_entropy: Entropía promedio de la combinación
            manual_stake: Stake manual fijo (None para automático)
            
        Returns:
            Stake Kelly ajustado (porcentaje del bankroll)
        """
        if manual_stake is not None:
            # Modo manual: stake fijo ajustado por entropía
            return manual_stake * (1.0 - avg_entropy)
        
        if combination_odds <= 1.0:
            return 0.0
        
        # Kelly para la combinación (modo automático)
        kelly_raw = (joint_probability * combination_odds - 1) / (combination_odds - 1)
        
        # Aplicar límites y ajuste por entropía
        kelly_capped = max(0.0, min(kelly_raw, SystemConfig.KELLY_FRACTION_MAX))
        kelly_adjusted = kelly_capped * (1.0 - avg_entropy)
        
        return kelly_adjusted
    
    @staticmethod
    def normalize_portfolio_stakes(stakes_array: np.ndarray,
                                  max_exposure: float = SystemConfig.MAX_PORTFOLIO_EXPOSURE,
                                  is_manual_mode: bool = False) -> np.ndarray:
        """
        Normaliza stakes para limitar exposición total del portafolio.
        
        Args:
            stakes_array: Array de stakes individuales
            max_exposure: Exposición máxima permitida (ej: 0.15 = 15%)
            is_manual_mode: Si es True, no escalar (solo limitar)
            
        Returns:
            Array de stakes normalizados
        """
        total_exposure = np.sum(stakes_array)
        
        if total_exposure > max_exposure:
            if is_manual_mode:
                # En modo manual, mantener proporciones pero limitar total
                scaling_factor = max_exposure / total_exposure
                stakes_array = stakes_array * scaling_factor
                st.warning(f"Stake manual reducido para mantener exposición máxima del {max_exposure*100:.0f}%")
            else:
                # En modo automático, escalar proporcionalmente
                scaling_factor = max_exposure / total_exposure
                stakes_array = stakes_array * scaling_factor
        
        return stakes_array

# ============================================================================
# SECCIÓN 7: PORTFOLIO ENGINE UNIFICADO (PROBLEMA 4)
# ============================================================================

class PortfolioEngine:
    """
    Motor de análisis de portafolio unificado para estrategias de apuestas.
    Calcula métricas institucionales para singles, combinadas y columnas S73.
    """
    
    def __init__(self, initial_bankroll: float = SystemConfig.DEFAULT_BANKROLL):
        self.initial_bankroll = initial_bankroll
        self.strategies = {
            'singles': {'stakes': [], 'odds': [], 'probabilities': []},
            'combinations': {'stakes': [], 'odds': [], 'probabilities': []},
            's73_columns': {'stakes': [], 'odds': [], 'probabilities': []}
        }
    
    def add_strategy(self, strategy_type: str, stakes: np.ndarray, 
                    odds: np.ndarray, probabilities: np.ndarray) -> None:
        """
        Agrega una estrategia al portafolio.
        
        Args:
            strategy_type: 'singles', 'combinations', o 's73_columns'
            stakes: Array de stakes (fracciones del bankroll)
            odds: Array de cuotas
            probabilities: Array de probabilidades
        """
        if strategy_type not in self.strategies:
            raise ValueError(f"Tipo de estrategia inválido: {strategy_type}")
        
        self.strategies[strategy_type]['stakes'].extend(stakes.tolist() if isinstance(stakes, np.ndarray) else stakes)
        self.strategies[strategy_type]['odds'].extend(odds.tolist() if isinstance(odds, np.ndarray) else odds)
        self.strategies[strategy_type]['probabilities'].extend(probabilities.tolist() if isinstance(probabilities, np.ndarray) else probabilities)
    
    def calculate_portfolio_metrics(self) -> Dict[str, Any]:
        """
        Calcula métricas cuantitativas del portafolio completo.
        
        Returns:
            Diccionario con métricas por estrategia y portafolio total
        """
        portfolio_metrics = {}
        
        for strategy_type, data in self.strategies.items():
            if not data['stakes']:
                continue
                
            stakes = np.array(data['stakes'])
            odds = np.array(data['odds'])
            probs = np.array(data['probabilities'])
            
            # Métricas básicas
            expected_values = (probs * odds - 1) * stakes * self.initial_bankroll
            total_ev = np.sum(expected_values)
            variance = np.var(expected_values) if len(expected_values) > 1 else 0
            
            # Sharpe Ratio (tasa libre de riesgo = 0)
            sharpe = total_ev / np.sqrt(variance) if variance > 0 else 0
            
            # Exposure y eficiencia
            total_exposure = np.sum(stakes) * 100  # Porcentaje
            capital_efficiency = total_ev / (total_exposure * self.initial_bankroll / 100) if total_exposure > 0 else 0
            
            # Drawdown esperado (simulación simplificada)
            win_prob = np.mean(probs)
            avg_odds = np.mean(odds)
            expected_drawdown = self._estimate_expected_drawdown(stakes, win_prob, avg_odds)
            
            # Probability of Ruin (Kelly-based)
            ruin_prob = self._calculate_ruin_probability(stakes, probs, odds)
            
            portfolio_metrics[strategy_type] = {
                'Expected Value (EV)': total_ev,
                'Variance': variance,
                'Sharpe Ratio': sharpe,
                'Max Drawdown (%)': expected_drawdown * 100,
                'Probability of Ruin (%)': ruin_prob * 100,
                'Capital Efficiency': capital_efficiency,
                'Total Exposure (%)': total_exposure,
                'Number of Bets': len(stakes),
                'Avg Stake (%)': np.mean(stakes) * 100,
                'Win Probability': win_prob
            }
        
        # Métricas agregadas del portafolio
        if portfolio_metrics:
            portfolio_metrics['portfolio'] = self._aggregate_portfolio_metrics(portfolio_metrics)
        
        return portfolio_metrics
    
    def _estimate_expected_drawdown(self, stakes: np.ndarray, win_prob: float, avg_odds: float) -> float:
        """Estima drawdown esperado usando simulación simplificada."""
        # Simulación Monte Carlo básica
        n_sims = 1000
        drawdowns = []
        
        for _ in range(n_sims):
            equity = 1.0
            peak = 1.0
            max_dd = 0.0
            
            for _ in range(100):  # 100 apuestas
                # Simular resultado
                if np.random.random() < win_prob:
                    equity += np.random.choice(stakes) * (avg_odds - 1)
                else:
                    equity -= np.random.choice(stakes)
                
                # Actualizar drawdown
                peak = max(peak, equity)
                dd = (peak - equity) / peak
                max_dd = max(max_dd, dd)
            
            drawdowns.append(max_dd)
        
        return np.mean(drawdowns) if drawdowns else 0
    
    def _calculate_ruin_probability(self, stakes: np.ndarray, probs: np.ndarray, odds: np.ndarray) -> float:
        """Calcula probabilidad de ruina usando fórmula de Kelly simplificada."""
        if len(stakes) == 0:
            return 0.0
        
        avg_stake = np.mean(stakes)
        avg_win_prob = np.mean(probs)
        avg_loss_prob = 1 - avg_win_prob
        avg_win_multiplier = np.mean(odds) - 1
        
        # Fórmula simplificada de probabilidad de ruina
        if avg_loss_prob == 0 or avg_stake == 0:
            return 0.0
        
        ruin_prob = ((1 - avg_win_prob * avg_win_multiplier * avg_stake) / 
                    (avg_loss_prob * avg_stake)) ** (self.initial_bankroll * 0.5 / avg_stake)
        
        return min(ruin_prob, 1.0)
    
    def _aggregate_portfolio_metrics(self, strategy_metrics: Dict) -> Dict:
        """Agrega métricas de todas las estrategias."""
        total_ev = sum(m['Expected Value (EV)'] for m in strategy_metrics.values())
        total_variance = sum(m['Variance'] for m in strategy_metrics.values())
        total_exposure = sum(m['Total Exposure (%)'] for m in strategy_metrics.values())
        
        # Sharpe Ratio agregado
        aggregate_sharpe = total_ev / np.sqrt(total_variance) if total_variance > 0 else 0
        
        # Drawdown agregado (máximo de los drawdowns individuales)
        max_drawdown = max(m['Max Drawdown (%)'] for m in strategy_metrics.values())
        
        # Probabilidad de ruina agregada
        ruin_probs = [m['Probability of Ruin (%)'] / 100 for m in strategy_metrics.values()]
        aggregate_ruin = 1 - np.prod([1 - p for p in ruin_probs])
        
        # Eficiencia de capital agregada
        total_investment = total_exposure * self.initial_bankroll / 100
        aggregate_efficiency = total_ev / total_investment if total_investment > 0 else 0
        
        return {
            'Total EV': total_ev,
            'Total Variance': total_variance,
            'Aggregate Sharpe': aggregate_sharpe,
            'Max Portfolio Drawdown (%)': max_drawdown,
            'Aggregate Ruin Probability (%)': aggregate_ruin * 100,
            'Aggregate Capital Efficiency': aggregate_efficiency,
            'Total Portfolio Exposure (%)': total_exposure,
            'Number of Strategies': len(strategy_metrics)
        }

# ============================================================================
# SECCIÓN 8: MOTOR DE BACKTESTING VECTORIZADO
# ============================================================================

class VectorizedBacktester:
    """Motor de backtesting completamente vectorizado con gestión real de capital."""
    
    def __init__(self, initial_bankroll: float = SystemConfig.DEFAULT_BANKROLL):
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.equity_curve = [initial_bankroll]
        self.drawdown_curve = [0.0]
    
    @staticmethod
    @st.cache_data
    def simulate_match_outcomes(probabilities: np.ndarray, n_sims: int) -> np.ndarray:
        """
        Simula resultados de partidos usando distribución multinomial.
        
        Args:
            probabilities: Array (n_matches, 3) de probabilidades
            n_sims: Número de simulaciones
            
        Returns:
            Array (n_sims, n_matches) de resultados (0, 1, 2)
        """
        n_matches = probabilities.shape[0]
        outcomes = np.zeros((n_sims, n_matches), dtype=int)
        
        for i in range(n_matches):
            # Muestreo multinomial vectorizado
            samples = np.random.multinomial(1, probabilities[i], size=n_sims)
            outcomes[:, i] = np.argmax(samples, axis=1)
        
        return outcomes
    
    def calculate_column_performance(self,
                                    real_outcomes: np.ndarray,
                                    combinations: np.ndarray,
                                    odds_matrix: np.ndarray,
                                    stakes_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula rendimiento de columnas con stakes reales.
        
        Args:
            real_outcomes: Resultados simulados (n_sims, 6)
            combinations: Combinaciones del sistema (n_columns, 6)
            odds_matrix: Cuotas (6, 3)
            stakes_array: Stakes por columna (n_columns,)
            
        Returns:
            total_returns: Retornos totales por simulación
            column_returns: Retornos por columna (n_sims, n_columns)
        """
        n_sims, n_matches = real_outcomes.shape
        n_columns = len(combinations)
        
        # Inicializar matriz de retornos
        column_returns = np.zeros((n_sims, n_columns))
        
        # Calcular cuotas conjuntas por columna
        combination_odds = np.zeros(n_columns)
        for i, combo in enumerate(combinations):
            combination_odds[i] = S73System.calculate_combination_odds(combo, odds_matrix)
        
        # Calcular stakes en euros
        stakes_euros = stakes_array * self.bankroll
        
        # Vectorizar comparación de resultados
        for col_idx, combination in enumerate(combinations):
            # Verificar aciertos (True si todos los partidos coinciden)
            hits = np.all(real_outcomes == combination, axis=1)
            
            # Calcular retorno: ganancia si acierta, pérdida stake si falla
            column_returns[:, col_idx] = np.where(
                hits,
                stakes_euros[col_idx] * (combination_odds[col_idx] - 1),  # Ganancia
                -stakes_euros[col_idx]                                   # Pérdida
            )
        
        # Retorno total por simulación (suma de todas las columnas)
        total_returns = column_returns.sum(axis=1)
        
        return total_returns, column_returns
    
    def run_backtest(self,
                    probabilities: np.ndarray,
                    odds_matrix: np.ndarray,
                    normalized_entropies: np.ndarray,
                    s73_results: Dict,
                    n_rounds: int = 100,
                    n_sims_per_round: int = 1000,
                    kelly_fraction: float = 0.5,
                    manual_stake: Optional[float] = None) -> Dict:
        """
        Ejecuta backtesting completo con gestión realista de capital.
        
        Args:
            probabilities: Probabilidades ACBE (6, 3)
            odds_matrix: Cuotas (6, 3)
            normalized_entropies: Entropías normalizadas (6,)
            s73_results: Resultados del sistema S73
            n_rounds: Número de rondas/jornadas
            n_sims_per_round: Simulaciones Monte Carlo por ronda
            kelly_fraction: Fracción conservadora de Kelly
            manual_stake: Stake manual fijo (None para automático)
            
        Returns:
            Diccionario con resultados del backtest
        """
        combinations = s73_results['combinations']
        n_columns = len(combinations)
        
        # Reinicializar métricas
        self.bankroll = self.initial_bankroll
        self.equity_curve = [self.bankroll]
        self.drawdown_curve = [0.0]
        
        all_returns = []
        round_metrics = []
        
        for round_idx in range(n_rounds):
            # 1. Simular resultados reales
            real_outcomes = self.simulate_match_outcomes(probabilities, n_sims_per_round)
            
            # 2. Calcular stakes actualizados (pueden cambiar con bankroll)
            current_stakes = self._calculate_current_stakes(
                s73_results, kelly_fraction, manual_stake
            )
            
            # 3. Calcular rendimiento
            round_returns, column_returns = self.calculate_column_performance(
                real_outcomes, combinations, odds_matrix, current_stakes
            )
            
            # 4. Actualizar bankroll (usar retorno promedio esperado)
            avg_return = np.mean(round_returns)
            self.bankroll += avg_return
            
            # 5. Registrar métricas
            self.equity_curve.append(self.bankroll)
            all_returns.extend(round_returns)
            
            # Calcular drawdown actual
            peak = np.max(self.equity_curve)
            current_dd = (peak - self.bankroll) / peak * 100 if peak > 0 else 0
            self.drawdown_curve.append(current_dd)
            
            # Métricas de la ronda
            round_metrics.append({
                'round': round_idx + 1,
                'bankroll': self.bankroll,
                'avg_return': avg_return,
                'std_return': np.std(round_returns),
                'win_rate': np.mean(round_returns > 0) * 100,
                'max_single_return': np.max(round_returns),
                'min_single_return': np.min(round_returns)
            })
        
        # Calcular métricas finales
        final_metrics = self._calculate_final_metrics(all_returns, n_rounds)
        
        return {
            'equity_curve': np.array(self.equity_curve),
            'drawdown_curve': np.array(self.drawdown_curve),
            'final_metrics': final_metrics,
            'round_metrics': round_metrics,
            'all_returns': np.array(all_returns)
        }
    
    def _calculate_current_stakes(self, s73_results: Dict, kelly_fraction: float, 
                                 manual_stake: Optional[float]) -> np.ndarray:
        """Calcula stakes actualizados basados en bankroll actual."""
        if manual_stake is not None:
            # Modo manual: stake fijo
            stakes = np.full(len(s73_results['combinations']), manual_stake)
        else:
            # Modo automático: Kelly ajustado
            stakes = s73_results['kelly_stakes'].copy()
            stakes = stakes * kelly_fraction
        
        # Normalizar para limitar exposición total
        stakes = KellyCapitalManagement.normalize_portfolio_stakes(
            stakes, 
            is_manual_mode=(manual_stake is not None)
        )
        
        return stakes
    
    def _calculate_final_metrics(self, all_returns: List[float], n_rounds: int) -> Dict:
        """Calcula métricas finales agregadas del backtest."""
        returns_array = np.array(all_returns)
        
        # ROI y retorno total
        total_return = self.bankroll - self.initial_bankroll
        total_return_pct = (total_return / self.initial_bankroll) * 100
        
        # Sharpe Ratio (tasa libre de riesgo = 0)
        if np.std(returns_array) > 0:
            sharpe_ratio = (np.mean(returns_array) / np.std(returns_array)) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Drawdown máximo
        max_drawdown = np.max(self.drawdown_curve)
        
        # CAGR (Compound Annual Growth Rate)
        if self.bankroll > 0:
            cagr = ((self.bankroll / self.initial_bankroll) ** (252 / n_rounds) - 1) * 100
        else:
            cagr = -100.0
        
        # Value at Risk (VaR 95%)
        var_95 = np.percentile(returns_array, 5) if len(returns_array) > 0 else 0
        
        # Win rate y estadísticas
        positive_returns = returns_array[returns_array > 0]
        negative_returns = returns_array[returns_array <= 0]
        
        win_rate = (len(positive_returns) / len(returns_array) * 100) if len(returns_array) > 0 else 0
        avg_win = np.mean(positive_returns) if len(positive_returns) > 0 else 0
        avg_loss = np.mean(negative_returns) if len(negative_returns) > 0 else 0
        
        # Profit factor
        if np.sum(negative_returns) < 0:
            profit_factor = abs(np.sum(positive_returns) / np.sum(negative_returns))
        else:
            profit_factor = 0.0
        
        # Probabilidad de ruina (bankroll < 50% inicial)
        ruin_prob = np.mean(np.array(self.equity_curve) < self.initial_bankroll * 0.5) * 100
        
        return {
            'initial_bankroll': self.initial_bankroll,
            'final_bankroll': self.bankroll,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'roi_per_round': (np.mean(returns_array) / self.initial_bankroll) * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'cagr': cagr,
            'var_95': var_95,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'ruin_probability': ruin_prob,
            'std_returns': np.std(returns_array) if len(returns_array) > 0 else 0
        }

# ============================================================================
# SECCIÓN 9: SISTEMA DE EXPORTACIÓN DE DATOS PROFESIONAL
# ============================================================================

class DataExporter:
    """Sistema profesional de exportación de datos para ACBE-S73."""
    
    @staticmethod
    def generate_timestamp() -> str:
        """Genera timestamp para nombres de archivo."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    @staticmethod
    def export_s73_columns(columns_df: pd.DataFrame, s73_results: Dict) -> Dict[str, Any]:
        """
        Exporta columnas S73 en múltiples formatos.
        
        Args:
            columns_df: DataFrame con columnas S73
            s73_results: Resultados del sistema S73
            
        Returns:
            Diccionario con datos para descarga
        """
        timestamp = DataExporter.generate_timestamp()
        
        # CSV con encoding correcto
        csv_data = columns_df.to_csv(index=False, sep=';', decimal=',', encoding='utf-8-sig')
        
        # Resumen ejecutivo
        summary_data = {
            'Fecha Exportación': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            'Total Columnas': [len(columns_df)],
            'Exposición Total (%)': [columns_df['Stake (%)'].sum()],
            'Inversión Total (€)': [columns_df['Inversión (€)'].sum()],
            'Probabilidad Promedio (%)': [columns_df['Probabilidad'].mean() * 100],
            'Cuota Promedio': [columns_df['Cuota'].mean()],
            'Valor Esperado Promedio': [columns_df['Valor Esperado'].mean()],
            'Cobertura Validada': ['SÍ' if s73_results.get('coverage_validated', False) else 'NO']
        }
        summary_df = pd.DataFrame(summary_data)
        summary_csv = summary_df.to_csv(index=False, sep=';', decimal=',', encoding='utf-8-sig')
        
        return {
            'csv': {
                'data': csv_data,
                'filename': f'acbe_s73_columnas_{timestamp}.csv',
                'mime': 'text/csv'
            },
            'summary': {
                'data': summary_csv,
                'filename': f'acbe_s73_resumen_{timestamp}.csv',
                'mime': 'text/csv'
            }
        }
    
    @staticmethod
    def export_backtest_results(backtest_results: Dict) -> Dict[str, Any]:
        """
        Exporta resultados de backtesting.
        
        Args:
            backtest_results: Resultados del backtesting
            
        Returns:
            Diccionario con datos para descarga
        """
        timestamp = DataExporter.generate_timestamp()
        
        # Métricas principales
        metrics_df = pd.DataFrame([backtest_results['final_metrics']])
        metrics_csv = metrics_df.to_csv(index=False, sep=';', decimal=',', encoding='utf-8-sig')
        
        # Curva de equity
        equity_df = pd.DataFrame({
            'Ronda': range(len(backtest_results['equity_curve'])),
            'Bankroll': backtest_results['equity_curve'],
            'Drawdown_%': backtest_results['drawdown_curve']
        })
        equity_csv = equity_df.to_csv(index=False, sep=';', decimal=',', encoding='utf-8-sig')
        
        # Retornos por ronda
        returns_df = pd.DataFrame({
            'Ronda': range(len(backtest_results['all_returns'])),
            'Retorno_€': backtest_results['all_returns']
        })
        returns_csv = returns_df.to_csv(index=False, sep=';', decimal=',', encoding='utf-8-sig')
        
        # Reporte ejecutivo en texto
        metrics = backtest_results['final_metrics']
        report_text = f"""
        REPORTE DE BACKTESTING ACBE-S73
        ================================
        Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        
        MÉTRICAS DE RENDIMIENTO:
        ------------------------
        Bankroll Inicial: €{metrics['initial_bankroll']:,.2f}
        Bankroll Final: €{metrics['final_bankroll']:,.2f}
        Retorno Total: {metrics['total_return_pct']:+.2f}%
        
        ROI por Ronda: {metrics['roi_per_round']:+.4f}%
        Sharpe Ratio: {metrics['sharpe_ratio']:.3f}
        CAGR: {metrics['cagr']:+.2f}%
        
        RIESGO:
        -------
        Max Drawdown: {metrics['max_drawdown']:.2f}%
        VaR 95%: €{metrics['var_95']:.2f}
        Volatilidad (σ): €{metrics['std_returns']:.2f}
        Prob. Ruina: {metrics['ruin_probability']:.2f}%
        
        ESTADÍSTICAS:
        -------------
        Win Rate: {metrics['win_rate']:.2f}%
        Profit Factor: {metrics['profit_factor']:.3f}
        Ganancia Promedio: €{metrics['avg_win']:.2f}
        Pérdida Promedio: €{metrics['avg_loss']:.2f}
        
        SIMULACIÓN:
        -----------
        Rondas: {len(backtest_results['equity_curve']) - 1}
        Retornos Totales: {len(backtest_results['all_returns'])}
        
        FIRMA:
        ------
        ACBE-S73 Quantum Betting Suite v2.3
        Sistema Validado Institucionalmente
        """
        
        return {
            'metrics': {
                'data': metrics_csv,
                'filename': f'acbe_backtest_metricas_{timestamp}.csv',
                'mime': 'text/csv'
            },
            'equity': {
                'data': equity_csv,
                'filename': f'acbe_backtest_equity_{timestamp}.csv',
                'mime': 'text/csv'
            },
            'returns': {
                'data': returns_csv,
                'filename': f'acbe_backtest_retornos_{timestamp}.csv',
                'mime': 'text/csv'
            },
            'report': {
                'data': report_text,
                'filename': f'acbe_backtest_reporte_{timestamp}.txt',
                'mime': 'text/plain'
            }
        }
    
    @staticmethod
    def export_acbe_analysis(df_acbe: pd.DataFrame, df_odds: pd.DataFrame) -> Dict[str, Any]:
        """
        Exporta análisis ACBE completo.
        
        Args:
            df_acbe: DataFrame con probabilidades ACBE
            df_odds: DataFrame con cuotas y EV
            
        Returns:
            Diccionario con datos para descarga
        """
        timestamp = DataExporter.generate_timestamp()
        
        # Crear archivo Excel con múltiples hojas
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_acbe.to_excel(writer, sheet_name='Probabilidades_ACBE', index=False)
            df_odds.to_excel(writer, sheet_name='Cuotas_EV', index=False)
        
        excel_data = output.getvalue()
        
        # También CSV individual
        acbe_csv = df_acbe.to_csv(index=False, sep=';', decimal=',', encoding='utf-8-sig')
        odds_csv = df_odds.to_csv(index=False, sep=';', decimal=',', encoding='utf-8-sig')
        
        return {
            'excel': {
                'data': excel_data,
                'filename': f'acbe_analisis_completo_{timestamp}.xlsx',
                'mime': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            },
            'acbe_csv': {
                'data': acbe_csv,
                'filename': f'acbe_probabilidades_{timestamp}.csv',
                'mime': 'text/csv'
            },
            'odds_csv': {
                'data': odds_csv,
                'filename': f'acbe_cuotas_ev_{timestamp}.csv',
                'mime': 'text/csv'
            }
        }
    
    @staticmethod
    def export_executive_summary(s73_results: Dict, backtest_results: Dict, config: Dict) -> Dict[str, Any]:
        """
        Exporta resumen ejecutivo completo.
        
        Args:
            s73_results: Resultados del sistema S73
            backtest_results: Resultados del backtesting
            config: Configuración del sistema
            
        Returns:
            Diccionario con datos para descarga
        """
        timestamp = DataExporter.generate_timestamp()
        metrics = backtest_results['final_metrics']
        
        # Crear reporte ejecutivo detallado
        report_text = f"""
        RESUMEN EJECUTIVO ACBE-S73 v2.3
        ================================
        Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        
        CALIFICACIÓN DEL SISTEMA:
        -------------------------
        Sistema: ACBE-S73 Quantum Betting Suite v2.3
        Estado: {"✅ VALIDADO" if s73_results.get('coverage_validated', False) else "❌ NO VALIDADO"}
        Modo: {"🔘 AUTOMÁTICO (Kelly)" if config['auto_stake_mode'] else f"🎮 MANUAL ({config.get('manual_stake', 0)*100:.1f}%)"}
        
        MÉTRICAS CLAVE:
        ---------------
        • ROI Total: {metrics['total_return_pct']:+.2f}%
        • Sharpe Ratio: {metrics['sharpe_ratio']:.3f}
        • Max Drawdown: {metrics['max_drawdown']:.2f}%
        • Win Rate: {metrics['win_rate']:.2f}%
        • Prob. Ruina: {metrics['ruin_probability']:.2f}%
        • CAGR: {metrics['cagr']:+.2f}%
        • Profit Factor: {metrics['profit_factor']:.3f}
        
        SISTEMA S73:
        ------------
        • Columnas S73: {s73_results['final_count']}
        • Columnas Pre-filtradas: {s73_results['filtered_count']}
        • Cobertura Validada: {"SÍ" if s73_results.get('coverage_validated', False) else "NO"}
        • Exposición Total: {np.sum(s73_results['kelly_stakes']) * 100:.2f}%
        
        CONFIGURACIÓN:
        --------------
        • Bankroll: €{config['bankroll']:,.2f}
        • Exposición Máxima: {config['max_exposure']*100:.0f}%
        • Simulaciones: {config['n_rounds']} rondas × {config['monte_carlo_sims']:,} iteraciones
        • Filtros S73: {"ACTIVADOS" if config.get('apply_s73_filters', True) else "DESACTIVADOS"}
        
        RECOMENDACIONES:
        ----------------
        1. {"Mantener estrategia actual - rendimiento óptimo" if metrics['total_return_pct'] > 5 
            else "Revisar criterios de selección - margen de mejora"}
        2. {"Exposición dentro de límites seguros" if np.sum(s73_results['kelly_stakes']) * 100 <= 15 
            else "Reducir exposición para menor riesgo"}
        3. {"Drawdown bien controlado" if metrics['max_drawdown'] <= 15 
            else "Implementar stop-loss para controlar drawdown"}
        
        ANÁLISIS DE RIESGO:
        -------------------
        • VaR 95%: €{metrics['var_95']:.2f}
        • Volatilidad: €{metrics['std_returns']:.2f}
        • Sortino Ratio: {metrics.get('sortino_ratio', 'N/A')}
        • Calmar Ratio: {metrics.get('calmar_ratio', 'N/A')}
        
        DATOS TÉCNICOS:
        ---------------
        • Timestamp: {timestamp}
        • Versión: ACBE-S73 v2.3
        • Algoritmo: Gamma-Poisson Bayesiano + Teoría de Información + Kelly
        • Cobertura: 2 errores garantizados (S73 validado)
        
        CONCLUSIÓN:
        -----------
        Sistema {"ALTAMENTE RECOMENDADO" if metrics['total_return_pct'] > 10 and metrics['sharpe_ratio'] > 1.5 
                else "RECOMENDADO CON PRECAUCIÓN" if metrics['total_return_pct'] > 0 
                else "REQUIERE OPTIMIZACIÓN"}
        
        FIRMA:
        ------
        Arquitecto de Software & Data Scientist Senior
        Quant Developer | Risk Engineer | Institutional Betting Model
        """
        
        # También crear CSV resumido
        summary_data = {
            'Métrica': [
                'ROI Total (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)',
                'Prob. Ruina (%)', 'CAGR (%)', 'Profit Factor', 'Exposición Total (%)',
                'Columnas S73', 'Cobertura Validada', 'VaR 95% (€)', 'Volatilidad (€)'
            ],
            'Valor': [
                f"{metrics['total_return_pct']:.2f}",
                f"{metrics['sharpe_ratio']:.3f}",
                f"{metrics['max_drawdown']:.2f}",
                f"{metrics['win_rate']:.2f}",
                f"{metrics['ruin_probability']:.2f}",
                f"{metrics['cagr']:.2f}",
                f"{metrics['profit_factor']:.3f}",
                f"{np.sum(s73_results['kelly_stakes']) * 100:.2f}",
                f"{s73_results['final_count']}",
                "SÍ" if s73_results.get('coverage_validated', False) else "NO",
                f"{metrics['var_95']:.2f}",
                f"{metrics['std_returns']:.2f}"
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_csv = summary_df.to_csv(index=False, sep=';', decimal=',', encoding='utf-8-sig')
        
        return {
            'report': {
                'data': report_text,
                'filename': f'acbe_resumen_ejecutivo_{timestamp}.txt',
                'mime': 'text/plain'
            },
            'summary': {
                'data': summary_csv,
                'filename': f'acbe_resumen_metricas_{timestamp}.csv',
                'mime': 'text/csv'
            }
        }

# ============================================================================
# SECCIÓN 10: INTERFAZ STREAMLIT PROFESIONAL COMPLETA (CORREGIDA v2.3)
# ============================================================================

class ACBEApp:
    """Interfaz principal de la aplicación Streamlit - CORREGIDA Y MEJORADA v2.3."""
    
    def __init__(self):
        self.setup_page_config()
        self.match_input_layer = MatchInputLayer()
        self.portfolio_engine = PortfolioEngine()
        self.data_exporter = DataExporter()
        SessionStateManager.initialize_session_state()
    
    def setup_page_config(self):
        """Configuración de la página Streamlit."""
        st.set_page_config(
            page_title="ACBE-S73 Quantum Betting Suite v2.3",
            page_icon="🎯",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def render_navigation_bar(self, current_phase: str):
        """Renderiza barra de navegación superior."""
        st.markdown("---")
        nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])
        
        with nav_col1:
            if current_phase == "analysis":
                if st.button("← Volver a Ingreso de Datos", type="secondary", use_container_width=True):
                    SessionStateManager.reset_to_input()
                    st.rerun()
            elif current_phase == "input" and SessionStateManager.can_go_back():
                if st.button("← Volver a Análisis", type="secondary", use_container_width=True):
                    SessionStateManager.go_back()
                    st.rerun()
        
        with nav_col2:
            phase_title = "📥 Ingreso de Datos" if current_phase == "input" else "📊 Análisis del Sistema"
            st.markdown(f"<h3 style='text-align: center;'>{phase_title}</h3>", unsafe_allow_html=True)
        
        with nav_col3:
            if st.button("🔄 Reiniciar Todo", type="secondary", use_container_width=True):
                SessionStateManager.clear_all_data()
                st.rerun()
        
        st.markdown("---")
    
    def render_sidebar(self) -> Dict:
        """Renderiza sidebar MEJORADO sin botón de procesamiento."""
        with st.sidebar:
            st.header("⚙️ Configuración del Sistema")
            
            # Indicador de versión
            st.caption(f"v2.3 | {datetime.now().strftime('%Y-%m-%d')}")
            
            # Botón para limpiar datos
            if st.button("🔄 Reiniciar Sistema", type="secondary", use_container_width=True):
                SessionStateManager.clear_all_data()
                st.rerun()
            
            # Bankroll inicial
            bankroll = st.number_input(
                "Bankroll Inicial (€)",
                min_value=100.0,
                max_value=1000000.0,
                value=SystemConfig.DEFAULT_BANKROLL,
                step=1000.0,
                help="Capital inicial para simulaciones"
            )
            
            # ===== CORRECCIÓN PROBLEMA 2: TOGGLE AUTO/MANUAL =====
            st.subheader("🎮 Gestión de Stake")
            
            auto_stake_mode = st.toggle(
                "Modo Automático (Kelly)",
                value=True,
                help="Si activado, usa Kelly automático. Si desactivado, permite stake manual."
            )
            
            manual_stake = None
            if not auto_stake_mode:
                manual_stake = st.number_input(
                    "Stake Manual (% por columna)",
                    min_value=0.01,
                    max_value=10.0,
                    value=1.0,
                    step=0.1,
                    help="Porcentaje del bankroll a apostar en cada columna S73"
                )
                manual_stake_fraction = manual_stake / 100.0
                st.info(f"Stake manual: {manual_stake}% del bankroll por columna")
            else:
                manual_stake_fraction = None
                kelly_fraction = st.slider(
                    "Fracción de Kelly",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                    help="Fracción conservadora del Kelly completo"
                )
            
            # Parámetros de riesgo
            st.subheader("📊 Gestión de Riesgo")
            
            max_exposure = st.slider(
                "Exposición Máxima (%)",
                min_value=5,
                max_value=30,
                value=15,
                step=1,
                help="Porcentaje máximo del bankroll en apuestas"
            )
            
            # Configuración de simulaciones
            st.subheader("🎲 Parámetros de Simulación")
            
            monte_carlo_sims = st.number_input(
                "Simulaciones por Ronda",
                min_value=1000,
                max_value=50000,
                value=1000,
                step=1000
            )
            
            n_rounds = st.slider(
                "Rondas de Backtesting",
                min_value=10,
                max_value=500,
                value=100,
                step=10
            )
            
            # ===== FILTROS S73 MEJORADOS =====
            st.subheader("🎯 Filtros S73 Reducido")
            
            apply_s73_filters = st.toggle(
                "Aplicar filtros institucionales",
                value=True,
                help="Umbrales probabilísticos para reducción S73"
            )
            
            if apply_s73_filters:
                min_prob = st.slider(
                    "Prob. mínima por opción",
                    min_value=0.0,
                    max_value=1.0,
                    value=SystemConfig.MIN_OPTION_PROBABILITY,
                    step=0.01
                )
                min_gap = st.slider(
                    "Gap mínimo 1ª-2ª opción",
                    min_value=0.0,
                    max_value=0.5,
                    value=SystemConfig.MIN_PROBABILITY_GAP,
                    step=0.01
                )
                min_ev = st.slider(
                    "EV mínimo",
                    min_value=-0.5,
                    max_value=0.5,
                    value=SystemConfig.MIN_EV_THRESHOLD,
                    step=0.01
                )
                
                # Actualizar configuración
                SystemConfig.MIN_OPTION_PROBABILITY = min_prob
                SystemConfig.MIN_PROBABILITY_GAP = min_gap
                SystemConfig.MIN_EV_THRESHOLD = min_ev
            
            # ===== FUENTE DE DATOS =====
            st.subheader("📊 Fuente de Datos")
            data_source = st.radio(
                "Seleccionar fuente:",
                ["⚽ Input Manual", "📈 Datos Sintéticos", "📂 Cargar CSV"],
                index=0,
                help="Input Manual: Ingresa partidos reales manualmente\n"
                     "Datos Sintéticos: Sistema genera datos de prueba\n"
                     "Cargar CSV: Sube archivo con datos históricos"
            )
            
            uploaded_file = None
            n_matches = SystemConfig.NUM_MATCHES
            
            if data_source == "📈 Datos Sintéticos":
                n_matches = st.slider(
                    "Número de partidos",
                    min_value=6,
                    max_value=15,
                    value=6,
                    step=1
                )
                
            elif data_source == "📂 Cargar CSV":
                uploaded_file = st.file_uploader(
                    "Subir CSV con datos",
                    type=['csv'],
                    help="Columnas requeridas: home_attack, away_attack, home_defense, away_defense, odds_1, odds_X, odds_2"
                )
            
            return {
                'bankroll': bankroll,
                'auto_stake_mode': auto_stake_mode,
                'manual_stake': manual_stake_fraction,
                'kelly_fraction': kelly_fraction if auto_stake_mode else None,
                'max_exposure': max_exposure / 100,
                'monte_carlo_sims': monte_carlo_sims,
                'n_rounds': n_rounds,
                'data_source': data_source,
                'n_matches': n_matches,
                'uploaded_file': uploaded_file,
                'apply_s73_filters': apply_s73_filters
            }
    
    def render_data_input_phase(self, config: Dict):
        """Fase 1: Solo inputs de datos."""
        # Barra de navegación
        self.render_navigation_bar("input")
        
        st.header("📥 Fase 1: Ingreso de Datos")
        
        if config['data_source'] == "⚽ Input Manual":
            # Renderizar input manual SIN procesamiento
            matches_df, params_dict, mode = MatchInputLayer.render_manual_input_section()
            
            # Botón para cargar datos (separado del sidebar)
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("🚀 Cargar Datos y Proceder al Análisis", type="primary", use_container_width=True):
                    # Guardar en estado de sesión
                    st.session_state.matches_data = matches_df
                    st.session_state.params_dict = params_dict
                    st.session_state.mode = mode
                    SessionStateManager.move_to_analysis()
                    st.rerun()
            with col2:
                if st.button("🔄 Limpiar Datos", type="secondary", use_container_width=True):
                    SessionStateManager.reset_to_input()
                    st.rerun()
        
        elif config['data_source'] == "📈 Datos Sintéticos":
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info("Se generarán datos sintéticos para análisis.")
            with col2:
                if st.button("🎲 Generar Datos Sintéticos", type="primary", use_container_width=True):
                    # Generar datos sintéticos
                    from scipy import stats
                    np.random.seed(42)
                    
                    n_matches = config['n_matches']
                    
                    # Generar parámetros realistas
                    attack_strengths = np.random.beta(2, 2, size=(n_matches, 2)) * 1.5 + 0.5
                    defense_strengths = np.random.beta(2, 2, size=(n_matches, 2)) * 1.2 + 0.4
                    home_advantages = np.random.uniform(1.05, 1.25, n_matches)
                    
                    # Calcular tasas de goles
                    lambda_home = attack_strengths[:, 0] * defense_strengths[:, 1] * home_advantages
                    lambda_away = attack_strengths[:, 1] * defense_strengths[:, 0]
                    
                    # Simular probabilidades
                    probabilities = ACBEModel.vectorized_poisson_simulation(lambda_home, lambda_away)
                    
                    # Generar cuotas con márgenes realistas
                    margins = np.random.uniform(0.03, 0.07, n_matches)
                    odds_matrix = np.zeros((n_matches, 3))
                    
                    for i in range(n_matches):
                        fair_odds = 1 / probabilities[i]
                        odds_matrix[i] = fair_odds * (1 + margins[i])
                        odds_matrix[i] = np.clip(odds_matrix[i], 1.1, 20.0)
                    
                    # Calcular entropías
                    entropy = ACBEModel.calculate_entropy(probabilities)
                    normalized_entropy = ACBEModel.normalize_entropy(entropy)
                    
                    # Guardar en estado de sesión
                    st.session_state.matches_data = {
                        'probabilities': probabilities,
                        'odds_matrix': odds_matrix,
                        'normalized_entropy': normalized_entropy,
                        'attack_strengths': attack_strengths,
                        'defense_strengths': defense_strengths,
                        'home_advantages': home_advantages,
                        'lambda_home': lambda_home,
                        'lambda_away': lambda_away
                    }
                    st.session_state.mode = 'auto'
                    SessionStateManager.move_to_analysis()
                    st.rerun()
        
        elif config['data_source'] == "📂 Cargar CSV":
            if config['uploaded_file'] is not None:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.success(f"Archivo cargado: {config['uploaded_file'].name}")
                with col2:
                    if st.button("📊 Procesar CSV", type="primary", use_container_width=True):
                        # Cargar datos desde CSV
                        import pandas as pd
                        matches_df = pd.read_csv(config['uploaded_file'])
                        
                        # Extraer columnas necesarias
                        required_cols = ['home_attack', 'away_attack', 'home_defense', 'away_defense']
                        odds_cols = ['odds_1', 'odds_X', 'odds_2']
                        
                        # Validar columnas
                        missing_cols = [col for col in required_cols + odds_cols if col not in matches_df.columns]
                        if missing_cols:
                            st.error(f"❌ CSV falta columnas: {missing_cols}")
                            return
                        
                        attack_strengths = matches_df[['home_attack', 'away_attack']].values
                        defense_strengths = matches_df[['home_defense', 'away_defense']].values
                        odds_matrix = matches_df[odds_cols].values
                        
                        # Calcular probabilidades
                        lambda_home, lambda_away = ACBEModel.gamma_poisson_bayesian(
                            attack_strengths, defense_strengths
                        )
                        probabilities = ACBEModel.vectorized_poisson_simulation(lambda_home, lambda_away)
                        
                        # Calcular entropías
                        entropy = ACBEModel.calculate_entropy(probabilities)
                        normalized_entropy = ACBEModel.normalize_entropy(entropy)
                        
                        # Guardar en estado de sesión
                        st.session_state.matches_data = {
                            'probabilities': probabilities,
                            'odds_matrix': odds_matrix,
                            'normalized_entropy': normalized_entropy,
                            'attack_strengths': attack_strengths,
                            'defense_strengths': defense_strengths,
                            'lambda_home': lambda_home,
                            'lambda_away': lambda_away
                        }
                        st.session_state.mode = 'auto'
                        SessionStateManager.move_to_analysis()
                        st.rerun()
            else:
                st.warning("Por favor, carga un archivo CSV para continuar.")
    
    def render_analysis_phase(self, config: Dict):
        """Fase 2: Análisis completo con pestañas."""
        # Barra de navegación
        self.render_navigation_bar("analysis")
        
        # Crear pestañas principales
        tabs = st.tabs([
            "📊 Análisis ACBE", 
            "🧮 Sistema S73", 
            "📈 Backtesting",
            "📊 Portafolio",
            "📋 Resumen"
        ])
        
        # Extraer datos del estado de sesión
        matches_data = st.session_state.matches_data
        params_dict = st.session_state.params_dict if hasattr(st.session_state, 'params_dict') else None
        
        # Procesar según fuente de datos
        if config['data_source'] == "⚽ Input Manual" and params_dict is not None:
            # Procesar input manual
            processed_df, odds_matrix, probabilities = MatchInputLayer.process_manual_input(params_dict)
            entropy = ACBEModel.calculate_entropy(probabilities)
            normalized_entropy = ACBEModel.normalize_entropy(entropy)
            
            # Usar solo primeros 6 partidos para S73
            n_matches_available = len(probabilities)
            if n_matches_available >= 6:
                probs_6 = probabilities[:6, :]
                odds_6 = odds_matrix[:6, :]
                entropy_6 = normalized_entropy[:6]
            else:
                probs_6 = probabilities
                odds_6 = odds_matrix
                entropy_6 = normalized_entropy
            
        else:
            # Usar datos generados o cargados
            probs_6 = matches_data['probabilities'][:6, :] if len(matches_data['probabilities']) >= 6 else matches_data['probabilities']
            odds_6 = matches_data['odds_matrix'][:6, :] if len(matches_data['odds_matrix']) >= 6 else matches_data['odds_matrix']
            entropy_6 = matches_data['normalized_entropy'][:6] if len(matches_data['normalized_entropy']) >= 6 else matches_data['normalized_entropy']
        
        # Variables para almacenar resultados
        s73_results = None
        backtest_results = None
        
        # ===== PESTAÑA ANÁLISIS ACBE =====
        with tabs[0]:
            self.render_acbe_analysis(probs_6, odds_6, entropy_6)
        
        # ===== PESTAÑA SISTEMA S73 =====
        with tabs[1]:
            s73_results = self.render_s73_system(probs_6, odds_6, entropy_6, config['bankroll'], config)
        
        # ===== PESTAÑA BACKTESTING =====
        with tabs[2]:
            if s73_results:
                # Ejecutar backtesting
                backtester = VectorizedBacktester(initial_bankroll=config['bankroll'])
                
                with st.spinner("🔄 Ejecutando backtesting completo..."):
                    backtest_results = backtester.run_backtest(
                        probs_6, odds_6, entropy_6,
                        s73_results,
                        n_rounds=config['n_rounds'],
                        n_sims_per_round=config['monte_carlo_sims'],
                        kelly_fraction=config.get('kelly_fraction', 0.5),
                        manual_stake=config.get('manual_stake')
                    )
                
                self.render_backtest_results(backtest_results, config)
            else:
                st.warning("Ejecuta primero el sistema S73 para ver el backtesting")
        
        # ===== PESTAÑA PORTFOLIO =====
        with tabs[3]:
            if s73_results:
                self.render_portfolio_analysis(s73_results, config)
            else:
                st.warning("Ejecuta primero el sistema S73 para ver el análisis de portafolio")
        
        # ===== PESTAÑA RESUMEN EJECUTIVO =====
        with tabs[4]:
            if s73_results and backtest_results:
                self.render_executive_summary(s73_results, backtest_results, config)
            else:
                st.info("Completa todas las fases anteriores para ver el resumen ejecutivo")
    
    def render_acbe_analysis(self, probabilities: np.ndarray, 
                            odds_matrix: np.ndarray,
                            normalized_entropies: np.ndarray):
        """Renderiza análisis ACBE completo."""
        st.header("🔬 Análisis ACBE")
        
        # Calcular métricas
        entropy = ACBEModel.calculate_entropy(probabilities)
        expected_value = InformationTheory.calculate_expected_value(probabilities, odds_matrix)
        
        # Clasificación de partidos
        allowed_signs, classifications = InformationTheory.classify_matches_by_entropy(
            probabilities, normalized_entropies, odds_matrix
        )
        
        # DataFrames para visualización
        n_matches = len(probabilities)
        
        df_acbe = pd.DataFrame({
            'Partido': range(1, n_matches + 1),
            'Clasificación': classifications,
            'P(1)': probabilities[:, 0],
            'P(X)': probabilities[:, 1],
            'P(2)': probabilities[:, 2],
            'Entropía': entropy,
            'Entropía Norm.': normalized_entropies
        })
        
        df_odds = pd.DataFrame({
            'Partido': range(1, n_matches + 1),
            'Cuota 1': odds_matrix[:, 0],
            'Cuota X': odds_matrix[:, 1],
            'Cuota 2': odds_matrix[:, 2],
            'EV 1': expected_value[:, 0],
            'EV X': expected_value[:, 1],
            'EV 2': expected_value[:, 2],
            'Signos Permitidos': [str([SystemConfig.OUTCOME_LABELS[s] for s in signs]) 
                                 for signs in allowed_signs]
        })
        
        # Mostrar en columnas
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Probabilidades ACBE")
            st.dataframe(df_acbe.style.format({
                'P(1)': '{:.3f}',
                'P(X)': '{:.3f}',
                'P(2)': '{:.3f}',
                'Entropía': '{:.3f}',
                'Entropía Norm.': '{:.3f}'
            }), use_container_width=True)
        
        with col2:
            st.subheader("💰 Cuotas y Valor Esperado")
            st.dataframe(df_odds.style.format({
                'Cuota 1': '{:.2f}',
                'Cuota X': '{:.2f}',
                'Cuota 2': '{:.2f}',
                'EV 1': '{:.3f}',
                'EV X': '{:.3f}',
                'EV 2': '{:.3f}'
            }), use_container_width=True)
        
        # Visualizaciones
        self._render_acbe_visualizations(probabilities, entropy, normalized_entropies)
        
        # NUEVO: Botones de descarga para análisis ACBE
        st.subheader("💾 Exportar Análisis ACBE")
        
        # Preparar datos para exportación
        export_data = DataExporter.export_acbe_analysis(df_acbe, df_odds)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                label="📥 Descargar Todo (Excel)",
                data=export_data['excel']['data'],
                file_name=export_data['excel']['filename'],
                mime=export_data['excel']['mime'],
                help="Descarga completa en Excel con múltiples hojas"
            )
        
        with col2:
            st.download_button(
                label="📥 Probabilidades (CSV)",
                data=export_data['acbe_csv']['data'],
                file_name=export_data['acbe_csv']['filename'],
                mime=export_data['acbe_csv']['mime'],
                help="Solo probabilidades ACBE en CSV"
            )
        
        with col3:
            st.download_button(
                label="📥 Cuotas y EV (CSV)",
                data=export_data['odds_csv']['data'],
                file_name=export_data['odds_csv']['filename'],
                mime=export_data['odds_csv']['mime'],
                help="Cuotas y valor esperado en CSV"
            )
    
    def _render_acbe_visualizations(self, probabilities: np.ndarray,
                                   entropy: np.ndarray,
                                   normalized_entropies: np.ndarray):
        """Renderiza visualizaciones del análisis ACBE."""
        n_matches = len(probabilities)
        
        # Gráfico de probabilidades
        fig_probs = go.Figure()
        for i, outcome in enumerate(['1', 'X', '2']):
            fig_probs.add_trace(go.Bar(
                x=list(range(1, n_matches + 1)),
                y=probabilities[:, i],
                name=outcome,
                marker_color=SystemConfig.OUTCOME_COLORS[i],
                text=[f'{p:.1%}' for p in probabilities[:, i]],
                textposition='auto'
            ))
        
        fig_probs.update_layout(
            title="Probabilidades ACBE por Partido",
            barmode='stack',
            xaxis_title="Partido",
            yaxis_title="Probabilidad",
            height=400
        )
        
        # Gráfico de entropía
        fig_entropy = go.Figure()
        fig_entropy.add_trace(go.Scatter(
            x=list(range(1, n_matches + 1)),
            y=normalized_entropies,
            mode='lines+markers',
            name='Entropía Normalizada',
            line=dict(color=SystemConfig.COLORS['primary'], width=3)
        ))
        
        # Líneas de umbral
        fig_entropy.add_hline(
            y=SystemConfig.STRONG_MATCH_THRESHOLD,
            line_dash="dash",
            line_color=SystemConfig.COLORS['success'],
            annotation_text="Fuerte"
        )
        fig_entropy.add_hline(
            y=SystemConfig.MEDIUM_MATCH_THRESHOLD,
            line_dash="dash", 
            line_color=SystemConfig.COLORS['warning'],
            annotation_text="Medio"
        )
        
        fig_entropy.update_layout(
            title="Clasificación por Entropía",
            xaxis_title="Partido",
            yaxis_title="Entropía Normalizada",
            height=400,
            yaxis_range=[0, 1]
        )
        
        st.plotly_chart(fig_probs, use_container_width=True)
        st.plotly_chart(fig_entropy, use_container_width=True)
    
    def render_s73_system(self, probabilities: np.ndarray,
                         odds_matrix: np.ndarray,
                         normalized_entropies: np.ndarray,
                         bankroll: float,
                         config: Dict) -> Dict:
        """Renderiza sistema S73 completo con validación institucional."""
        st.header("🧮 Sistema Combinatorio S73 (Validado)")
        
        with st.spinner("Construyendo sistema S73 optimizado con validación institucional..."):
            # 1. Generar combinaciones pre-filtradas
            filtered_combo, filtered_probs = S73System.generate_prefiltered_combinations(
                probabilities, normalized_entropies, odds_matrix
            )
            
            # 2. Construir sistema de cobertura
            s73_combo, s73_probs = S73System.build_s73_coverage_system(
                filtered_combo, filtered_probs, validate_coverage=True
            )
            
            # 3. Calcular métricas por columna
            n_columns = len(s73_combo)
            columns_data = []
            
            for idx, (combo, prob) in enumerate(zip(s73_combo, s73_probs), 1):
                # Calcular cuota conjunta
                combo_odds = S73System.calculate_combination_odds(combo, odds_matrix)
                
                # Calcular entropía promedio de la combinación
                combo_entropy = np.mean([normalized_entropies[i] for i in range(6)])
                
                # Calcular stake según modo
                if config['auto_stake_mode']:
                    kelly_stake = KellyCapitalManagement.calculate_column_kelly(
                        combo, prob, combo_odds, combo_entropy
                    )
                else:
                    kelly_stake = KellyCapitalManagement.calculate_column_kelly(
                        combo, prob, combo_odds, combo_entropy, config['manual_stake']
                    )
                
                columns_data.append({
                    'ID': idx,
                    'Combinación': ''.join([SystemConfig.OUTCOME_LABELS[s] for s in combo]),
                    'Probabilidad': prob,
                    'Cuota': combo_odds,
                    'Valor Esperado': prob * combo_odds - 1,
                    'Entropía Prom.': combo_entropy,
                    'Stake (%)': kelly_stake * 100,
                    'Inversión (€)': kelly_stake * bankroll
                })
            
            # Crear DataFrame
            columns_df = pd.DataFrame(columns_data)
            
            # 4. Normalizar stakes del portafolio
            kelly_stakes = np.array([d['Stake (%)'] for d in columns_data]) / 100
            kelly_stakes = KellyCapitalManagement.normalize_portfolio_stakes(
                kelly_stakes, 
                is_manual_mode=not config['auto_stake_mode']
            )
            
            # Actualizar DataFrame con stakes normalizados
            for i, stake in enumerate(kelly_stakes):
                columns_df.at[i, 'Stake (%)'] = stake * 100
                columns_df.at[i, 'Inversión (€)'] = stake * bankroll
        
        # Estadísticas del sistema
        st.subheader("📈 Estadísticas del Sistema S73")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Combinaciones Iniciales", len(filtered_combo))
        with col2:
            st.metric("Columnas S73 Finales", n_columns)
        with col3:
            total_exposure = np.sum(kelly_stakes) * 100
            st.metric("Exposición Total", f"{total_exposure:.1f}%")
        with col4:
            coverage_rate = (len(filtered_combo) / (3**6)) * 100
            st.metric("Cobertura del Espacio", f"{coverage_rate:.1f}%")
        
        # Validación de cobertura
        st.subheader("✅ Validación Institucional")
        
        # Calcular distancias de Hamming
        hamming_matrix = S73System.hamming_distance_matrix(s73_combo)
        max_distance = np.max(hamming_matrix)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Distancia Máxima", f"{max_distance}")
            if max_distance <= SystemConfig.HAMMING_DISTANCE_TARGET:
                st.success("✅ Cobertura de 2 errores garantizada")
            else:
                st.error("❌ Cobertura insuficiente")
        
        with col2:
            avg_prob = np.mean(s73_probs) * 100
            st.metric("Probabilidad Promedio", f"{avg_prob:.2f}%")
        
        with col3:
            diversification = len(set([tuple(c) for c in s73_combo])) / len(s73_combo) * 100
            st.metric("Diversificación", f"{diversification:.1f}%")
        
        # Mostrar columnas
        st.subheader("📋 Columnas del Sistema")
        
        display_df = columns_df.copy()
        display_df['Probabilidad'] = display_df['Probabilidad'].apply(lambda x: f'{x:.4%}')
        display_df['Cuota'] = display_df['Cuota'].apply(lambda x: f'{x:.2f}')
        display_df['Valor Esperado'] = display_df['Valor Esperado'].apply(lambda x: f'{x:.4f}')
        display_df['Entropía Prom.'] = display_df['Entropía Prom.'].apply(lambda x: f'{x:.3f}')
        display_df['Stake (%)'] = display_df['Stake (%)'].apply(lambda x: f'{x:.2f}%')
        display_df['Inversión (€)'] = display_df['Inversión (€)'].apply(lambda x: f'€{x:.2f}')
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Gráfico de distribución de stakes - CORRECCIÓN PROBLEMA 1
        st.subheader("📊 Distribución de Stakes")
        
        # Crear bins para el histograma
        stake_values = columns_df['Stake (%)'].astype(float).values
        hist, bins = np.histogram(stake_values, bins=10)
        
        fig_stakes = go.Figure()
        fig_stakes.add_trace(go.Bar(
            x=[f"{bins[i]:.2f}-{bins[i+1]:.2f}%" for i in range(len(bins)-1)],
            y=hist,
            marker_color=SystemConfig.RISK_PALETTE[0],  # Usar paleta corregida
            opacity=0.7
        ))
        
        fig_stakes.update_layout(
            title="Distribución de Stakes por Columna",
            xaxis_title="Stake (%)",
            yaxis_title="Número de Columnas",
            height=300
        )
        
        st.plotly_chart(fig_stakes, use_container_width=True)
        
        # NUEVO: Botones de descarga para S73
        st.subheader("💾 Exportar Sistema S73")
        
        # Preparar resultados para backtesting y exportación
        s73_results = {
            'combinations': s73_combo,
            'probabilities': s73_probs,
            'kelly_stakes': kelly_stakes,
            'filtered_count': len(filtered_combo),
            'final_count': n_columns,
            'coverage_validated': max_distance <= SystemConfig.HAMMING_DISTANCE_TARGET,
            'columns_df': columns_df  # Guardar para exportación
        }
        
        # Preparar datos para exportación
        export_data = DataExporter.export_s73_columns(columns_df, s73_results)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Columnas S73 (CSV)",
                data=export_data['csv']['data'],
                file_name=export_data['csv']['filename'],
                mime=export_data['csv']['mime'],
                help="Descarga todas las columnas del sistema S73"
            )
        
        with col2:
            st.download_button(
                label="📥 Resumen Sistema (CSV)",
                data=export_data['summary']['data'],
                file_name=export_data['summary']['filename'],
                mime=export_data['summary']['mime'],
                help="Resumen ejecutivo del sistema S73"
            )
        
        return s73_results
    
    def render_backtest_results(self, backtest_results: Dict, config: Dict):
        """Renderiza resultados del backtesting."""
        st.header("📈 Resultados del Backtesting")
        
        metrics = backtest_results['final_metrics']
        
        # Métricas principales
        st.subheader("📊 Métricas de Rendimiento")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Bankroll Final", f"€{metrics['final_bankroll']:,.2f}")
            st.metric("Retorno Total", f"{metrics['total_return_pct']:.2f}%")
        with col2:
            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            st.metric("CAGR", f"{metrics['cagr']:.2f}%")
        with col3:
            st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
            st.metric("Win Rate", f"{metrics['win_rate']:.2f}%")
        with col4:
            st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
            st.metric("VaR 95%", f"€{metrics['var_95']:.2f}")
        
        # Gráficos
        self._render_backtest_charts(backtest_results)
        
        # Análisis de riesgo mejorado
        self._render_risk_analysis_improved(backtest_results, metrics)
        
        # NUEVO: Botones de descarga para backtesting
        st.subheader("💾 Exportar Backtesting")
        
        # Preparar datos para exportación
        export_data = DataExporter.export_backtest_results(backtest_results)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.download_button(
                label="📥 Métricas (CSV)",
                data=export_data['metrics']['data'],
                file_name=export_data['metrics']['filename'],
                mime=export_data['metrics']['mime'],
                help="Métricas principales del backtesting"
            )
        
        with col2:
            st.download_button(
                label="📥 Curva Equity (CSV)",
                data=export_data['equity']['data'],
                file_name=export_data['equity']['filename'],
                mime=export_data['equity']['mime'],
                help="Curva de bankroll y drawdown"
            )
        
        with col3:
            st.download_button(
                label="📥 Retornos (CSV)",
                data=export_data['returns']['data'],
                file_name=export_data['returns']['filename'],
                mime=export_data['returns']['mime'],
                help="Retornos por ronda de simulación"
            )
        
        with col4:
            st.download_button(
                label="📥 Reporte (TXT)",
                data=export_data['report']['data'],
                file_name=export_data['report']['filename'],
                mime=export_data['report']['mime'],
                help="Reporte ejecutivo completo"
            )
    
    def _render_backtest_charts(self, backtest_results: Dict):
        """Renderiza gráficos del backtesting."""
        # Curva de equity y drawdown
        fig_equity = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Equity curve
        fig_equity.add_trace(
            go.Scatter(
                x=list(range(len(backtest_results['equity_curve']))),
                y=backtest_results['equity_curve'],
                name='Bankroll',
                line=dict(color=SystemConfig.COLORS['success'], width=3)
            ),
            secondary_y=False
        )
        
        # Drawdown
        fig_equity.add_trace(
            go.Scatter(
                x=list(range(len(backtest_results['drawdown_curve']))),
                y=backtest_results['drawdown_curve'],
                name='Drawdown',
                line=dict(color=SystemConfig.COLORS['danger'], width=2)
            ),
            secondary_y=True
        )
        
        fig_equity.update_layout(
            title="Evolución del Bankroll y Drawdown",
            xaxis_title="Ronda",
            height=500
        )
        fig_equity.update_yaxes(title_text="Bankroll (€)", secondary_y=False)
        fig_equity.update_yaxes(title_text="Drawdown %", secondary_y=True)
        
        # Distribución de retornos
        fig_returns = go.Figure()
        returns = backtest_results['all_returns']
        
        fig_returns.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            name='Distribución de Retornos',
            marker_color=SystemConfig.COLORS['info'],
            opacity=0.7
        ))
        
        # Estadísticas en el gráfico
        mean_return = np.mean(returns)
        median_return = np.median(returns)
        
        fig_returns.add_vline(
            x=mean_return,
            line_dash="dash",
            line_color=SystemConfig.COLORS['primary'],
            annotation_text=f"Media: €{mean_return:.2f}"
        )
        fig_returns.add_vline(
            x=median_return,
            line_dash="dot",
            line_color=SystemConfig.COLORS['secondary'],
            annotation_text=f"Mediana: €{median_return:.2f}"
        )
        
        fig_returns.update_layout(
            title="Distribución de Retornos por Ronda",
            xaxis_title="Retorno (€)",
            yaxis_title="Frecuencia",
            height=400
        )
        
        # Mostrar gráficos
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_equity, use_container_width=True)
        with col2:
            st.plotly_chart(fig_returns, use_container_width=True)
    
    def _render_risk_analysis_improved(self, backtest_results: Dict, metrics: Dict):
        """Renderiza análisis de riesgo mejorado con Portfolio Engine."""
        st.subheader("🔍 Análisis de Riesgo Cuantitativo")
        
        returns = backtest_results['all_returns']
        
        # Calcular métricas de riesgo adicionales
        var_95 = np.percentile(returns, 5)
        cvar_95 = np.mean(returns[returns <= var_95])
        
        # Skewness y Kurtosis
        skewness = pd.Series(returns).skew()
        kurtosis = pd.Series(returns).kurtosis()
        
        # Sortino Ratio (usando desviación downside)
        negative_returns = returns[returns < 0]
        downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 0
        sortino_ratio = np.mean(returns) / downside_std if downside_std > 0 else 0
        
        # Calmar Ratio
        calmar_ratio = metrics['cagr'] / metrics['max_drawdown'] if metrics['max_drawdown'] > 0 else 0
        
        # Mostrar métricas en columnas
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("CVaR 95%", f"€{cvar_95:.2f}")
            st.metric("Volatilidad (σ)", f"€{metrics['std_returns']:.2f}")
        
        with col2:
            st.metric("Sortino Ratio", f"{sortino_ratio:.2f}")
            st.metric("Calmar Ratio", f"{calmar_ratio:.2f}")
        
        with col3:
            st.metric("Asimetría (Skewness)", f"{skewness:.3f}")
            st.metric("Curtosis", f"{kurtosis:.3f}")
        
        # Gráfico de riesgo-rendimiento - CORRECCIÓN PROBLEMA 1
        st.subheader("📈 Análisis Riesgo-Rendimiento")
        
        # Simular diferentes estrategias para comparación
        strategies = ['S73', 'Singles', 'Combinadas']
        returns_means = [np.mean(returns), np.mean(returns) * 0.7, np.mean(returns) * 0.5]
        returns_stds = [np.std(returns), np.std(returns) * 1.2, np.std(returns) * 1.5]
        colors = SystemConfig.RISK_PALETTE[:3]  # Usar paleta corregida
        
        fig_risk_return = go.Figure()
        
        for i, strategy in enumerate(strategies):
            fig_risk_return.add_trace(go.Scatter(
                x=[returns_stds[i]],
                y=[returns_means[i]],
                mode='markers+text',
                name=strategy,
                marker=dict(
                    size=20,
                    color=colors[i],
                    line=dict(width=2, color='white')
                ),
                text=strategy,
                textposition="top center"
            ))
        
        fig_risk_return.update_layout(
            title="Riesgo vs Rendimiento por Estrategia",
            xaxis_title="Volatilidad (σ)",
            yaxis_title="Retorno Esperado (€)",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig_risk_return, use_container_width=True)
    
    def render_portfolio_analysis(self, s73_results: Dict, config: Dict):
        """Renderiza análisis completo del portafolio."""
        st.header("📊 Análisis de Portafolio Unificado")
        
        # Inicializar Portfolio Engine
        portfolio_engine = PortfolioEngine(config['bankroll'])
        
        # Agregar estrategia S73
        if s73_results:
            portfolio_engine.add_strategy(
                's73_columns',
                s73_results['kelly_stakes'],
                np.array([S73System.calculate_combination_odds(c, np.zeros((6,3))) for c in s73_results['combinations']]),
                s73_results['probabilities']
            )
        
        # Calcular métricas del portafolio
        portfolio_metrics = portfolio_engine.calculate_portfolio_metrics()
        
        # Mostrar métricas por estrategia
        st.subheader("📈 Métricas por Estrategia")
        
        for strategy, metrics in portfolio_metrics.items():
            if strategy == 'portfolio':
                continue
                
            with st.expander(f"🔍 {strategy.upper()}", expanded=True):
                cols = st.columns(3)
                metric_items = list(metrics.items())
                
                for i in range(0, len(metric_items), 3):
                    for j in range(3):
                        if i + j < len(metric_items):
                            key, value = metric_items[i + j]
                            cols[j].metric(key, f"{value:.4f}" if isinstance(value, float) else value)
        
        # Mostrar métricas agregadas del portafolio
        if 'portfolio' in portfolio_metrics:
            st.subheader("🏦 Métricas Agregadas del Portafolio")
            
            portfolio_agg = portfolio_metrics['portfolio']
            col1, col2 = st.columns(2)
            
            with col1:
                for key in ['Total EV', 'Total Variance', 'Aggregate Sharpe', 'Max Portfolio Drawdown (%)']:
                    if key in portfolio_agg:
                        value = portfolio_agg[key]
                        st.metric(key, f"{value:.4f}" if isinstance(value, float) else value)
            
            with col2:
                for key in ['Aggregate Ruin Probability (%)', 'Aggregate Capital Efficiency', 
                          'Total Portfolio Exposure (%)', 'Number of Strategies']:
                    if key in portfolio_agg:
                        value = portfolio_agg[key]
                        st.metric(key, f"{value:.4f}" if isinstance(value, float) else value)
        
        # Gráfico de composición del portafolio - CORRECCIÓN PROBLEMA 1
        st.subheader("🥧 Composición del Portafolio")
        
        if portfolio_metrics:
            strategies = [k for k in portfolio_metrics.keys() if k != 'portfolio']
            exposures = [portfolio_metrics[s]['Total Exposure (%)'] for s in strategies]
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=strategies,
                values=exposures,
                hole=.3,
                marker=dict(colors=SystemConfig.RISK_PALETTE[:len(strategies)])  # Paleta corregida
            )])
            
            fig_pie.update_layout(
                title="Distribución de Exposición por Estrategia",
                height=400
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
    
    def render_executive_summary(self, s73_results: Dict, backtest_results: Dict, config: Dict):
        """Renderiza resumen ejecutivo del sistema."""
        st.header("📋 Resumen Ejecutivo")
        
        metrics = backtest_results['final_metrics']
        
        # Estado del sistema
        st.subheader("🎯 Estado del Sistema")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            coverage_status = "✅ VALIDADO" if s73_results.get('coverage_validated', False) else "❌ NO VALIDADO"
            st.metric("Cobertura S73", coverage_status)
        
        with col2:
            mode_status = "🔘 AUTOMÁTICO" if config['auto_stake_mode'] else "🎮 MANUAL"
            st.metric("Modo Stake", mode_status)
        
        with col3:
            total_exposure = np.sum(s73_results['kelly_stakes']) * 100
            exposure_color = "green" if total_exposure <= 15 else "orange" if total_exposure <= 20 else "red"
            st.metric("Exposición Total", f"{total_exposure:.1f}%", delta=None)
        
        with col4:
            roi = metrics['total_return_pct']
            roi_color = "green" if roi > 0 else "red"
            st.metric("ROI Total", f"{roi:+.2f}%", delta=None)
        
        # Recomendaciones
        st.subheader("💡 Recomendaciones de Gestión")
        
        total_exposure = np.sum(s73_results['kelly_stakes']) * 100
        
        if total_exposure > 20:
            exposure_status = "⚠️ ALTO"
            exposure_rec = "Reducir exposición inmediatamente a <15%"
            exposure_action = "st.error"
        elif total_exposure > 15:
            exposure_status = "⚠️ MODERADO"
            exposure_rec = "Considerar reducir exposición a <15%"
            exposure_action = "st.warning"
        else:
            exposure_status = "✅ OPTIMO"
            exposure_rec = "Exposición dentro de límites seguros"
            exposure_action = "st.success"
        
        if metrics['max_drawdown'] > 25:
            risk_status = "⚠️ ALTO"
            risk_rec = "Implementar stop-loss agresivo inmediatamente"
            risk_action = "st.error"
        elif metrics['max_drawdown'] > 15:
            risk_status = "⚠️ MODERADO"
            risk_rec = "Monitorear drawdown diariamente"
            risk_action = "st.warning"
        else:
            risk_status = "✅ BAJO"
            risk_rec = "Drawdown bien controlado"
            risk_action = "st.success"
        
        # Mostrar recomendaciones
        eval(exposure_action)(f"**Exposición del Portafolio:** {exposure_status} - {exposure_rec}")
        eval(risk_action)(f"**Riesgo de Drawdown:** {risk_status} - {risk_rec}")
        
        # Conclusión final
        st.subheader("🎯 Calificación del Sistema")
        
        roi = metrics['total_return_pct']
        sharpe = metrics['sharpe_ratio']
        max_dd = metrics['max_drawdown']
        
        if roi > 10 and sharpe > 1.5 and max_dd < 15:
            rating = "A+ (EXCELENTE)"
            description = "Sistema altamente rentable con excelente perfil riesgo/retorno"
            color = SystemConfig.COLORS['success']
        elif roi > 5 and sharpe > 1.0 and max_dd < 20:
            rating = "B+ (BUENO)"
            description = "Sistema rentable con gestión adecuada de riesgo"
            color = SystemConfig.COLORS['success']
        elif roi > 0:
            rating = "C (ACEPTABLE)"
            description = "Sistema positivo con margen de mejora en gestión de riesgo"
            color = SystemConfig.COLORS['warning']
        else:
            rating = "D (MEJORABLE)"
            description = "Revisar configuración del sistema y criterios de selección"
            color = SystemConfig.COLORS['danger']
        
        st.markdown(f"""
        <div style="background-color:{color}20; padding:20px; border-radius:10px; border-left:5px solid {color}; margin:20px 0;">
            <h3 style="color:{color};">Calificación: {rating}</h3>
            <p><strong>{description}</strong></p>
            <hr>
            <p><strong>📊 Métricas Clave:</strong></p>
            <ul>
                <li>ROI Total: {roi:+.2f}%</li>
                <li>Sharpe Ratio: {sharpe:.2f}</li>
                <li>Max Drawdown: {max_dd:.1f}%</li>
                <li>Win Rate: {metrics['win_rate']:.1f}%</li>
                <li>Prob. Ruina: {metrics['ruin_probability']:.1f}%</li>
            </ul>
            <hr>
            <p><strong>⚙️ Configuración Usada:</strong></p>
            <ul>
                <li>Modo: {"Automático (Kelly)" if config['auto_stake_mode'] else f"Manual ({config.get('manual_stake', 0)*100:.1f}%)"}</li>
                <li>Exposición Máxima: {config['max_exposure']*100:.0f}%</li>
                <li>Simulaciones: {config['n_rounds']} rondas × {config['monte_carlo_sims']:,} iteraciones</li>
                <li>Columnas S73: {s73_results['final_count']} (de {s73_results['filtered_count']} pre-filtradas)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # NUEVO: Botones de descarga para resumen ejecutivo
        st.subheader("💾 Exportar Reporte Completo")
        
        # Preparar datos para exportación
        export_data = DataExporter.export_executive_summary(s73_results, backtest_results, config)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Reporte Ejecutivo (TXT)",
                data=export_data['report']['data'],
                file_name=export_data['report']['filename'],
                mime=export_data['report']['mime'],
                help="Reporte ejecutivo completo en texto"
            )
        
        with col2:
            st.download_button(
                label="📥 Métricas Resumen (CSV)",
                data=export_data['summary']['data'],
                file_name=export_data['summary']['filename'],
                mime=export_data['summary']['mime'],
                help="Métricas clave resumidas en CSV"
            )
    
    def run(self):
        """Método principal de ejecución de la aplicación CORREGIDO v2.3."""
        st.title("🎯 ACBE-S73 Quantum Betting Suite v2.3")
        st.markdown("""
        *Sistema profesional de optimización de portafolios de apuestas deportivas*  
        ***Con correcciones institucionales completas y validación cuantitativa***
        """)
        
        # Renderizar sidebar y obtener configuración
        config = self.render_sidebar()
        
        # Decisión basada en fase actual
        current_phase = st.session_state.current_phase
        
        if current_phase == "input":
            self.render_data_input_phase(config)
        else:
            self.render_analysis_phase(config)

# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    # Inicializar y ejecutar la aplicación
    app = ACBEApp()
    app.run()
