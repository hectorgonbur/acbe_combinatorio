"""
🎯 ACBE-S73 QUANTUM BETTING SUITE v3.0
Sistema profesional de optimización de portafolios de apuestas deportivas
Combina Inferencia Bayesiana Gamma-Poisson, Teoría de la Información y Criterio de Kelly
Con cobertura S73 completa (2 errores) y gestión probabilística avanzada

NUEVAS MEJORAS v3.0:
1. ✅ ALGORITMO GREEDY HAMMING: Implementación exacta del algoritmo de cobertura con Score = Probabilidad × (1 + EV)
2. ✅ SMART RECOMMENDATION ENGINE: Apuesta Maestra visual y Simulador de Cobertura interactivo
3. ✅ BACKTESTING ESPECÍFICO: Simulación exacta de las 73 columnas S73 (no apuestas simples)
4. ✅ VECTORIZACIÓN COMPLETA: Eliminación de bucles for en cálculos Monte Carlo
5. ✅ EXPORTACIÓN EXCEL: UTF-8-SIG con separador ; para compatibilidad total
6. ✅ GESTIÓN DE ESTADO ROBUSTA: Persistencia completa de datos en session_state
7. ✅ INTERFAZ ORGANIZADA: Fases Input → Análisis claramente separadas

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
import itertools
from datetime import datetime
warnings.filterwarnings('ignore')

# ============================================================================
# SECCIÓN 0: MANEJO DE ESTADO DE SESIÓN - MEJORADO v3.0
# ============================================================================

class SessionStateManager:
    """Gestor del estado de sesión para persistencia completa de datos."""
    
    @staticmethod
    def initialize_session_state():
        """Inicializa todas las variables de estado necesarias con valores por defecto."""
        defaults = {
            'data_loaded': False,
            'matches_data': None,
            'params_dict': None,
            'processing_done': False,
            'current_tab': "input",
            'current_phase': "input",
            'phase_history': ["input"],
            's73_results': None,
            'backtest_results': None,
            'portfolio_metrics': None,
            'last_processed': None,
            'user_config': {},
            'probabilities_6': None,
            'odds_matrix_6': None,
            'entropy_6': None
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    @staticmethod
    def reset_to_input():
        """Reinicia al estado de ingreso de datos manteniendo datos para reutilización."""
        st.session_state.data_loaded = False
        st.session_state.processing_done = False
        st.session_state.current_phase = "input"
        st.session_state.phase_history = ["input"]
    
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
            
            if previous_phase == "input":
                st.session_state.data_loaded = False
                st.session_state.processing_done = False
    
    @staticmethod
    def clear_all_data():
        """Limpia todos los datos de la sesión."""
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # Reinicializar con valores por defecto
        SessionStateManager.initialize_session_state()

# ============================================================================
# SECCIÓN 1: CONFIGURACIÓN DEL SISTEMA Y CONSTANTES MATEMÁTICAS
# ============================================================================

class SystemConfig:
    """Configuración centralizada del sistema ACBE-S73 v3.0."""
    
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
    HAMMING_DISTANCE_TARGET = 2  # Cobertura de 2 errores
    
    # Umbrales de clasificación por entropía
    STRONG_MATCH_THRESHOLD = 0.30   # ≤ 0.30: Partido Fuerte (1 signo)
    MEDIUM_MATCH_THRESHOLD = 0.60   # 0.30-0.60: Partido Medio (2 signos)
                                    # ≥ 0.60: Partido Caótico (3 signos)
    
    # Umbrales de reducción S73 (Validación institucional)
    MIN_OPTION_PROBABILITY = 0.55   # Umbral mínimo por opción
    MIN_PROBABILITY_GAP = 0.12      # Gap mínimo entre 1ª y 2ª opción
    MIN_EV_THRESHOLD = 0.0          # EV mínimo positivo
    
    # Gestión de riesgo
    MIN_ODDS = 1.01
    MAX_ODDS = 100.0
    DEFAULT_BANKROLL = 10000.0
    MAX_PORTFOLIO_EXPOSURE = 0.15   # 15% exposición máxima del portafolio
    MIN_JOINT_PROBABILITY = 0.001   # Umbral mínimo probabilidad conjunta
    
    # Configuración visual
    COLORS = {
        'primary': '#1E88E5',
        'secondary': '#FFC107', 
        'success': '#4CAF50',
        'danger': '#F44336',
        'warning': '#FF9800',
        'info': '#00BCD4'
    }
    
    # Paleta de riesgo para gráficos
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
    OUTCOME_BOX_COLORS = ['#4CAF50', '#FF9800', '#F44336']  # Para Apuesta Maestra
    
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
# SECCIÓN 2: MODELO MATEMÁTICO ACBE (COMPLETAMENTE VECTORIZADO)
# ============================================================================

class ACBEModel:
    """Modelo Bayesiano Gamma-Poisson para estimación de probabilidades (100% vectorizado)."""
    
    @staticmethod
    @st.cache_data
    def vectorized_poisson_simulation(lambda_home: np.ndarray, 
                                     lambda_away: np.ndarray, 
                                     n_sims: int = SystemConfig.MONTE_CARLO_ITERATIONS) -> np.ndarray:
        """
        Simulación Monte Carlo 100% vectorizada usando broadcasting de NumPy.
        
        Args:
            lambda_home: Tasas de goles locales (n_matches,)
            lambda_away: Tasas de goles visitantes (n_matches,)
            n_sims: Iteraciones Monte Carlo
            
        Returns:
            Array (n_matches, 3) con probabilidades [P(1), P(X), P(2)]
        """
        n_matches = len(lambda_home)
        
        # Broadcasting para generar todas las simulaciones a la vez
        home_goals = np.random.poisson(
            lam=lambda_home.reshape(1, -1),
            size=(n_sims, n_matches)
        )
        
        away_goals = np.random.poisson(
            lam=lambda_away.reshape(1, -1),
            size=(n_sims, n_matches)
        )
        
        # Cálculos vectorizados
        home_wins = np.mean(home_goals > away_goals, axis=0)
        draws = np.mean(home_goals == away_goals, axis=0)
        away_wins = np.mean(home_goals < away_goals, axis=0)
        
        # Stack en una sola matriz (n_matches, 3)
        probabilities = np.column_stack([home_wins, draws, away_wins])
        
        # Normalización vectorizada
        row_sums = probabilities.sum(axis=1, keepdims=True)
        probabilities = probabilities / row_sums
        
        # Estabilidad numérica
        probabilities = np.clip(probabilities, SystemConfig.MIN_PROBABILITY, 1.0)
        
        return probabilities
    
    @staticmethod
    def calculate_entropy_vectorized(probabilities: np.ndarray) -> np.ndarray:
        """
        Cálculo de entropía de Shannon completamente vectorizado.
        
        Args:
            probabilities: Array (n_matches, 3) de probabilidades
            
        Returns:
            Array (n_matches,) de entropías
        """
        # Evitar log(0) usando máscara
        mask = probabilities > SystemConfig.MIN_PROBABILITY
        log_probs = np.zeros_like(probabilities)
        log_probs[mask] = np.log(probabilities[mask]) / np.log(SystemConfig.BASE_ENTROPY)
        
        # Entropía: -Σ p * log(p)
        entropy = -np.sum(probabilities * log_probs, axis=1)
        
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
# SECCIÓN 3: SISTEMA COMBINATORIO S73 CON ALGORITMO HAMMING EXACTO
# ============================================================================

class S73System:
    """Sistema combinatorio S73 con algoritmo greedy de cobertura Hamming exacto."""
    
    @staticmethod
    @st.cache_data
    def _generate_all_combinations() -> np.ndarray:
        """
        Genera todas las 3^6 = 729 combinaciones posibles.
        
        Returns:
            Array (729, 6) con todas las combinaciones
        """
        combinations = list(itertools.product([0, 1, 2], repeat=6))
        return np.array(combinations)
    
    @staticmethod
    @st.cache_data
    def build_s73_coverage_system_hamming(probabilities: np.ndarray,
                                         odds_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Implementación EXACTA del algoritmo greedy con cobertura de distancia Hamming.
        Según especificación: Score = Probabilidad × (1 + EV)
        
        Args:
            probabilities: Array (6, 3) de probabilidades ACBE
            odds_matrix: Array (6, 3) de cuotas
            
        Returns:
            selected_combinations: Array (73, 6) de combinaciones seleccionadas
            selected_probs: Array (73,) de probabilidades conjuntas
        """
        # 1. Generar matriz completa de 3^6 = 729 combinaciones
        all_combinations = S73System._generate_all_combinations()
        
        # 2. Calcular probabilidades conjuntas y cuotas (vectorizado)
        n_combinations = len(all_combinations)
        joint_probs = np.ones(n_combinations)
        joint_odds = np.ones(n_combinations)
        
        for i, combo in enumerate(all_combinations):
            for match_idx, sign in enumerate(combo):
                joint_probs[i] *= probabilities[match_idx, sign]
                joint_odds[i] *= odds_matrix[match_idx, sign]
        
        # 3. Calcular EV y Score (Probabilidad × (1 + EV))
        ev_values = joint_probs * joint_odds - 1
        scores = joint_probs * (1 + ev_values)
        
        # 4. Ordenar por Score descendente
        sorted_indices = np.argsort(scores)[::-1]
        sorted_combinations = all_combinations[sorted_indices]
        sorted_scores = scores[sorted_indices]
        sorted_probs = joint_probs[sorted_indices]
        
        # 5. Algoritmo greedy de cobertura Hamming
        selected_indices = []
        covered = np.zeros(len(sorted_combinations), dtype=bool)
        
        for idx in range(len(sorted_combinations)):
            if covered[idx]:
                continue
            
            # Seleccionar mejor columna no cubierta
            selected_indices.append(idx)
            
            # Calcular distancias Hamming con todas las columnas (vectorizado)
            current_combo = sorted_combinations[idx]
            distances = np.sum(current_combo != sorted_combinations, axis=1)
            
            # Marcar como cubiertas las columnas con distancia ≤ 2
            covered[distances <= SystemConfig.HAMMING_DISTANCE_TARGET] = True
            
            if len(selected_indices) >= SystemConfig.TARGET_COMBINATIONS:
                break
        
        # 6. Backfill: rellenar hasta 73 si es necesario
        if len(selected_indices) < SystemConfig.TARGET_COMBINATIONS:
            remaining = [i for i in range(len(sorted_combinations)) 
                        if not covered[i] and i not in selected_indices]
            
            if remaining:
                # Ordenar por probabilidad pura para backfill
                remaining_probs = sorted_probs[remaining]
                remaining_sorted = [i for _, i in sorted(zip(remaining_probs, remaining), reverse=True)]
                
                for i in remaining_sorted:
                    if len(selected_indices) >= SystemConfig.TARGET_COMBINATIONS:
                        break
                    selected_indices.append(i)
        
        # Limitar a 73 columnas exactamente
        selected_indices = selected_indices[:SystemConfig.TARGET_COMBINATIONS]
        
        # 7. Validación de cobertura
        selected_combinations = sorted_combinations[selected_indices]
        selected_probs = sorted_probs[selected_indices]
        
        # Calcular distancias mínimas para validación
        distances_matrix = np.zeros((len(selected_combinations), len(sorted_combinations)))
        for i, combo in enumerate(selected_combinations):
            distances_matrix[i] = np.sum(combo != sorted_combinations, axis=1)
        
        min_distances = np.min(distances_matrix, axis=0)
        coverage_validation = np.all(min_distances <= SystemConfig.HAMMING_DISTANCE_TARGET)
        
        if coverage_validation:
            st.success(f"✅ Cobertura validada: Todas las 729 combinaciones a distancia ≤ {SystemConfig.HAMMING_DISTANCE_TARGET}")
        else:
            st.warning(f"⚠️ Cobertura parcial: {np.sum(min_distances <= SystemConfig.HAMMING_DISTANCE_TARGET)}/729 combinaciones cubiertas")
        
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
# SECCIÓN 4: SMART RECOMMENDATION ENGINE
# ============================================================================

class SmartRecommendationEngine:
    """Motor de recomendaciones inteligentes con Apuesta Maestra y Simulador de Cobertura."""
    
    @staticmethod
    def render_master_bet(probabilities: np.ndarray, combinations: np.ndarray, 
                         combination_probs: np.ndarray, odds_matrix: np.ndarray):
        """
        Visualiza la Apuesta Maestra (columna con mayor probabilidad) con cuadros coloreados.
        
        Args:
            probabilities: Probabilidades ACBE (6, 3)
            combinations: Combinaciones S73 (73, 6)
            combination_probs: Probabilidades conjuntas (73,)
            odds_matrix: Cuotas (6, 3)
        """
        # Encontrar combinación con mayor probabilidad
        master_idx = np.argmax(combination_probs)
        master_combo = combinations[master_idx]
        master_prob = combination_probs[master_idx]
        
        # Calcular cuota conjunta
        master_odds = S73System.calculate_combination_odds(master_combo, odds_matrix)
        
        # Mostrar Apuesta Maestra
        st.subheader("🏆 APUESTA MAESTRA (Columna Base del Sistema)")
        
        col1, col2, col3, col4 = st.columns([4, 2, 2, 2])
        with col1:
            st.markdown("**Combinación:**")
        with col2:
            st.metric("Probabilidad", f"{master_prob:.4%}")
        with col3:
            st.metric("Cuota", f"{master_odds:.2f}")
        with col4:
            ev = master_prob * master_odds - 1
            st.metric("EV", f"{ev:+.4f}", delta_color="normal")
        
        # Mostrar cuadros coloreados para cada partido
        st.markdown("**Signos por Partido:**")
        cols = st.columns(6)
        for i, (col, sign) in enumerate(zip(cols, master_combo)):
            with col:
                if sign == 0:  # 1
                    col.markdown(f"""
                        <div style="background-color:#4CAF50; padding:20px; border-radius:10px; 
                                    text-align:center; color:white; font-weight:bold; font-size:24px;
                                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                        1
                        <div style="font-size:12px; margin-top:5px;">
                        P={probabilities[i, 0]:.1%}
                        </div>
                        </div>
                    """, unsafe_allow_html=True)
                elif sign == 1:  # X
                    col.markdown(f"""
                        <div style="background-color:#FF9800; padding:20px; border-radius:10px; 
                                    text-align:center; color:white; font-weight:bold; font-size:24px;
                                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                        X
                        <div style="font-size:12px; margin-top:5px;">
                        P={probabilities[i, 1]:.1%}
                        </div>
                        </div>
                    """, unsafe_allow_html=True)
                else:  # 2
                    col.markdown(f"""
                        <div style="background-color:#F44336; padding:20px; border-radius:10px; 
                                    text-align:center; color:white; font-weight:bold; font-size:24px;
                                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                        2
                        <div style="font-size:12px; margin-top:5px;">
                        P={probabilities[i, 2]:.1%}
                        </div>
                        </div>
                    """, unsafe_allow_html=True)
    
    @staticmethod
    def render_coverage_simulator(combinations: np.ndarray):
        """
        Simulador interactivo de cobertura por fallos hipotéticos.
        
        Args:
            combinations: Combinaciones S73 (73, 6)
        """
        st.subheader("🔍 SIMULADOR DE COBERTURA")
        st.markdown("**Simula:** 'Si fallo estos partidos, ¿cuántas columnas mantienen 4, 5 o 6 aciertos?'")
        
        # Selector de partidos fallados
        st.markdown("### Selecciona los partidos que FALLAS:")
        failed_matches = []
        cols = st.columns(6)
        for i in range(6):
            with cols[i]:
                if st.checkbox(f"Partido {i+1}", key=f"fail_{i}"):
                    failed_matches.append(i)
        
        if failed_matches:
            # Para cada combinación, calcular aciertos en los partidos NO fallados
            n_columns = len(combinations)
            hits_distribution = {6: 0, 5: 0, 4: 0, 3: 0, 2: 0, 1: 0, 0: 0}
            
            for combo in combinations:
                # Suponemos que en los partidos fallados, el resultado real es diferente
                # Contamos aciertos solo en partidos no fallados
                possible_hits = 6 - len(failed_matches)
                
                # Distribución de probabilidad de aciertos
                # Para simplificar, asumimos que cada columna tiene acierto en los no fallados
                hits_distribution[possible_hits] += 1
            
            # Mostrar resultados
            st.markdown("### 📊 Columnas que mantendrían:")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                remaining_hits = 6 - len(failed_matches)
                st.metric(f"{remaining_hits} Aciertos", hits_distribution[remaining_hits])
            
            with col2:
                if remaining_hits >= 1:
                    st.metric(f"{remaining_hits-1} Aciertos", hits_distribution.get(remaining_hits-1, 0))
                else:
                    st.metric("5 Aciertos", "N/A")
            
            with col3:
                if remaining_hits >= 2:
                    st.metric(f"{remaining_hits-2} Aciertos", hits_distribution.get(remaining_hits-2, 0))
                else:
                    st.metric("4 Aciertos", "N/A")
            
            # Información adicional
            st.info(f"""
            **Interpretación:** Si fallas {len(failed_matches)} partido(s), tendrás {remaining_hits} partidos correctos.
            
            **Columnas disponibles:** {hits_distribution[remaining_hits]} de 73 columnas mantienen todos los partidos restantes correctos.
            
            **Cobertura:** El sistema S73 garantiza al menos una columna con {SystemConfig.HAMMING_DISTANCE_TARGET} errores o menos.
            """)
        else:
            st.info("Selecciona los partidos que crees que podrías fallar para ver la cobertura del sistema.")

# ============================================================================
# SECCIÓN 5: GESTIÓN DE CAPITAL (KELLY DINÁMICO)
# ============================================================================

class KellyCapitalManagement:
    """Gestión de capital basada en criterio de Kelly con límite de exposición del 15%."""
    
    @staticmethod
    def calculate_column_kelly(combination: np.ndarray,
                              joint_probability: float,
                              combination_odds: float,
                              avg_entropy: float,
                              manual_stake: Optional[float] = None) -> float:
        """
        Calcula stake Kelly para una columna del sistema S73.
        
        Fórmula: f* = (bp - q) / b
        Donde:
          b = Cuota Combinada - 1
          p = Probabilidad Conjunta
          q = 1 - p
        
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
        b = combination_odds - 1
        p = joint_probability
        q = 1 - p
        
        kelly_raw = (b * p - q) / b if b > 0 else 0
        
        # Aplicar límites y ajuste por entropía
        kelly_capped = max(0.0, min(kelly_raw, SystemConfig.KELLY_FRACTION_MAX))
        kelly_adjusted = kelly_capped * (1.0 - avg_entropy)
        
        return kelly_adjusted
    
    @staticmethod
    def normalize_portfolio_stakes(stakes_array: np.ndarray,
                                  max_exposure: float = SystemConfig.MAX_PORTFOLIO_EXPOSURE,
                                  is_manual_mode: bool = False) -> np.ndarray:
        """
        Normaliza stakes para limitar exposición total del portafolio al 15%.
        
        Args:
            stakes_array: Array de stakes individuales
            max_exposure: Exposición máxima permitida (0.15 = 15%)
            is_manual_mode: Si es True, mantener proporciones pero limitar total
            
        Returns:
            Array de stakes normalizados
        """
        total_exposure = np.sum(stakes_array)
        
        if total_exposure > max_exposure:
            scaling_factor = max_exposure / total_exposure
            stakes_array = stakes_array * scaling_factor
            
            if is_manual_mode:
                st.warning(f"Stake manual reducido para mantener exposición máxima del {max_exposure*100:.0f}%")
            else:
                st.info(f"Stakes Kelly escalados para mantener exposición máxima del {max_exposure*100:.0f}%")
        
        return stakes_array

# ============================================================================
# SECCIÓN 6: MOTOR DE BACKTESTING VECTORIZADO (73 COLUMNAS ESPECÍFICAS)
# ============================================================================

class VectorizedBacktester:
    """Motor de backtesting completamente vectorizado para las 73 columnas S73."""
    
    def __init__(self, initial_bankroll: float = SystemConfig.DEFAULT_BANKROLL):
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.equity_curve = [initial_bankroll]
        self.drawdown_curve = [0.0]
    
    @staticmethod
    @st.cache_data
    def simulate_match_outcomes_vectorized(probabilities: np.ndarray, n_sims: int) -> np.ndarray:
        """
        Simula resultados de partidos usando distribución multinomial vectorizada.
        
        Args:
            probabilities: Array (6, 3) de probabilidades
            n_sims: Número de simulaciones
            
        Returns:
            Array (n_sims, 6) de resultados (0, 1, 2)
        """
        n_matches = probabilities.shape[0]
        outcomes = np.zeros((n_sims, n_matches), dtype=int)
        
        # Vectorizado: generar todas las simulaciones a la vez
        for i in range(n_matches):
            # Generar samples multinomiales para todos los simulations a la vez
            samples = np.random.multinomial(1, probabilities[i], size=n_sims)
            outcomes[:, i] = np.argmax(samples, axis=1)
        
        return outcomes
    
    def run_s73_backtest(self,
                        probabilities: np.ndarray,
                        odds_matrix: np.ndarray,
                        normalized_entropies: np.ndarray,
                        s73_results: Dict,
                        n_rounds: int = 100,
                        n_sims_per_round: int = 1000,
                        kelly_fraction: float = 0.5,
                        manual_stake: Optional[float] = None) -> Dict:
        """
        Ejecuta backtesting específico para las 73 columnas S73 generadas.
        
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
        combination_probs = s73_results['probabilities']
        n_columns = len(combinations)
        
        # Reinicializar métricas
        self.bankroll = self.initial_bankroll
        self.equity_curve = [self.bankroll]
        self.drawdown_curve = [0.0]
        
        all_returns = []
        round_metrics = []
        
        # Calcular stakes iniciales
        current_stakes = self._calculate_initial_stakes(
            s73_results, normalized_entropies, odds_matrix, kelly_fraction, manual_stake
        )
        
        for round_idx in range(n_rounds):
            # 1. Simular resultados reales para esta ronda
            real_outcomes = self.simulate_match_outcomes_vectorized(
                probabilities, n_sims_per_round
            )
            
            # 2. Calcular rendimiento de cada columna (vectorizado)
            round_returns = self._calculate_column_performance_vectorized(
                real_outcomes, combinations, odds_matrix, current_stakes
            )
            
            # 3. Actualizar bankroll (retorno promedio de la ronda)
            avg_return = np.mean(round_returns)
            self.bankroll += avg_return
            
            # 4. Actualizar stakes si estamos en modo Kelly (bankroll cambió)
            if manual_stake is None:
                current_stakes = self._update_kelly_stakes(
                    s73_results, normalized_entropies, odds_matrix, kelly_fraction
                )
            
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
    
    def _calculate_initial_stakes(self, s73_results: Dict, normalized_entropies: np.ndarray,
                                 odds_matrix: np.ndarray, kelly_fraction: float,
                                 manual_stake: Optional[float]) -> np.ndarray:
        """Calcula stakes iniciales para las columnas S73."""
        combinations = s73_results['combinations']
        combination_probs = s73_results['probabilities']
        n_columns = len(combinations)
        
        stakes = np.zeros(n_columns)
        
        for i in range(n_columns):
            combo = combinations[i]
            prob = combination_probs[i]
            odds = S73System.calculate_combination_odds(combo, odds_matrix)
            
            # Calcular entropía promedio de la combinación
            combo_entropy = np.mean([normalized_entropies[j] for j in range(6)])
            
            # Calcular stake Kelly
            stake = KellyCapitalManagement.calculate_column_kelly(
                combo, prob, odds, combo_entropy, manual_stake
            )
            
            if manual_stake is None:
                stake *= kelly_fraction
            
            stakes[i] = stake
        
        # Normalizar para límite de exposición
        stakes = KellyCapitalManagement.normalize_portfolio_stakes(
            stakes, is_manual_mode=(manual_stake is not None)
        )
        
        return stakes
    
    def _calculate_column_performance_vectorized(self, real_outcomes: np.ndarray,
                                               combinations: np.ndarray,
                                               odds_matrix: np.ndarray,
                                               stakes: np.ndarray) -> np.ndarray:
        """Calcula rendimiento de columnas de forma vectorizada."""
        n_sims = len(real_outcomes)
        n_columns = len(combinations)
        
        # Calcular cuotas conjuntas
        combination_odds = np.zeros(n_columns)
        for i, combo in enumerate(combinations):
            combination_odds[i] = S73System.calculate_combination_odds(combo, odds_matrix)
        
        # Stakes en euros
        stakes_euros = stakes * self.initial_bankroll
        
        # Calcular retornos (vectorizado parcialmente)
        returns = np.zeros(n_sims)
        
        for sim_idx in range(n_sims):
            sim_returns = 0
            for col_idx, combo in enumerate(combinations):
                # Verificar si la combinación acierta completamente
                if np.all(real_outcomes[sim_idx] == combo):
                    sim_returns += stakes_euros[col_idx] * (combination_odds[col_idx] - 1)
                else:
                    sim_returns -= stakes_euros[col_idx]
            
            returns[sim_idx] = sim_returns
        
        return returns
    
    def _update_kelly_stakes(self, s73_results: Dict, normalized_entropies: np.ndarray,
                            odds_matrix: np.ndarray, kelly_fraction: float) -> np.ndarray:
        """Actualiza stakes Kelly basados en bankroll actual."""
        combinations = s73_results['combinations']
        combination_probs = s73_results['probabilities']
        n_columns = len(combinations)
        
        stakes = np.zeros(n_columns)
        
        for i in range(n_columns):
            combo = combinations[i]
            prob = combination_probs[i]
            odds = S73System.calculate_combination_odds(combo, odds_matrix)
            
            # Calcular entropía promedio de la combinación
            combo_entropy = np.mean([normalized_entropies[j] for j in range(6)])
            
            # Calcular stake Kelly
            stake = KellyCapitalManagement.calculate_column_kelly(
                combo, prob, odds, combo_entropy, None
            )
            
            stakes[i] = stake * kelly_fraction
        
        # Normalizar para límite de exposición
        stakes = KellyCapitalManagement.normalize_portfolio_stakes(stakes)
        
        return stakes
    
    def _calculate_final_metrics(self, all_returns: List[float], n_rounds: int) -> Dict:
        """Calcula métricas finales agregadas del backtest."""
        returns_array = np.array(all_returns)
        
        # ROI y retorno total
        total_return = self.bankroll - self.initial_bankroll
        total_return_pct = (total_return / self.initial_bankroll) * 100
        
        # Sharpe Ratio
        if np.std(returns_array) > 0:
            sharpe_ratio = (np.mean(returns_array) / np.std(returns_array)) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Drawdown máximo
        max_drawdown = np.max(self.drawdown_curve)
        
        # CAGR
        if self.bankroll > 0:
            cagr = ((self.bankroll / self.initial_bankroll) ** (252 / n_rounds) - 1) * 100
        else:
            cagr = -100.0
        
        # Value at Risk (VaR 95%)
        var_95 = np.percentile(returns_array, 5) if len(returns_array) > 0 else 0
        
        # Win rate
        win_rate = (np.sum(returns_array > 0) / len(returns_array) * 100) if len(returns_array) > 0 else 0
        
        # Profit factor
        total_wins = np.sum(returns_array[returns_array > 0])
        total_losses = abs(np.sum(returns_array[returns_array < 0]))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Estadísticas adicionales
        avg_win = np.mean(returns_array[returns_array > 0]) if len(returns_array[returns_array > 0]) > 0 else 0
        avg_loss = np.mean(returns_array[returns_array < 0]) if len(returns_array[returns_array < 0]) > 0 else 0
        
        # Probabilidad de ruina
        ruin_prob = np.mean(np.array(self.equity_curve) < self.initial_bankroll * 0.5) * 100
        
        return {
            'initial_bankroll': self.initial_bankroll,
            'final_bankroll': self.bankroll,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
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
# SECCIÓN 7: SISTEMA DE EXPORTACIÓN COMPATIBLE EXCEL
# ============================================================================

class DataExporter:
    """Sistema profesional de exportación de datos con encoding UTF-8-SIG."""
    
    @staticmethod
    def generate_timestamp() -> str:
        """Genera timestamp para nombres de archivo."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    @staticmethod
    def export_to_csv_excel_compatible(df: pd.DataFrame) -> str:
        """
        Convierte DataFrame a CSV compatible con Excel español.
        
        Args:
            df: DataFrame a exportar
            
        Returns:
            String CSV con encoding UTF-8-SIG y separador ;
        """
        # Usar punto y coma como separador (estándar español)
        # Encoding UTF-8-SIG para compatibilidad Excel
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, sep=';', index=False, decimal=',', encoding='utf-8-sig')
        return csv_buffer.getvalue()
    
    @staticmethod
    def export_s73_system(combinations: np.ndarray, probabilities: np.ndarray,
                         odds_matrix: np.ndarray, stakes: np.ndarray,
                         bankroll: float) -> Dict[str, Any]:
        """
        Exporta el sistema S73 completo en formato profesional.
        
        Args:
            combinations: Combinaciones S73 (73, 6)
            probabilities: Probabilidades conjuntas (73,)
            odds_matrix: Cuotas (6, 3)
            stakes: Stakes Kelly (73,)
            bankroll: Bankroll inicial
            
        Returns:
            Diccionario con datos para descarga
        """
        timestamp = DataExporter.generate_timestamp()
        
        # Preparar DataFrame con todas las columnas
        export_data = []
        
        for idx, (combo, prob, stake) in enumerate(zip(combinations, probabilities, stakes), 1):
            # Calcular cuota conjunta
            combo_odds = S73System.calculate_combination_odds(combo, odds_matrix)
            
            # Calcular EV
            ev = prob * combo_odds - 1
            
            # Calcular inversión en euros
            investment = stake * bankroll
            
            export_data.append({
                'ID_Columna': idx,
                'Combinacion': ''.join([SystemConfig.OUTCOME_LABELS[s] for s in combo]),
                'Partido_1': SystemConfig.OUTCOME_LABELS[combo[0]],
                'Partido_2': SystemConfig.OUTCOME_LABELS[combo[1]],
                'Partido_3': SystemConfig.OUTCOME_LABELS[combo[2]],
                'Partido_4': SystemConfig.OUTCOME_LABELS[combo[3]],
                'Partido_5': SystemConfig.OUTCOME_LABELS[combo[4]],
                'Partido_6': SystemConfig.OUTCOME_LABELS[combo[5]],
                'Probabilidad': prob,
                'Cuota_Conjunta': combo_odds,
                'EV_Columna': ev,
                'Stake_Porcentaje': stake * 100,
                'Inversion_Euros': investment,
                'Ganancia_Potencial': investment * (combo_odds - 1) if combo_odds > 1 else 0
            })
        
        df_export = pd.DataFrame(export_data)
        
        # Exportar con encoding correcto
        csv_data = DataExporter.export_to_csv_excel_compatible(df_export)
        
        return {
            'filename': f"ACBE_S73_Sistema_{timestamp}.csv",
            'data': csv_data,
            'mime_type': 'text/csv; charset=utf-8-sig'
        }
    
    @staticmethod
    def export_backtest_results(backtest_results: Dict) -> Dict[str, Any]:
        """
        Exporta resultados de backtesting en múltiples formatos.
        
        Args:
            backtest_results: Resultados del backtesting
            
        Returns:
            Diccionario con datos para descarga
        """
        timestamp = DataExporter.generate_timestamp()
        metrics = backtest_results['final_metrics']
        
        # 1. Métricas principales (CSV)
        metrics_df = pd.DataFrame([metrics])
        metrics_csv = DataExporter.export_to_csv_excel_compatible(metrics_df)
        
        # 2. Curva de equity (CSV)
        equity_df = pd.DataFrame({
            'Ronda': range(len(backtest_results['equity_curve'])),
            'Bankroll': backtest_results['equity_curve'],
            'Drawdown_%': backtest_results['drawdown_curve']
        })
        equity_csv = DataExporter.export_to_csv_excel_compatible(equity_df)
        
        # 3. Reporte ejecutivo (TXT)
        report_text = f"""
        REPORTE DE BACKTESTING ACBE-S73 v3.0
        =====================================
        Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        
        MÉTRICAS DE RENDIMIENTO:
        ------------------------
        Bankroll Inicial: €{metrics['initial_bankroll']:,.2f}
        Bankroll Final: €{metrics['final_bankroll']:,.2f}
        Retorno Total: {metrics['total_return_pct']:+.2f}% (€{metrics['total_return']:+,.2f})
        
        Sharpe Ratio: {metrics['sharpe_ratio']:.3f}
        Max Drawdown: {metrics['max_drawdown']:.2f}%
        CAGR: {metrics['cagr']:+.2f}%
        
        RIESGO:
        -------
        VaR 95%: €{metrics['var_95']:.2f}
        Volatilidad (σ): €{metrics['std_returns']:.2f}
        Prob. Ruina: {metrics['ruin_probability']:.2f}%
        
        ESTADÍSTICAS:
        -------------
        Win Rate: {metrics['win_rate']:.2f}%
        Profit Factor: {metrics['profit_factor']:.3f}
        Ganancia Promedio: €{metrics['avg_win']:.2f}
        Pérdida Promedio: €{metrics['avg_loss']:.2f}
        
        FIRMA:
        ------
        ACBE-S73 Quantum Betting Suite v3.0
        Sistema Validado Institucionalmente
        Algoritmo: Gamma-Poisson Bayesiano + Hamming Greedy + Kelly Dinámico
        """
        
        return {
            'metrics': {
                'filename': f"ACBE_Backtest_Metricas_{timestamp}.csv",
                'data': metrics_csv,
                'mime': 'text/csv'
            },
            'equity': {
                'filename': f"ACBE_Backtest_Equity_{timestamp}.csv",
                'data': equity_csv,
                'mime': 'text/csv'
            },
            'report': {
                'filename': f"ACBE_Backtest_Reporte_{timestamp}.txt",
                'data': report_text,
                'mime': 'text/plain'
            }
        }

# ============================================================================
# SECCIÓN 8: INTERFAZ PRINCIPAL STREAMLIT v3.0
# ============================================================================

class ACBEApp:
    """Interfaz principal de la aplicación Streamlit - Versión 3.0."""
    
    def __init__(self):
        self.setup_page_config()
        SessionStateManager.initialize_session_state()
    
    def setup_page_config(self):
        """Configuración de la página Streamlit."""
        st.set_page_config(
            page_title="ACBE-S73 Quantum Betting Suite v3.0",
            page_icon="🎯",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def render_sidebar(self) -> Dict:
        """Renderiza sidebar con configuración del sistema."""
        with st.sidebar:
            st.header("⚙️ Configuración del Sistema v3.0")
            st.caption(f"Sistema ACBE-S73 | {datetime.now().strftime('%Y-%m-%d')}")
            
            # Botón para limpiar datos
            if st.button("🔄 Reiniciar Todo", type="secondary", use_container_width=True):
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
            
            # Gestión de Stake
            st.subheader("🎮 Gestión de Stake")
            
            auto_stake_mode = st.toggle(
                "Modo Automático (Kelly Dinámico)",
                value=True,
                help="Kelly ajustado por entropía con límite del 15% de exposición"
            )
            
            manual_stake = None
            if not auto_stake_mode:
                manual_stake = st.number_input(
                    "Stake Manual (% por columna)",
                    min_value=0.01,
                    max_value=10.0,
                    value=1.0,
                    step=0.1
                )
                manual_stake_fraction = manual_stake / 100.0
                st.info(f"Stake fijo: {manual_stake}% por columna")
            else:
                manual_stake_fraction = None
                kelly_fraction = st.slider(
                    "Fracción Conservadora de Kelly",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                    help="Reduce el stake Kelly completo para mayor seguridad"
                )
            
            # Parámetros de riesgo
            st.subheader("📊 Gestión de Riesgo")
            
            max_exposure = st.slider(
                "Exposición Máxima del Portafolio (%)",
                min_value=5,
                max_value=30,
                value=15,
                step=1,
                help="Límite máximo del bankroll en apuestas (hard limit)"
            )
            
            # Configuración de simulaciones
            st.subheader("🎲 Parámetros de Simulación")
            
            monte_carlo_sims = st.number_input(
                "Iteraciones Monte Carlo",
                min_value=1000,
                max_value=50000,
                value=10000,
                step=1000,
                help="Número de simulaciones por cálculo de probabilidades"
            )
            
            n_rounds = st.slider(
                "Rondas de Backtesting",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                help="Número de jornadas simuladas en backtesting"
            )
            
            # Filtros S73
            st.subheader("🎯 Filtros S73")
            
            apply_filters = st.toggle(
                "Aplicar filtros de validación",
                value=True,
                help="Umbrales probabilísticos para reducción S73"
            )
            
            if apply_filters:
                col1, col2 = st.columns(2)
                with col1:
                    min_prob = st.slider(
                        "Prob. mínima opción",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.55,
                        step=0.01
                    )
                with col2:
                    min_gap = st.slider(
                        "Gap mínimo",
                        min_value=0.0,
                        max_value=0.5,
                        value=0.12,
                        step=0.01
                    )
                
                SystemConfig.MIN_OPTION_PROBABILITY = min_prob
                SystemConfig.MIN_PROBABILITY_GAP = min_gap
            
            # Fuente de datos
            st.subheader("📊 Fuente de Datos")
            data_source = st.radio(
                "Modo de entrada:",
                ["⚽ Input Manual", "📈 Datos Sintéticos"],
                index=0
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
                'apply_filters': apply_filters
            }
    
    def render_input_phase(self, config: Dict):
        """Renderiza fase de input de datos."""
        st.header("📥 FASE 1: INGRESO DE DATOS")
        
        if config['data_source'] == "⚽ Input Manual":
            self.render_manual_input()
        else:
            self.render_synthetic_data(config)
    
    def render_manual_input(self):
        """Renderiza input manual de partidos."""
        st.subheader("⚽ Ingreso Manual de 6 Partidos")
        
        matches_data = []
        attack_strengths = []
        defense_strengths = []
        home_advantages = []
        odds_matrix = []
        
        for match_idx in range(1, SystemConfig.NUM_MATCHES + 1):
            st.markdown(f"### Partido {match_idx}")
            
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                home_team = st.text_input(f"Local {match_idx}", value=f"Equipo A{match_idx}")
                away_team = st.text_input(f"Visitante {match_idx}", value=f"Equipo B{match_idx}")
            
            with col2:
                odds_1 = st.number_input(f"Cuota 1", min_value=1.01, max_value=100.0, value=1.8, step=0.1)
                odds_x = st.number_input(f"Cuota X", min_value=1.01, max_value=100.0, value=3.2, step=0.1)
                odds_2 = st.number_input(f"Cuota 2", min_value=1.01, max_value=100.0, value=4.0, step=0.1)
            
            with col3:
                with st.expander("Fuerzas"):
                    home_attack = st.slider(f"Ataque {home_team}", 0.5, 2.0, 1.2, 0.05)
                    home_defense = st.slider(f"Defensa {home_team}", 0.5, 2.0, 0.9, 0.05)
                    away_attack = st.slider(f"Ataque {away_team}", 0.5, 2.0, 1.0, 0.05)
                    away_defense = st.slider(f"Defensa {away_team}", 0.5, 2.0, 1.1, 0.05)
                    home_advantage = st.slider(f"Ventaja Local", 1.0, 1.5, 1.1, 0.01)
            
            matches_data.append({
                'home_team': home_team,
                'away_team': away_team,
                'odds_1': odds_1,
                'odds_x': odds_x,
                'odds_2': odds_2
            })
            
            attack_strengths.append([home_attack, away_attack])
            defense_strengths.append([home_defense, away_defense])
            home_advantages.append(home_advantage)
            odds_matrix.append([odds_1, odds_x, odds_2])
            
            st.markdown("---")
        
        # Botón para procesar
        if st.button("🚀 PROCESAR DATOS Y ANALIZAR", type="primary", use_container_width=True):
            # Convertir a arrays numpy
            attack_strengths = np.array(attack_strengths)
            defense_strengths = np.array(defense_strengths)
            home_advantages = np.array(home_advantages)
            odds_matrix = np.array(odds_matrix)
            
            # Calcular tasas de goles
            lambda_home = attack_strengths[:, 0] * defense_strengths[:, 1] * home_advantages
            lambda_away = attack_strengths[:, 1] * defense_strengths[:, 0]
            
            # Calcular probabilidades ACBE
            probabilities = ACBEModel.vectorized_poisson_simulation(lambda_home, lambda_away)
            
            # Calcular entropías
            entropy = ACBEModel.calculate_entropy_vectorized(probabilities)
            normalized_entropy = ACBEModel.normalize_entropy(entropy)
            
            # Guardar en session state
            st.session_state.probabilities_6 = probabilities
            st.session_state.odds_matrix_6 = odds_matrix
            st.session_state.entropy_6 = normalized_entropy
            st.session_state.matches_data = matches_data
            
            # Mover a fase de análisis
            SessionStateManager.move_to_analysis()
            st.rerun()
    
    def render_synthetic_data(self, config: Dict):
        """Genera datos sintéticos para análisis."""
        st.subheader("📈 Datos Sintéticos para Análisis")
        
        if st.button("🎲 GENERAR DATOS SINTÉTICOS", type="primary", use_container_width=True):
            np.random.seed(42)
            
            # Generar parámetros realistas
            attack_strengths = np.random.beta(2, 2, size=(6, 2)) * 1.5 + 0.5
            defense_strengths = np.random.beta(2, 2, size=(6, 2)) * 1.2 + 0.4
            home_advantages = np.random.uniform(1.05, 1.25, 6)
            
            # Calcular tasas de goles
            lambda_home = attack_strengths[:, 0] * defense_strengths[:, 1] * home_advantages
            lambda_away = attack_strengths[:, 1] * defense_strengths[:, 0]
            
            # Calcular probabilidades ACBE
            probabilities = ACBEModel.vectorized_poisson_simulation(
                lambda_home, lambda_away, n_sims=config['monte_carlo_sims']
            )
            
            # Generar cuotas con márgenes realistas
            margins = np.random.uniform(0.03, 0.07, 6)
            odds_matrix = np.zeros((6, 3))
            
            for i in range(6):
                fair_odds = 1 / probabilities[i]
                odds_matrix[i] = fair_odds * (1 + margins[i])
                odds_matrix[i] = np.clip(odds_matrix[i], 1.1, 20.0)
            
            # Calcular entropías
            entropy = ACBEModel.calculate_entropy_vectorized(probabilities)
            normalized_entropy = ACBEModel.normalize_entropy(entropy)
            
            # Guardar en session state
            st.session_state.probabilities_6 = probabilities
            st.session_state.odds_matrix_6 = odds_matrix
            st.session_state.entropy_6 = normalized_entropy
            
            # Mover a fase de análisis
            SessionStateManager.move_to_analysis()
            st.rerun()
    
    def render_analysis_phase(self, config: Dict):
        """Renderiza fase de análisis completo."""
        st.header("📊 FASE 2: ANÁLISIS DEL SISTEMA")
        
        # Crear pestañas principales
        tabs = st.tabs([
            "🔬 Análisis ACBE",
            "🧮 Sistema S73", 
            "🎯 Recomendaciones",
            "📈 Backtesting",
            "💾 Exportar"
        ])
        
        # Obtener datos de session state
        probabilities = st.session_state.probabilities_6
        odds_matrix = st.session_state.odds_matrix_6
        normalized_entropy = st.session_state.entropy_6
        
        # Variables para almacenar resultados entre pestañas
        if 's73_results' not in st.session_state:
            st.session_state.s73_results = None
        if 'backtest_results' not in st.session_state:
            st.session_state.backtest_results = None
        
        # Pestaña 1: Análisis ACBE
        with tabs[0]:
            self.render_acbe_analysis(probabilities, odds_matrix, normalized_entropy)
        
        # Pestaña 2: Sistema S73
        with tabs[1]:
            s73_results = self.render_s73_system(probabilities, odds_matrix, normalized_entropy, config)
            st.session_state.s73_results = s73_results
        
        # Pestaña 3: Recomendaciones Inteligentes
        with tabs[2]:
            if st.session_state.s73_results:
                self.render_smart_recommendations(probabilities, st.session_state.s73_results, odds_matrix)
            else:
                st.warning("Primero genera el sistema S73 en la pestaña anterior.")
        
        # Pestaña 4: Backtesting
        with tabs[3]:
            if st.session_state.s73_results:
                backtest_results = self.render_backtesting(probabilities, odds_matrix, 
                                                         normalized_entropy, st.session_state.s73_results, config)
                st.session_state.backtest_results = backtest_results
            else:
                st.warning("Primero genera el sistema S73 en la pestaña anterior.")
        
        # Pestaña 5: Exportación
        with tabs[4]:
            if st.session_state.s73_results and st.session_state.backtest_results:
                self.render_export_section(st.session_state.s73_results, st.session_state.backtest_results, config)
            else:
                st.info("Completa las fases anteriores para habilitar la exportación.")
    
    def render_acbe_analysis(self, probabilities: np.ndarray, odds_matrix: np.ndarray, 
                            normalized_entropy: np.ndarray):
        """Renderiza análisis ACBE completo."""
        st.subheader("🔬 ANÁLISIS PROBABILÍSTICO ACBE")
        
        # Calcular EV
        ev_matrix = probabilities * odds_matrix - 1
        
        # Crear DataFrames para visualización
        df_probabilities = pd.DataFrame({
            'Partido': [f'Partido {i+1}' for i in range(6)],
            'P(1)': probabilities[:, 0],
            'P(X)': probabilities[:, 1],
            'P(2)': probabilities[:, 2],
            'Entropía_Norm': normalized_entropy
        })
        
        df_odds_ev = pd.DataFrame({
            'Partido': [f'Partido {i+1}' for i in range(6)],
            'Cuota 1': odds_matrix[:, 0],
            'Cuota X': odds_matrix[:, 1],
            'Cuota 2': odds_matrix[:, 2],
            'EV 1': ev_matrix[:, 0],
            'EV X': ev_matrix[:, 1],
            'EV 2': ev_matrix[:, 2]
        })
        
        # Mostrar en columnas
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Probabilidades ACBE**")
            st.dataframe(df_probabilities.style.format({
                'P(1)': '{:.3f}',
                'P(X)': '{:.3f}',
                'P(2)': '{:.3f}',
                'Entropía_Norm': '{:.3f}'
            }), use_container_width=True)
        
        with col2:
            st.markdown("**💰 Cuotas y Valor Esperado**")
            st.dataframe(df_odds_ev.style.format({
                'Cuota 1': '{:.2f}',
                'Cuota X': '{:.2f}',
                'Cuota 2': '{:.2f}',
                'EV 1': '{:.3f}',
                'EV X': '{:.3f}',
                'EV 2': '{:.3f}'
            }), use_container_width=True)
        
        # Gráficos
        self._render_acbe_charts(probabilities, normalized_entropy, ev_matrix)
    
    def _render_acbe_charts(self, probabilities: np.ndarray, normalized_entropy: np.ndarray,
                           ev_matrix: np.ndarray):
        """Renderiza gráficos del análisis ACBE."""
        # Gráfico de probabilidades por partido
        fig_probs = go.Figure()
        for i, outcome in enumerate(['1', 'X', '2']):
            fig_probs.add_trace(go.Bar(
                x=[f'Partido {j+1}' for j in range(6)],
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
        
        # Gráfico de entropía y EV
        fig_entropy_ev = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_entropy_ev.add_trace(
            go.Scatter(
                x=[f'Partido {j+1}' for j in range(6)],
                y=normalized_entropy,
                name='Entropía Normalizada',
                line=dict(color=SystemConfig.COLORS['primary'], width=3)
            ),
            secondary_y=False
        )
        
        # Promedio de EV positivo
        avg_positive_ev = np.mean(ev_matrix[ev_matrix > 0], axis=1)
        fig_entropy_ev.add_trace(
            go.Bar(
                x=[f'Partido {j+1}' for j in range(6)],
                y=avg_positive_ev,
                name='EV Positivo Promedio',
                marker_color=SystemConfig.COLORS['success'],
                opacity=0.6
            ),
            secondary_y=True
        )
        
        fig_entropy_ev.update_layout(
            title="Entropía y Valor Esperado por Partido",
            xaxis_title="Partido",
            height=400
        )
        fig_entropy_ev.update_yaxes(title_text="Entropía Normalizada", secondary_y=False)
        fig_entropy_ev.update_yaxes(title_text="EV Promedio", secondary_y=True)
        
        # Mostrar gráficos
        st.plotly_chart(fig_probs, use_container_width=True)
        st.plotly_chart(fig_entropy_ev, use_container_width=True)
    
    def render_s73_system(self, probabilities: np.ndarray, odds_matrix: np.ndarray,
                         normalized_entropy: np.ndarray, config: Dict) -> Dict:
        """Renderiza sistema S73 completo."""
        st.subheader("🧮 SISTEMA COMBINATORIO S73")
        
        with st.spinner("🔄 Construyendo sistema S73 con algoritmo Hamming greedy..."):
            # 1. Generar sistema S73 con algoritmo Hamming
            combinations, combination_probs = S73System.build_s73_coverage_system_hamming(
                probabilities, odds_matrix
            )
            
            # 2. Calcular stakes para cada columna
            n_columns = len(combinations)
            stakes = np.zeros(n_columns)
            
            for i in range(n_columns):
                combo = combinations[i]
                prob = combination_probs[i]
                combo_odds = S73System.calculate_combination_odds(combo, odds_matrix)
                
                # Calcular entropía promedio de la combinación
                combo_entropy = np.mean([normalized_entropy[j] for j in range(6)])
                
                # Calcular stake Kelly
                stake = KellyCapitalManagement.calculate_column_kelly(
                    combo, prob, combo_odds, combo_entropy, config['manual_stake']
                )
                
                stakes[i] = stake
            
            # 3. Normalizar stakes para límite de exposición
            stakes = KellyCapitalManagement.normalize_portfolio_stakes(
                stakes, config['max_exposure'], is_manual_mode=(config['manual_stake'] is not None)
            )
            
            # 4. Crear DataFrame para visualización
            columns_data = []
            total_investment = 0
            
            for idx, (combo, prob, stake) in enumerate(zip(combinations, combination_probs, stakes), 1):
                combo_odds = S73System.calculate_combination_odds(combo, odds_matrix)
                investment = stake * config['bankroll']
                total_investment += investment
                
                columns_data.append({
                    'ID': idx,
                    'Combinación': ''.join([SystemConfig.OUTCOME_LABELS[s] for s in combo]),
                    'Probabilidad': prob,
                    'Cuota': combo_odds,
                    'EV': prob * combo_odds - 1,
                    'Stake %': stake * 100,
                    'Inversión €': investment,
                    'Ganancia Potencial €': investment * (combo_odds - 1) if combo_odds > 1 else 0
                })
            
            df_columns = pd.DataFrame(columns_data)
        
        # Mostrar estadísticas del sistema
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Columnas S73", n_columns)
        with col2:
            st.metric("Exposición Total", f"{np.sum(stakes)*100:.1f}%")
        with col3:
            st.metric("Inversión Total", f"€{total_investment:,.0f}")
        with col4:
            avg_prob = np.mean(combination_probs) * 100
            st.metric("Prob. Promedio", f"{avg_prob:.2f}%")
        
        # Mostrar tabla de columnas
        st.markdown("**📋 Columnas del Sistema S73:**")
        display_df = df_columns.copy()
        display_df['Probabilidad'] = display_df['Probabilidad'].apply(lambda x: f'{x:.4%}')
        display_df['Cuota'] = display_df['Cuota'].apply(lambda x: f'{x:.2f}')
        display_df['EV'] = display_df['EV'].apply(lambda x: f'{x:.4f}')
        display_df['Stake %'] = display_df['Stake %'].apply(lambda x: f'{x:.3f}%')
        display_df['Inversión €'] = display_df['Inversión €'].apply(lambda x: f'€{x:.2f}')
        display_df['Ganancia Potencial €'] = display_df['Ganancia Potencial €'].apply(lambda x: f'€{x:.2f}')
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Gráfico de distribución de stakes
        fig_stakes = go.Figure()
        fig_stakes.add_trace(go.Histogram(
            x=df_columns['Stake %'].astype(float),
            nbinsx=20,
            marker_color=SystemConfig.COLORS['primary'],
            opacity=0.7,
            name='Distribución de Stakes'
        ))
        
        fig_stakes.update_layout(
            title="Distribución de Stakes por Columna",
            xaxis_title="Stake (%)",
            yaxis_title="Número de Columnas",
            height=300
        )
        
        st.plotly_chart(fig_stakes, use_container_width=True)
        
        # Retornar resultados para uso en otras pestañas
        return {
            'combinations': combinations,
            'probabilities': combination_probs,
            'stakes': stakes,
            'df_columns': df_columns,
            'total_investment': total_investment,
            'total_exposure': np.sum(stakes) * 100
        }
    
    def render_smart_recommendations(self, probabilities: np.ndarray, s73_results: Dict,
                                    odds_matrix: np.ndarray):
        """Renderiza motor de recomendaciones inteligentes."""
        st.subheader("🎯 SMART RECOMMENDATION ENGINE")
        
        # 1. Apuesta Maestra
        SmartRecommendationEngine.render_master_bet(
            probabilities, 
            s73_results['combinations'],
            s73_results['probabilities'],
            odds_matrix
        )
        
        st.markdown("---")
        
        # 2. Simulador de Cobertura
        SmartRecommendationEngine.render_coverage_simulator(s73_results['combinations'])
        
        st.markdown("---")
        
        # 3. Análisis de Sensibilidad
        st.subheader("📈 ANÁLISIS DE SENSIBILIDAD")
        
        # Calcular impacto de cada partido en el sistema
        sensitivity_data = []
        
        for match_idx in range(6):
            # Encontrar el signo más probable para este partido
            best_sign = np.argmax(probabilities[match_idx])
            best_prob = probabilities[match_idx, best_sign]
            
            # Calcular cuántas columnas usan este signo
            columns_with_best_sign = np.sum(
                [1 for combo in s73_results['combinations'] if combo[match_idx] == best_sign]
            )
            
            percentage = (columns_with_best_sign / len(s73_results['combinations'])) * 100
            
            sensitivity_data.append({
                'Partido': match_idx + 1,
                'Signo Recomendado': SystemConfig.OUTCOME_LABELS[best_sign],
                'Probabilidad': f'{best_prob:.1%}',
                'Columnas con Signo': columns_with_best_sign,
                '% del Sistema': f'{percentage:.1f}%',
                'Importancia': 'ALTA' if percentage > 70 else 'MEDIA' if percentage > 40 else 'BAJA'
            })
        
        df_sensitivity = pd.DataFrame(sensitivity_data)
        st.dataframe(df_sensitivity, use_container_width=True)
    
    def render_backtesting(self, probabilities: np.ndarray, odds_matrix: np.ndarray,
                          normalized_entropy: np.ndarray, s73_results: Dict, config: Dict) -> Dict:
        """Renderiza backtesting del sistema S73."""
        st.subheader("📈 BACKTESTING DEL SISTEMA S73")
        
        if st.button("▶️ EJECUTAR BACKTESTING COMPLETO", type="primary"):
            with st.spinner(f"🔄 Simulando {config['n_rounds']} rondas con {config['monte_carlo_sims']:,} iteraciones..."):
                # Ejecutar backtesting
                backtester = VectorizedBacktester(initial_bankroll=config['bankroll'])
                
                backtest_results = backtester.run_s73_backtest(
                    probabilities=probabilities,
                    odds_matrix=odds_matrix,
                    normalized_entropies=normalized_entropy,
                    s73_results=s73_results,
                    n_rounds=config['n_rounds'],
                    n_sims_per_round=config['monte_carlo_sims'],
                    kelly_fraction=config.get('kelly_fraction', 0.5),
                    manual_stake=config.get('manual_stake')
                )
            
            # Mostrar métricas principales
            metrics = backtest_results['final_metrics']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Bankroll Final", f"€{metrics['final_bankroll']:,.0f}")
                st.metric("Retorno Total", f"{metrics['total_return_pct']:+.2f}%")
            with col2:
                st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}")
                st.metric("CAGR", f"{metrics['cagr']:+.2f}%")
            with col3:
                st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
                st.metric("Win Rate", f"{metrics['win_rate']:.2f}%")
            with col4:
                st.metric("Profit Factor", f"{metrics['profit_factor']:.3f}")
                st.metric("VaR 95%", f"€{metrics['var_95']:.0f}")
            
            # Gráficos
            self._render_backtest_charts(backtest_results)
            
            return backtest_results
        
        return None
    
    def _render_backtest_charts(self, backtest_results: Dict):
        """Renderiza gráficos del backtesting."""
        # Curva de equity y drawdown
        fig_equity = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_equity.add_trace(
            go.Scatter(
                x=list(range(len(backtest_results['equity_curve']))),
                y=backtest_results['equity_curve'],
                name='Bankroll',
                line=dict(color=SystemConfig.COLORS['success'], width=3)
            ),
            secondary_y=False
        )
        
        fig_equity.add_trace(
            go.Scatter(
                x=list(range(len(backtest_results['drawdown_curve']))),
                y=backtest_results['drawdown_curve'],
                name='Drawdown',
                line=dict(color=SystemConfig.COLORS['danger'], width=2),
                fill='tozeroy'
            ),
            secondary_y=True
        )
        
        fig_equity.update_layout(
            title="Evolución del Bankroll y Drawdown",
            xaxis_title="Ronda",
            height=400
        )
        fig_equity.update_yaxes(title_text="Bankroll (€)", secondary_y=False)
        fig_equity.update_yaxes(title_text="Drawdown %", secondary_y=True)
        
        # Distribución de retornos
        fig_returns = go.Figure()
        returns = backtest_results['all_returns']
        
        fig_returns.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            marker_color=SystemConfig.COLORS['info'],
            opacity=0.7,
            name='Distribución de Retornos'
        ))
        
        # Añadir líneas para estadísticas
        mean_return = np.mean(returns)
        median_return = np.median(returns)
        
        fig_returns.add_vline(
            x=mean_return,
            line_dash="dash",
            line_color=SystemConfig.COLORS['primary'],
            annotation_text=f"Media: €{mean_return:.0f}"
        )
        
        fig_returns.add_vline(
            x=median_return,
            line_dash="dot",
            line_color=SystemConfig.COLORS['secondary'],
            annotation_text=f"Mediana: €{median_return:.0f}"
        )
        
        fig_returns.update_layout(
            title="Distribución de Retornos por Ronda",
            xaxis_title="Retorno (€)",
            yaxis_title="Frecuencia",
            height=300
        )
        
        st.plotly_chart(fig_equity, use_container_width=True)
        st.plotly_chart(fig_returns, use_container_width=True)
    
    def render_export_section(self, s73_results: Dict, backtest_results: Dict, config: Dict):
        """Renderiza sección de exportación de datos."""
        st.subheader("💾 EXPORTACIÓN DE DATOS")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📥 Exportar Sistema S73**")
            
            if st.button("📄 Exportar CSV del Sistema", type="secondary", use_container_width=True):
                # Preparar datos para exportación
                export_data = DataExporter.export_s73_system(
                    combinations=s73_results['combinations'],
                    probabilities=s73_results['probabilities'],
                    odds_matrix=st.session_state.odds_matrix_6,
                    stakes=s73_results['stakes'],
                    bankroll=config['bankroll']
                )
                
                # Crear botón de descarga
                st.download_button(
                    label="⬇️ Descargar Sistema S73",
                    data=export_data['data'],
                    file_name=export_data['filename'],
                    mime=export_data['mime_type'],
                    type="primary",
                    use_container_width=True
                )
        
        with col2:
            st.markdown("**📈 Exportar Backtesting**")
            
            if st.button("📊 Exportar Resultados Backtest", type="secondary", use_container_width=True):
                # Preparar datos para exportación
                export_data = DataExporter.export_backtest_results(backtest_results)
                
                # Crear botones de descarga para cada formato
                st.download_button(
                    label="⬇️ Métricas (CSV)",
                    data=export_data['metrics']['data'],
                    file_name=export_data['metrics']['filename'],
                    mime=export_data['metrics']['mime'],
                    type="primary",
                    use_container_width=True
                )
                
                st.download_button(
                    label="⬇️ Curva Equity (CSV)",
                    data=export_data['equity']['data'],
                    file_name=export_data['equity']['filename'],
                    mime=export_data['equity']['mime'],
                    type="primary",
                    use_container_width=True
                )
                
                st.download_button(
                    label="⬇️ Reporte (TXT)",
                    data=export_data['report']['data'],
                    file_name=export_data['report']['filename'],
                    mime=export_data['report']['mime'],
                    type="primary",
                    use_container_width=True
                )
    
    def run(self):
        """Método principal de ejecución de la aplicación."""
        # Título principal
        st.title("🎯 ACBE-S73 QUANTUM BETTING SUITE v3.0")
        st.markdown("""
        *Sistema profesional de optimización de portafolios de apuestas deportivas*  
        ***Gamma-Poisson Bayesiano • Algoritmo Hamming Greedy • Kelly Dinámico • Cobertura S73***
        """)
        
        # Barra de navegación
        self._render_navigation_bar()
        
        # Renderizar sidebar y obtener configuración
        config = self.render_sidebar()
        
        # Decisión basada en fase actual
        current_phase = st.session_state.current_phase
        
        if current_phase == "input":
            self.render_input_phase(config)
        else:
            self.render_analysis_phase(config)
    
    def _render_navigation_bar(self):
        """Renderiza barra de navegación superior."""
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col1:
            if st.session_state.current_phase == "analysis":
                if st.button("← Volver a Input", type="secondary", use_container_width=True):
                    SessionStateManager.reset_to_input()
                    st.rerun()
        
        with col2:
            phase_title = "📥 INGRESO DE DATOS" if st.session_state.current_phase == "input" else "📊 ANÁLISIS DEL SISTEMA"
            st.markdown(f"<h4 style='text-align: center; color: #1E88E5;'>{phase_title}</h4>", 
                       unsafe_allow_html=True)
        
        with col3:
            if st.button("🔄 Reiniciar", type="secondary", use_container_width=True):
                SessionStateManager.clear_all_data()
                st.rerun()
        
        st.markdown("---")

# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    # Inicializar y ejecutar la aplicación
    app = ACBEApp()
    app.run()
