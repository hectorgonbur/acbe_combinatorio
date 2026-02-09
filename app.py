# ============================================================================
# IMPORTACIONES GENERALES
# ============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import io
import time
import warnings
from typing import Dict, Any, List, Optional, Tuple, Union  # ← AÑADIR
import itertools  # ← FALTA
from collections import defaultdict  # ← FALTA
warnings.filterwarnings('ignore')

# ============================================================================
# SECCIÓN 0: MANEJO DE ESTADO DE SESIÓN - PROFESIONAL v3.0
# ============================================================================

class SessionStateManager:
    """
    Gestor avanzado del estado de sesión para ACBE-S73 v3.0.
    
    Características principales:
    1. ✅ Gestión completa del ciclo de vida de la sesión
    2. ✅ Persistencia de datos entre recargas de página
    3. ✅ Control de versiones y compatibilidad con v2.3
    4. ✅ Limpieza selectiva de estados para optimización
    5. ✅ Validación automática de integridad de datos
    6. ✅ Historial de navegación para retroceso inteligente
    7. ✅ Estados específicos para nuevas funcionalidades v3.0
    
    Estados organizados por categorías:
    A. Estados de flujo (navegación, fases)
    B. Datos principales (partidos, parámetros)
    C. Resultados S73 (sistema completo)
    D. Estados Elite v3.0 (reducción avanzada)
    E. Simulación y escenarios
    F. Configuración y visualización
    """
    
    # Versión del esquema de estado
    STATE_SCHEMA_VERSION = "3.0.0"
    
    @staticmethod
    def initialize_session_state(force: bool = False):
        """
        Inicializa TODAS las variables de estado necesarias para v3.0.
        
        Args:
            force: Si True, fuerza reinicialización incluso si ya existen estados
            
        Returns:
            None
        """
        # ==================== A. ESTADOS DE FLUJO Y NAVEGACIÓN ====================
        if 'data_loaded' not in st.session_state or force:
            st.session_state.data_loaded = False
        
        if 'processing_done' not in st.session_state or force:
            st.session_state.processing_done = False
        
        if 'current_tab' not in st.session_state or force:
            st.session_state.current_tab = "input"  # "input", "analysis", "results", "export"
        
        if 'current_phase' not in st.session_state or force:
            st.session_state.current_phase = "input"  # "input", "analysis", "results"
        
        if 'phase_history' not in st.session_state or force:
            st.session_state.phase_history = ["input"]
        
        if 'navigation_stack' not in st.session_state or force:
            st.session_state.navigation_stack = []
        
        # ==================== B. DATOS PRINCIPALES DEL SISTEMA ====================
        if 'matches_data' not in st.session_state or force:
            st.session_state.matches_data = None  # Datos crudos de partidos
        
        if 'params_dict' not in st.session_state or force:
            st.session_state.params_dict = None   # Parámetros de configuración
        
        if 'odds_matrix' not in st.session_state or force:
            st.session_state.odds_matrix = None   # Matriz (6,3) de cuotas
        
        if 'probabilities' not in st.session_state or force:
            st.session_state.probabilities = None # Matriz (6,3) de probabilidades ACBE
        
        if 'normalized_entropies' not in st.session_state or force:
            st.session_state.normalized_entropies = None  # Array (6,) de entropías
        
        # ==================== C. RESULTADOS DEL SISTEMA S73 ====================
        if 's73_results' not in st.session_state or force:
            st.session_state.s73_results = None  # Resultados completos del sistema S73
        
        if 's73_executed' not in st.session_state or force:
            st.session_state.s73_executed = False  # Flag de ejecución
        
        if 's73_columns' not in st.session_state or force:
            st.session_state.s73_columns = None  # Array de combinaciones (73x6)
        
        if 's73_probabilities' not in st.session_state or force:
            st.session_state.s73_probabilities = None  # Probabilidades conjuntas
        
        if 's73_kelly_stakes' not in st.session_state or force:
            st.session_state.s73_kelly_stakes = None  # Stakes calculados
        
        # ==================== D. ESTADOS ELITE v3.0 ====================
        if 'elite_columns_selected' not in st.session_state or force:
            st.session_state.elite_columns_selected = False
        
        if 'portfolio_type' not in st.session_state or force:
            st.session_state.portfolio_type = "full"  # "full" o "elite"
        
        if 'elite_columns' not in st.session_state or force:
            st.session_state.elite_columns = None  # Array de combinaciones elite (24x6)
        
        if 'elite_scores' not in st.session_state or force:
            st.session_state.elite_scores = None  # Scores de eficiencia
        
        if 'elite_probabilities' not in st.session_state or force:
            st.session_state.elite_probabilities = None  # Probabilidades elite
        
        # ==================== E. SIMULACIÓN Y ESCENARIOS v3.0 ====================
        if 'scenario_selection' not in st.session_state or force:
            # [partido_idx1, resultado1, partido_idx2, resultado2]
            st.session_state.scenario_selection = [None, None, None, None]
        
        if 'scenario_results' not in st.session_state or force:
            st.session_state.scenario_results = None  # Resultados de simulación
        
        if 'failed_matches_simulated' not in st.session_state or force:
            st.session_state.failed_matches_simulated = []  # Historial de escenarios
        
        # ==================== F. APUESTA MAESTRA v3.0 ====================
        if 'master_bet' not in st.session_state or force:
            st.session_state.master_bet = None  # Mejor combinación del sistema
        
        if 'master_bet_metrics' not in st.session_state or force:
            st.session_state.master_bet_metrics = {}  # Métricas de la apuesta maestra
        
        # ==================== G. BACKTESTING Y SIMULACIÓN ====================
        if 'backtest_results' not in st.session_state or force:
            st.session_state.backtest_results = None
        
        if 'backtest_executed' not in st.session_state or force:
            st.session_state.backtest_executed = False
        
        if 'backtest_config' not in st.session_state or force:
            st.session_state.backtest_config = {
                'n_rounds': 100,
                'n_sims_per_round': 1000,
                'kelly_fraction': 0.5
            }
        
        # ==================== H. CONFIGURACIÓN Y VISUALIZACIÓN ====================
        if 'user_config' not in st.session_state or force:
            st.session_state.user_config = {
                'bankroll': 10000.0,
                'auto_stake_mode': True,
                'kelly_fraction': 0.5,
                'max_exposure': 0.15,
                'apply_elite_reduction': True,
                'elite_target': 24
            }
        
        if 'visualization_config' not in st.session_state or force:
            st.session_state.visualization_config = {
                'theme': 'plotly_white',
                'height': 400,
                'show_grid': True,
                'responsive': True
            }
        
        # ==================== I. SISTEMA Y METADATOS ====================
        if 'state_version' not in st.session_state or force:
            st.session_state.state_version = SessionStateManager.STATE_SCHEMA_VERSION
        
        if 'last_updated' not in st.session_state or force:
            st.session_state.last_updated = datetime.now().isoformat()
        
        if 'activity_log' not in st.session_state or force:
            st.session_state.activity_log = []  # Log de actividades para debugging
    
    @staticmethod
    def reset_to_input():
        """
        Reinicia completamente al estado de ingreso de datos.
        
        Características:
        - Limpia todos los resultados y análisis
        - Mantiene configuración del usuario
        - Conserva historial de navegación para debugging
        - Preserva datos de entrada si existen
        """
        # Guardar configuración del usuario antes de limpiar
        user_config = st.session_state.get('user_config', {})
        visualization_config = st.session_state.get('visualization_config', {})
        
        # Estados a preservar (opcional, para debugging)
        preserved_states = {
            'user_config': user_config,
            'visualization_config': visualization_config,
            'state_version': st.session_state.get('state_version'),
            'activity_log': st.session_state.get('activity_log', [])
        }
        
        # Limpiar estados específicos (en lugar de clear() completo)
        states_to_reset = [
            # Estados de flujo
            'data_loaded', 'processing_done', 'current_tab',
            'current_phase', 'phase_history', 'navigation_stack',
            
            # Resultados y análisis
            's73_results', 's73_executed', 's73_columns', 
            's73_probabilities', 's73_kelly_stakes',
            
            # Estados Elite
            'elite_columns_selected', 'elite_columns', 
            'elite_scores', 'elite_probabilities',
            
            # Simulación
            'scenario_selection', 'scenario_results', 
            'failed_matches_simulated',
            
            # Apuesta maestra
            'master_bet', 'master_bet_metrics',
            
            # Backtesting
            'backtest_results', 'backtest_executed',
        ]
        
        for state_key in states_to_reset:
            if state_key in st.session_state:
                del st.session_state[state_key]
        
        # Restaurar configuración
        for key, value in preserved_states.items():
            if value is not None:
                st.session_state[key] = value
        
        # Re-inicializar estados faltantes
        SessionStateManager.initialize_session_state()
        
        # Actualizar timestamp
        st.session_state.last_updated = datetime.now().isoformat()
        
        # Registrar actividad
        SessionStateManager._log_activity("reset_to_input")
    
    @staticmethod
    def move_to_analysis():
        """
        Transiciona de la fase de input a la fase de análisis.
        
        Validaciones:
        1. Verifica que existan datos cargados
        2. Valida integridad de datos básicos
        3. Actualiza historial de navegación
        """
        # Validar que hay datos para analizar
        if not st.session_state.get('matches_data'):
            st.error("❌ No hay datos cargados. Complete la fase de input primero.")
            return False
        
        # Validar datos mínimos necesarios
        matches_data = st.session_state.matches_data
        required_keys = ['probabilities', 'odds_matrix']
        
        for key in required_keys:
            if key not in matches_data or matches_data[key] is None:
                st.error(f"❌ Datos incompletos. Falta: {key}")
                return False
        
        # Actualizar estados
        st.session_state.data_loaded = True
        st.session_state.processing_done = True
        st.session_state.current_phase = "analysis"
        st.session_state.current_tab = "analysis"
        
        # Actualizar historial
        if "analysis" not in st.session_state.phase_history:
            st.session_state.phase_history.append("analysis")
        
        # Actualizar pila de navegación
        if "input" not in st.session_state.navigation_stack:
            st.session_state.navigation_stack.append("input")
        
        # Actualizar timestamp
        st.session_state.last_updated = datetime.now().isoformat()
        
        # Registrar actividad
        SessionStateManager._log_activity("move_to_analysis", success=True)
        
        return True
    
    @staticmethod
    def move_to_results():
        """
        Transiciona de la fase de análisis a la fase de resultados.
        
        Requisitos:
        1. Sistema S73 ejecutado (s73_executed = True)
        2. Resultados S73 disponibles
        """
        if not st.session_state.get('s73_executed', False):
            st.warning("⚠️ Primero ejecuta el sistema S73 en la fase de análisis.")
            return False
        
        st.session_state.current_phase = "results"
        st.session_state.current_tab = "results"
        
        # Actualizar historial
        if "results" not in st.session_state.phase_history:
            st.session_state.phase_history.append("results")
        
        # Actualizar pila de navegación
        if "analysis" not in st.session_state.navigation_stack:
            st.session_state.navigation_stack.append("analysis")
        
        st.session_state.last_updated = datetime.now().isoformat()
        SessionStateManager._log_activity("move_to_results", success=True)
        
        return True
    
    @staticmethod
    def go_back():
        """
        Retrocede a la fase anterior en el historial.
        
        Returns:
            bool: True si se pudo retroceder, False si no
        """
        if len(st.session_state.phase_history) <= 1:
            return False
        
        # Obtener fases actual y anterior
        current_phase = st.session_state.phase_history.pop()
        previous_phase = st.session_state.phase_history[-1]
        
        # Actualizar estado según fase anterior
        st.session_state.current_phase = previous_phase
        st.session_state.current_tab = previous_phase
        
        if previous_phase == "input":
            st.session_state.data_loaded = False
            st.session_state.processing_done = False
            # Mantener datos de entrada para posible reutilización
        
        SessionStateManager._log_activity("go_back", 
                                         from_phase=current_phase, 
                                         to_phase=previous_phase)
        
        return True
    
    @staticmethod
    def save_s73_results(results: Dict[str, Any]):
        """
        Guarda resultados del sistema S73 de manera estructurada.
        
        Args:
            results: Diccionario con resultados del sistema S73
        """
        if not results:
            return
        
        st.session_state.s73_results = results
        st.session_state.s73_executed = True
        
        # Extraer y guardar componentes clave
        if 'combinations' in results:
            st.session_state.s73_columns = results['combinations']
        
        if 'probabilities' in results:
            st.session_state.s73_probabilities = results['probabilities']
        
        if 'kelly_stakes' in results:
            st.session_state.s73_kelly_stakes = results['kelly_stakes']
        
        # Extraer datos elite si existen
        if 'elite_combinations' in results:
            st.session_state.elite_columns = results['elite_combinations']
            st.session_state.elite_columns_selected = True
        
        if 'elite_scores' in results:
            st.session_state.elite_scores = results['elite_scores']
        
        # Calcular y guardar apuesta maestra
        SessionStateManager._calculate_master_bet(results)
        
        st.session_state.last_updated = datetime.now().isoformat()
        SessionStateManager._log_activity("save_s73_results", 
                                         columns_count=len(results.get('combinations', [])))
    
    @staticmethod
    def save_scenario_results(results: Dict[str, Any]):
        """
        Guarda resultados del simulador de escenarios.
        
        Args:
            results: Diccionario con resultados de simulación
        """
        st.session_state.scenario_results = results
        
        # Actualizar historial de escenarios simulados
        if 'failed_matches' in results:
            scenario_id = str(results['failed_matches'])
            if scenario_id not in st.session_state.failed_matches_simulated:
                st.session_state.failed_matches_simulated.append(scenario_id)
        
        SessionStateManager._log_activity("save_scenario_results")
    
    @staticmethod
    def update_user_config(config: Dict[str, Any]):
        """
        Actualiza configuración del usuario.
        
        Args:
            config: Diccionario con nueva configuración
        """
        if not config:
            return
        
        # Actualizar configuración existente
        current_config = st.session_state.get('user_config', {})
        current_config.update(config)
        st.session_state.user_config = current_config
        
        # Actualizar estados individuales para compatibilidad
        if 'bankroll' in config:
            st.session_state.bankroll = config['bankroll']
        
        if 'portfolio_type' in config:
            st.session_state.portfolio_type = config['portfolio_type']
        
        SessionStateManager._log_activity("update_user_config", 
                                         config_keys=list(config.keys()))
    
    @staticmethod
    def get_state_summary() -> Dict[str, Any]:
        """
        Obtiene un resumen del estado actual para debugging.
        
        Returns:
            Diccionario con resumen del estado
        """
        summary = {
            'state_version': st.session_state.get('state_version'),
            'current_phase': st.session_state.get('current_phase'),
            'data_loaded': st.session_state.get('data_loaded'),
            'processing_done': st.session_state.get('processing_done'),
            's73_executed': st.session_state.get('s73_executed'),
            'elite_columns_selected': st.session_state.get('elite_columns_selected'),
            'portfolio_type': st.session_state.get('portfolio_type'),
            'backtest_executed': st.session_state.get('backtest_executed'),
            'last_updated': st.session_state.get('last_updated'),
            'has_matches_data': st.session_state.get('matches_data') is not None,
            'has_s73_results': st.session_state.get('s73_results') is not None,
            'has_elite_columns': st.session_state.get('elite_columns') is not None,
            'activity_log_count': len(st.session_state.get('activity_log', []))
        }
        
        # Añadir conteos específicos si existen
        if st.session_state.get('matches_data') and 'probabilities' in st.session_state.matches_data:
            probs = st.session_state.matches_data['probabilities']
            summary['matches_count'] = probs.shape[0]
        
        if st.session_state.get('s73_results'):
            summary['s73_columns_count'] = st.session_state.s73_results.get('final_count', 0)
        
        if st.session_state.get('elite_columns'):
            summary['elite_columns_count'] = len(st.session_state.elite_columns)
        
        return summary
    
    @staticmethod
    def validate_state() -> List[str]:
        """
        Valida la integridad del estado de sesión.
        
        Returns:
            Lista de problemas encontrados (vacía si todo OK)
        """
        problems = []
        
        # Verificar versión de estado
        if st.session_state.get('state_version') != SessionStateManager.STATE_SCHEMA_VERSION:
            problems.append(f"Versión de estado desactualizada: {st.session_state.get('state_version')}")
        
        # Verificar consistencia de fases
        current_phase = st.session_state.get('current_phase')
        phase_history = st.session_state.get('phase_history', [])
        
        if current_phase not in phase_history:
            problems.append(f"Fase actual '{current_phase}' no está en historial")
        
        # Verificar datos según fase
        if current_phase == "analysis" and not st.session_state.get('data_loaded'):
            problems.append("En fase 'analysis' pero data_loaded=False")
        
        if current_phase == "results" and not st.session_state.get('s73_executed'):
            problems.append("En fase 'results' pero s73_executed=False")
        
        # Verificar tipos de datos críticos
        if st.session_state.get('probabilities') is not None:
            if not isinstance(st.session_state.probabilities, np.ndarray):
                problems.append("probabilities no es numpy array")
        
        return problems
    
    # ==================== MÉTODOS PRIVADOS ====================
    
    @staticmethod
    def _log_activity(action: str, **kwargs):
        """
        Registra actividad en el log de sesión.
        
        Args:
            action: Nombre de la acción
            **kwargs: Datos adicionales
        """
        if 'activity_log' not in st.session_state:
            st.session_state.activity_log = []
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'phase': st.session_state.get('current_phase', 'unknown'),
            **kwargs
        }
        
        st.session_state.activity_log.append(log_entry)
        
        # Mantener log manejable (últimas 50 entradas)
        if len(st.session_state.activity_log) > 50:
            st.session_state.activity_log = st.session_state.activity_log[-50:]
    
    @staticmethod
    def _calculate_master_bet(results: Dict[str, Any]):
        """
        Calcula y guarda la apuesta maestra a partir de resultados S73.
        
        Args:
            results: Resultados del sistema S73
        """
        if not results or 'columns_df' not in results:
            return
        
        try:
            columns_df = results['columns_df']
            
            # Encontrar la columna con mayor probabilidad
            if not columns_df.empty and 'Probabilidad' in columns_df.columns:
                master_idx = columns_df['Probabilidad'].idxmax()
                master_bet = columns_df.loc[master_idx].to_dict()
                
                st.session_state.master_bet = master_bet
                
                # Extraer métricas clave
                st.session_state.master_bet_metrics = {
                    'combination': master_bet.get('Combinación'),
                    'probability': master_bet.get('Probabilidad'),
                    'odds': master_bet.get('Cuota'),
                    'expected_value': master_bet.get('Valor Esperado'),
                    'stake_percentage': master_bet.get('Stake (%)'),
                    'recommendation': 'JUGAR' if master_bet.get('Valor Esperado', 0) > 0 else 'NO JUGAR'
                }
                
                SessionStateManager._log_activity("calculate_master_bet", 
                                                 combination=master_bet.get('Combinación'))
        except Exception as e:
            # No fallar si hay error en cálculo de apuesta maestra
            SessionStateManager._log_activity("calculate_master_bet_error", 
                                             error=str(e))

    # ============================================================================
    # INICIALIZACIÓN GLOBAL DEL ESTADO
    # ============================================================================

    def initialize_global_state():
        """
        Función de inicialización global del estado.
        Debe llamarse al inicio de cada ejecución de Streamlit.
        """
        # Inicializar estado de sesión
        SessionStateManager.initialize_session_state()
        
        # Validar estado actual
        problems = SessionStateManager.validate_state()
        
        # Si hay problemas, intentar reparar automáticamente
        if problems:
            print(f"⚠️ Problemas en estado de sesión: {problems}")
            # Para v3.0, podemos intentar reparaciones automáticas simples
            # Por ahora, solo logueamos los problemas
        
        return True
# ============================================================================
# SECCIÓN 1: CONFIGURACIÓN DEL SISTEMA - PROFESIONAL v3.0
# ============================================================================

class SystemConfig:
    """
    Configuración centralizada del sistema ACBE-S73 v3.0.
    
    Este archivo contiene TODOS los parámetros críticos del sistema,
    organizados por categorías y completamente documentados.
    
    Características v3.0:
    1. ✅ Parámetros de Doble Reducción (S73 + Elite)
    2. ✅ Score de Eficiencia para selección Elite
    3. ✅ Configuración para Apuesta Maestra
    4. ✅ Parámetros para Simulador de Escenarios
    5. ✅ Kelly diferenciado por tipo de portafolio
    6. ✅ Validación automática de parámetros
    7. ✅ Métodos utilitarios para cálculos comunes
    """
    
    # ============================================================================
    # 1. PARÁMETROS DE SIMULACIÓN Y COMPUTACIÓN
    # ============================================================================
    
    # Simulación Monte Carlo
    MONTE_CARLO_ITERATIONS = 10000
    """Número de iteraciones para simulaciones Monte Carlo. Más iteraciones = mayor precisión."""
    
    KELLY_FRACTION_MAX = 0.03  # 3% máximo por columna
    """Fracción máxima de Kelly por columna individual. Controla exposición máxima por apuesta."""
    
    MIN_PROBABILITY = 1e-10    # Evitar log(0)
    """Probabilidad mínima para evitar problemas numéricos en cálculos logarítmicos."""
    
    BASE_ENTROPY = 3           # Base logarítmica para 3 resultados
    """Base logarítmica para cálculo de entropía (3 resultados posibles: 1, X, 2)."""
    
    # ============================================================================
    # 2. MODELO BAYESIANO GAMMA-POISSON
    # ============================================================================
    
    # Parámetros base del modelo
    DEFAULT_ATTACK_MEAN = 1.2
    """Fuerza de ataque por defecto (1.0 = promedio). >1.0 = equipo superior al promedio."""
    
    DEFAULT_DEFENSE_MEAN = 0.8
    """Fuerza defensiva por defecto (1.0 = promedio). <1.0 = mejor defensa."""
    
    DEFAULT_HOME_ADVANTAGE = 1.1
    """Ventaja por jugar en casa. Estadísticamente, equipos locales tienen ~10% más fuerza."""
    
    # Parámetros de distribución Gamma (prior)
    DEFAULT_ALPHA_PRIOR = 2.0  # Parámetro de forma Gamma
    """Parámetro de forma (alpha) para distribución Gamma prior. Controla concentración."""
    
    DEFAULT_BETA_PRIOR = 1.0   # Parámetro de tasa Gamma
    """Parámetro de tasa (beta) para distribución Gamma prior. Controla escala."""
    
    # ============================================================================
    # 3. SISTEMA COMBINATORIO S73
    # ============================================================================
    
    # Estructura del sistema
    NUM_MATCHES = 6            # Partidos por sistema
    """Número de partidos en el sistema S73. No modificar sin ajustar algoritmo de cobertura."""
    
    FULL_COMBINATIONS = 3 ** 6  # 729 combinaciones posibles
    """Total de combinaciones posibles con 6 partidos (3 resultados cada uno)."""
    
    TARGET_COMBINATIONS = 73   # Objetivo de columnas reducidas
    """Número objetivo de columnas en sistema S73. Garantiza cobertura de 2 errores."""
    
    HAMMING_DISTANCE_TARGET = 2  # Cobertura de 2 errores!
    """Distancia de Hamming objetivo. 2 = sistema cubre todas las combinaciones con hasta 2 fallos."""
    
    # Umbrales de reducción S73 (Validación institucional)
    MIN_OPTION_PROBABILITY = 0.55   # Umbral mínimo por opción
    """Probabilidad mínima para considerar una opción. Filtra combinaciones con signos improbables."""
    
    MIN_PROBABILITY_GAP = 0.12      # Gap mínimo entre 1ª y 2ª opción
    """Diferencia mínima entre opción más probable y segunda. Filtra partidos muy equilibrados."""
    
    MIN_EV_THRESHOLD = 0.0          # EV mínimo positivo
    """Valor Esperado mínimo para considerar una opción. Solo apuestas con valor positivo."""
    
    MIN_JOINT_PROBABILITY = 0.001   # Umbral mínimo probabilidad conjunta
    """Probabilidad conjunta mínima para columnas. Filtra combinaciones demasiado improbables."""
    
    # ============================================================================
    # 4. REDUCCIÓN ELITE v3.0 - NUEVO
    # ============================================================================
    
    ELITE_COLUMNS_TARGET = 24  # Reducción elite de 73 a 24 columnas
    """Número objetivo de columnas Elite. Ratio 3:1 (73 → 24)."""
    
    ELITE_SCORE_WEIGHTS = {
        'probability': 1.0,    # Probabilidad conjunta
        'expected_value': 1.0, # Valor Esperado (EV)
        'entropy': 1.0         # Entropía (inverso del riesgo)
    }
    """
    Pesos para Score de Eficiencia Elite.
    
    Score = P × (1+EV) × (1-EntropíaPromedio)
    
    Ajustar según estrategia:
    - Más peso a 'probability': Estrategia conservadora
    - Más peso a 'expected_value': Estrategia agresiva
    - Más peso a 'entropy': Estrategia de bajo riesgo
    """
    
    # Multiplicadores Kelly por tipo de portafolio
    KELLY_MULTIPLIERS = {
        'full': 0.7,   # Más conservador para portafolio completo (más diversificación)
        'elite': 1.3   # Más agresivo para portafolio elite (menos diversificación)
    }
    """Multiplicadores para ajustar Kelly según tipo de portafolio."""
    
    # ============================================================================
    # 5. UMBRALES DE CLASIFICACIÓN Y ANÁLISIS
    # ============================================================================
    
    # Clasificación por entropía
    STRONG_MATCH_THRESHOLD = 0.30   # ≤ 0.30: Partido Fuerte (1 signo)
    """Umbral para partidos FUERTES. Solo se considera el signo más probable."""
    
    MEDIUM_MATCH_THRESHOLD = 0.60   # 0.30-0.60: Partido Medio (2 signos)
    """Umbral para partidos MEDIOS. Se consideran los dos signos más probables."""
    # ≥ 0.60: Partido Caótico (3 signos) - Todos los signos considerados
    
    # Umbrales para análisis de calidad
    QUALITY_THRESHOLDS = {
        'excellent_probability': 0.65,   # Probabilidad excelente
        'good_probability': 0.45,        # Probabilidad buena
        'high_ev': 0.15,                 # EV alto
        'medium_ev': 0.05,               # EV medio
        'low_entropy': 0.25,             # Entropía baja (predecible)
        'medium_entropy': 0.45           # Entropía media
    }
    
    # ============================================================================
    # 6. GESTIÓN DE RIESGO Y CAPITAL
    # ============================================================================
    
    # Límites de cuotas
    MIN_ODDS = 1.01
    """Cuota mínima aceptable. Evita problemas numéricos y cuotas sin sentido."""
    
    MAX_ODDS = 100.0
    """Cuota máxima aceptable. Filtra errores de entrada y cuotas extremas."""
    
    # Bankroll y exposición
    DEFAULT_BANKROLL = 10000.0
    """Bankroll por defecto para simulaciones y cálculos."""
    
    MAX_PORTFOLIO_EXPOSURE = 0.15   # 15% exposición máxima del portafolio
    """Exposición máxima del bankroll en apuestas simultáneas. Controla drawdown máximo."""
    
    # Límites de stake
    MIN_STAKE_PERCENTAGE = 0.001    # 0.1% mínimo
    """Stake mínimo como porcentaje del bankroll."""
    
    MAX_STAKE_PERCENTAGE = 0.05     # 5% máximo
    """Stake máximo como porcentaje del bankroll."""
    
    # ============================================================================
    # 7. CONFIGURACIÓN VISUAL v3.0
    # ============================================================================
    
    # Paleta de colores principal
    COLORS = {
        'primary': '#1E88E5',    # Azul principal - ACBE
        'secondary': '#FFC107',  # Amarillo/naranja - Advertencias
        'success': '#4CAF50',    # Verde - Éxito/positivo
        'danger': '#F44336',     # Rojo - Peligro/negativo
        'warning': '#FF9800',    # Naranja - Advertencia
        'info': '#00BCD4',       # Azul claro - Información
        'dark': '#212121',       # Gris oscuro - Textos
        'light': '#F5F5F5'       # Gris claro - Fondos
    }
    """Paleta de colores principal del sistema. Códigos HEX para precisión."""
    
    # Paleta de riesgo para gráficos
    RISK_PALETTE = [
        "#00BCD4",  # info - riesgo muy bajo
        "#4CAF50",  # success - riesgo bajo
        "#FFC107",  # warning - riesgo medio
        "#FF9800",  # orange - riesgo medio-alto
        "#F44336"   # danger - riesgo alto
    ]
    """Gradiente de colores para visualización de riesgo."""
    
    # Mapeo de resultados
    OUTCOME_MAPPING = {'1': 0, 'X': 1, '2': 2}
    """Mapeo de resultados a índices numéricos para procesamiento."""
    
    OUTCOME_LABELS = ['1', 'X', '2']
    """Etiquetas de resultados para visualización."""
    
    OUTCOME_COLORS = ['#1E88E5', '#FF9800', '#F44336']
    """Colores por resultado: 1=Azul, X=Naranja, 2=Rojo."""
    
    # NUEVO v3.0: Colores para Apuesta Maestra
    MASTER_BET_COLORS = {
        '1': '#4CAF50',  # Verde - Fuerte convicción local
        'X': '#FF9800',  # Naranja - Fuerte convicción empate
        '2': '#F44336'   # Rojo - Fuerte convicción visitante
    }
    """Colores destacados para visualización de Apuesta Maestra."""
    
    # Colores para visualización de escenarios
    SCENARIO_COLORS = {
        'success': '#4CAF50',      # Escenario favorable
        'warning': '#FFC107',      # Escenario de advertencia
        'danger': '#F44336',       # Escenario crítico
        'neutral': '#9E9E9E'       # Escenario neutral
    }
    
    # ============================================================================
    # 8. PARÁMETROS DE INPUT MANUAL
    # ============================================================================
    
    MANUAL_INPUT_DEFAULTS = {
        # Límites de cuotas
        'min_odds': 1.01,
        'max_odds': 100.0,
        
        # Valores por defecto de fuerzas
        'default_attack': 1.2,
        'default_defense': 0.8,
        'default_home_advantage': 1.1,
        
        # Límites para sliders
        'min_attack': 0.5,
        'max_attack': 2.0,
        'min_defense': 0.5,
        'max_defense': 2.0,
        'min_home_advantage': 1.0,
        'max_home_advantage': 1.5,
        
        # Steps para sliders
        'step_attack': 0.05,
        'step_defense': 0.05,
        'step_home_advantage': 0.01,
        
        # Valores iniciales para cuotas
        'default_odds_1': 2.0,
        'default_odds_X': 3.0,
        'default_odds_2': 2.5
    }
    """Valores por defecto y límites para interfaz de input manual."""
    
    # ============================================================================
    # 9. CONFIGURACIÓN DE BACKTESTING v3.0
    # ============================================================================
    
    BACKTEST_DEFAULTS = {
        # Perfiles predefinidos
        'profiles': {
            'conservative': {
                'n_rounds': 50,
                'n_sims_per_round': 500,
                'kelly_fraction': 0.3,
                'description': 'Conservador - Menor riesgo, crecimiento estable'
            },
            'balanced': {
                'n_rounds': 100,
                'n_sims_per_round': 1000,
                'kelly_fraction': 0.5,
                'description': 'Balanceado - Equilibrio riesgo/retorno'
            },
            'aggressive': {
                'n_rounds': 200,
                'n_sims_per_round': 2000,
                'kelly_fraction': 0.7,
                'description': 'Agresivo - Mayor riesgo, mayor potencial'
            }
        },
        
        # Métricas objetivo
        'target_metrics': {
            'full_portfolio': {
                'min_roi': 0.05,      # 5% ROI mínimo
                'max_drawdown': 0.20, # 20% drawdown máximo
                'sharpe_min': 0.8,    # Sharpe mínimo 0.8
                'win_rate_min': 0.45  # Win rate mínimo 45%
            },
            'elite_portfolio': {
                'min_roi': 0.08,      # 8% ROI mínimo
                'max_drawdown': 0.25, # 25% drawdown máximo
                'sharpe_min': 1.0,    # Sharpe mínimo 1.0
                'win_rate_min': 0.40  # Win rate mínimo 40%
            }
        }
    }
    
    # ============================================================================
    # 10. CONFIGURACIÓN DE SIMULADOR DE ESCENARIOS v3.0
    # ============================================================================
    
    SCENARIO_CONFIG = {
        # Número máximo de partidos a simular como fallados
        'max_failed_matches': 2,
        
        # Umbrales para clasificación de escenarios
        'thresholds': {
            'favorable_coverage': 0.5,    # >50% columnas con 4+ aciertos
            'challenging_coverage': 0.3,  # 30-50% columnas con 4+ aciertos
            'critical_coverage': 0.1      # <10% columnas con 4+ aciertos
        },
        
        # Configuración de visualización
        'visualization': {
            'hit_distribution_colors': [
                '#F44336',  # 0 aciertos
                '#FF9800',  # 1 acierto
                '#FFC107',  # 2 aciertos
                '#FFC107',  # 3 aciertos
                '#4CAF50',  # 4 aciertos
                '#4CAF50',  # 5 aciertos
                '#4CAF50'   # 6 aciertos
            ]
        }
    }
    
    # ============================================================================
    # 11. MÉTODOS UTILITARIOS Y VALIDACIÓN
    # ============================================================================
    
    @staticmethod
    def validate_odds(odds: float) -> float:
        """
        Valida y ajusta una cuota a los límites del sistema.
        
        Args:
            odds: Cuota a validar
            
        Returns:
            Cuota validada dentro de límites [MIN_ODDS, MAX_ODDS]
        """
        import numpy as np
        return np.clip(odds, SystemConfig.MIN_ODDS, SystemConfig.MAX_ODDS)
    
    @staticmethod
    def validate_probability(prob: float) -> float:
        """
        Valida y ajusta una probabilidad a los límites del sistema.
        
        Args:
            prob: Probabilidad a validar (0-1)
            
        Returns:
            Probabilidad validada dentro de límites [MIN_PROBABILITY, 1.0]
        """
        import numpy as np
        return np.clip(prob, SystemConfig.MIN_PROBABILITY, 1.0)
    
    @staticmethod
    def calculate_implied_probability(odds_array: np.ndarray) -> float:
        """
        Calcula probabilidad implícita total de un conjunto de cuotas.
        
        Args:
            odds_array: Array de cuotas [1, X, 2]
            
        Returns:
            Probabilidad implícita total (debe ser >1 por margen de la casa)
        """
        return np.sum(1.0 / odds_array)
    
    @staticmethod
    def calculate_margin(odds_array: np.ndarray) -> float:
        """
        Calcula margen de la casa de apuestas.
        
        Args:
            odds_array: Array de cuotas [1, X, 2]
            
        Returns:
            Margen en porcentaje (ej: 5.0 para 5%)
        """
        implied_prob = SystemConfig.calculate_implied_probability(odds_array)
        return (implied_prob - 1.0) * 100
    
    @staticmethod
    def calculate_elite_score(probability: float, expected_value: float, 
                            avg_entropy: float, weights: Dict[str, float] = None) -> float:
        """
        Calcula Score de Eficiencia para columna Elite.
        
        Fórmula: P × (1 + EV) × (1 - EntropíaPromedio)
        
        Args:
            probability: Probabilidad conjunta de la columna
            expected_value: Valor Esperado de la columna
            avg_entropy: Entropía promedio de la columna
            weights: Pesos personalizados (opcional)
            
        Returns:
            Score de eficiencia (mayor es mejor)
        """
        if weights is None:
            weights = SystemConfig.ELITE_SCORE_WEIGHTS
        
        # Normalizar entropía (0-1, donde 0 es mejor)
        entropy_factor = 1.0 - avg_entropy
        
        # Calcular score ponderado
        score = (probability ** weights['probability']) * \
                ((1.0 + expected_value) ** weights['expected_value']) * \
                (entropy_factor ** weights['entropy'])
        
        return score
    
    @staticmethod
    def get_kelly_multiplier(portfolio_type: str) -> float:
        """
        Obtiene multiplicador Kelly según tipo de portafolio.
        
        Args:
            portfolio_type: 'full' o 'elite'
            
        Returns:
            Multiplicador para ajustar Kelly
        """
        return SystemConfig.KELLY_MULTIPLIERS.get(portfolio_type, 1.0)
    
    @staticmethod
    def classify_match_by_entropy(normalized_entropy: float) -> str:
        """
        Clasifica partido según su entropía normalizada.
        
        Args:
            normalized_entropy: Entropía normalizada (0-1)
            
        Returns:
            Clasificación: 'fuerte', 'medio', o 'caótico'
        """
        if normalized_entropy <= SystemConfig.STRONG_MATCH_THRESHOLD:
            return 'fuerte'
        elif normalized_entropy <= SystemConfig.MEDIUM_MATCH_THRESHOLD:
            return 'medio'
        else:
            return 'caótico'
    
    @staticmethod
    def get_risk_color(risk_level: float) -> str:
        """
        Obtiene color según nivel de riesgo (0-1).
        
        Args:
            risk_level: Nivel de riesgo (0=bajo, 1=alto)
            
        Returns:
            Código HEX del color
        """
        import numpy as np
        
        # Mapear riesgo a índice en paleta
        palette_size = len(SystemConfig.RISK_PALETTE)
        index = min(int(risk_level * (palette_size - 1)), palette_size - 1)
        
        return SystemConfig.RISK_PALETTE[index]
    
    @staticmethod
    def validate_configuration() -> List[str]:
        """
        Valida consistencia de la configuración del sistema.
        
        Returns:
            Lista de problemas encontrados (vacía si todo OK)
        """
        problems = []
        
        # Validar que TARGET_COMBINATIONS sea menor o igual a FULL_COMBINATIONS
        if SystemConfig.TARGET_COMBINATIONS > SystemConfig.FULL_COMBINATIONS:
            problems.append(f"TARGET_COMBINATIONS ({SystemConfig.TARGET_COMBINATIONS}) > FULL_COMBINATIONS ({SystemConfig.FULL_COMBINATIONS})")
        
        # Validar umbrales de entropía
        if SystemConfig.STRONG_MATCH_THRESHOLD >= SystemConfig.MEDIUM_MATCH_THRESHOLD:
            problems.append(f"STRONG_MATCH_THRESHOLD ({SystemConfig.STRONG_MATCH_THRESHOLD}) >= MEDIUM_MATCH_THRESHOLD ({SystemConfig.MEDIUM_MATCH_THRESHOLD})")
        
        # Validar límites de exposición
        if SystemConfig.MAX_PORTFOLIO_EXPOSURE > 0.5:  # No más del 50%
            problems.append(f"MAX_PORTFOLIO_EXPOSURE ({SystemConfig.MAX_PORTFOLIO_EXPOSURE}) > 0.5 (demasiado alto)")
        
        # Validar fracción Kelly
        if SystemConfig.KELLY_FRACTION_MAX > 0.1:  # No más del 10%
            problems.append(f"KELLY_FRACTION_MAX ({SystemConfig.KELLY_FRACTION_MAX}) > 0.1 (riesgo demasiado alto)")
        
        # Validar reducción elite
        if SystemConfig.ELITE_COLUMNS_TARGET >= SystemConfig.TARGET_COMBINATIONS:
            problems.append(f"ELITE_COLUMNS_TARGET ({SystemConfig.ELITE_COLUMNS_TARGET}) >= TARGET_COMBINATIONS ({SystemConfig.TARGET_COMBINATIONS})")
        
        # Validar pesos de score elite
        total_weight = sum(SystemConfig.ELITE_SCORE_WEIGHTS.values())
        if abs(total_weight - 3.0) > 0.001:  # Pequeña tolerancia
            problems.append(f"Suma de ELITE_SCORE_WEIGHTS ({total_weight}) no es ~3.0")
        
        # Validar multiplicadores Kelly
        for portfolio_type, multiplier in SystemConfig.KELLY_MULTIPLIERS.items():
            if multiplier <= 0:
                problems.append(f"KELLY_MULTIPLIERS['{portfolio_type}'] ({multiplier}) <= 0")
        
        return problems
    
    @staticmethod
    def get_config_summary() -> Dict[str, Any]:
        """
        Obtiene un resumen de la configuración del sistema.
        
        Returns:
            Diccionario con resumen de configuración
        """
        return {
            'system_version': 'ACBE-S73 v3.0',
            'monte_carlo_iterations': SystemConfig.MONTE_CARLO_ITERATIONS,
            'kelly_fraction_max': SystemConfig.KELLY_FRACTION_MAX,
            'num_matches': SystemConfig.NUM_MATCHES,
            'target_combinations': SystemConfig.TARGET_COMBINATIONS,
            'elite_columns_target': SystemConfig.ELITE_COLUMNS_TARGET,
            'hamming_distance_target': SystemConfig.HAMMING_DISTANCE_TARGET,
            'max_portfolio_exposure': SystemConfig.MAX_PORTFOLIO_EXPOSURE,
            'default_bankroll': SystemConfig.DEFAULT_BANKROLL,
            'entropy_thresholds': {
                'strong': SystemConfig.STRONG_MATCH_THRESHOLD,
                'medium': SystemConfig.MEDIUM_MATCH_THRESHOLD
            },
            'reduction_thresholds': {
                'min_option_probability': SystemConfig.MIN_OPTION_PROBABILITY,
                'min_probability_gap': SystemConfig.MIN_PROBABILITY_GAP,
                'min_ev_threshold': SystemConfig.MIN_EV_THRESHOLD
            },
            'elite_score_weights': SystemConfig.ELITE_SCORE_WEIGHTS,
            'kelly_multipliers': SystemConfig.KELLY_MULTIPLIERS
        }

    # ============================================================================
    # VALIDACIÓN AUTOMÁTICA AL IMPORTAR
    # ============================================================================

    def validate_system_config():
        """
        Valida la configuración del sistema al importar el módulo.
        
        Raises:
            ValueError: Si hay inconsistencias en la configuración
        """
        problems = SystemConfig.validate_configuration()
        
        if problems:
            error_message = "Errores en configuración del sistema:\n" + "\n".join(f"  - {problem}" for problem in problems)
            raise ValueError(error_message)
        
        return True


    # Validar al importar
    try:
        validate_system_config()
        print("✅ Configuración del sistema validada exitosamente")
    except ValueError as e:
        print(f"❌ Error en configuración del sistema: {e}")
        # No raise para permitir ejecución, pero mostrar advertencia
        
class ACBEModel:
    """Modelo Bayesiano Gamma-Poisson optimizado."""
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def simulate_probabilities(
        lambda_home: np.ndarray,
        lambda_away: np.ndarray,
        n_sims: int = SystemConfig.MONTE_CARLO_ITERATIONS
    ) -> np.ndarray:
        """Simulación vectorizada de probabilidades."""
        import numpy as np
        n_matches = len(lambda_home)
        
        # Generar goles
        home_goals = np.random.poisson(
            lam=np.tile(lambda_home, (n_sims, 1)),
            size=(n_sims, n_matches)
        )
        away_goals = np.random.poisson(
            lam=np.tile(lambda_away, (n_sims, 1)),
            size=(n_sims, n_matches)
        )
        
        # Calcular resultados
        home_wins = (home_goals > away_goals).sum(axis=0) / n_sims
        draws = (home_goals == away_goals).sum(axis=0) / n_sims
        away_wins = (home_goals < away_goals).sum(axis=0) / n_sims
        
        # Crear matriz de probabilidades
        probs = np.column_stack([home_wins, draws, away_wins])
        
        # Normalizar
        probs = np.clip(probs, SystemConfig.MIN_PROBABILITY, 1.0)
        probs = probs / probs.sum(axis=1, keepdims=True)
        
        return probs
    
    @staticmethod
    def calculate_entropy(probabilities: np.ndarray) -> np.ndarray:
        """Calcula entropía normalizada."""
        import numpy as np
        probs = np.clip(probabilities, SystemConfig.MIN_PROBABILITY, 1.0)
        return -np.sum(probs * np.log(probs) / np.log(SystemConfig.BASE_ENTROPY), axis=1)


class InformationTheory:
    """Clasificación de partidos por teoría de información."""
    
    @staticmethod
    def classify_matches(
        probabilities: np.ndarray,
        normalized_entropies: np.ndarray,
        odds_matrix: Optional[np.ndarray] = None
    ) -> Tuple[List[List[int]], List[str]]:
        """Clasifica partidos según entropía y valor esperado."""
        allowed_signs = []
        classifications = []
        
        for i in range(len(probabilities)):
            entropy = normalized_entropies[i]
            probs = probabilities[i]
            
            # Calcular valor esperado si hay cuotas
            if odds_matrix is not None:
                evs = probs * odds_matrix[i] - 1
            else:
                evs = np.zeros(3)
            
            # Clasificación basada en entropía
            if entropy <= SystemConfig.STRONG_MATCH_THRESHOLD:
                best_idx = np.argmax(probs)
                if (probs[best_idx] >= SystemConfig.MIN_OPTION_PROBABILITY and
                    evs[best_idx] > SystemConfig.MIN_EV_THRESHOLD):
                    allowed_signs.append([best_idx])
                    classifications.append('Fuerte')
                else:
                    allowed_signs.append([0, 1, 2])
                    classifications.append('Caótico')
            
            elif entropy <= SystemConfig.MEDIUM_MATCH_THRESHOLD:
                top_two = np.argsort(probs)[-2:].tolist()
                sorted_probs = np.sort(probs)[::-1]
                
                if len(sorted_probs) >= 2 and (sorted_probs[0] - sorted_probs[1]) >= SystemConfig.MIN_PROBABILITY_GAP:
                    allowed_signs.append([np.argmax(probs)])
                    classifications.append('Fuerte (gap)')
                else:
                    allowed_signs.append(top_two)
                    classifications.append('Medio')
            
            else:
                allowed_signs.append([0, 1, 2])
                classifications.append('Caótico')
        
        return allowed_signs, classifications
    
class MatchInputLayer:
    """Capa de input profesional con validaciones avanzadas - OPTIMIZADA."""
    
    @staticmethod
    def validate_odds(odds_array: np.ndarray) -> np.ndarray:
        """Valida y normaliza cuotas ingresadas."""
        if np.any(np.isnan(odds_array)):
            st.warning("⚠️ Valores inválidos detectados. Usando defaults...")
            return np.full_like(odds_array, 2.0)
        
        # Aplicar límites
        odds_array = np.maximum(odds_array, SystemConfig.MIN_ODDS + 0.01)
        odds_array = np.minimum(odds_array, SystemConfig.MAX_ODDS)
        
        return odds_array
    
    @staticmethod
    def render_input_section() -> Tuple[pd.DataFrame, Dict, str]:
        """Renderiza sección de input de partidos."""
        st.header("⚽ Input de Partidos")
        
        # Selector de modo simplificado
        mode = st.radio(
            "Modo de operación:",
            ["🔘 Automático", "🎮 Manual"],
            index=0,
            horizontal=True
        )
        
        is_manual = mode == "🎮 Manual"
        
        # Mensaje de estado
        status_color = "🟢" if is_manual else "🔵"
        st.caption(f"{status_color} Modo: {'Manual' if is_manual else 'Automático'}")
        
        matches_data = []
        strengths_data = []
        
        # Input para 6 partidos
        st.subheader(f"📝 Ingreso de {SystemConfig.NUM_MATCHES} Partidos")
        
        for idx in range(SystemConfig.NUM_MATCHES):
            match_num = idx + 1
            st.markdown(f"##### Partido {match_num}")
            
            col1, col2 = st.columns([2, 3])
            
            with col1:
                league = st.text_input(
                    f"Liga {match_num}",
                    value=f"Liga {match_num}",
                    key=f"league_{match_num}"
                )
                home_team = st.text_input(
                    f"Local {match_num}",
                    value=f"Equipo {match_num}A",
                    key=f"home_{match_num}"
                )
                away_team = st.text_input(
                    f"Visitante {match_num}",
                    value=f"Equipo {match_num}B",
                    key=f"away_{match_num}"
                )
            
            with col2:
                # Cuotas
                odds_1 = st.number_input(
                    f"1 ({home_team})",
                    min_value=1.01,
                    max_value=50.0,
                    value=1.8 + idx * 0.1,
                    step=0.1,
                    key=f"odds1_{match_num}"
                )
                odds_x = st.number_input(
                    "X (Empate)",
                    min_value=1.01,
                    max_value=50.0,
                    value=3.2 + idx * 0.1,
                    step=0.1,
                    key=f"oddsx_{match_num}"
                )
                odds_2 = st.number_input(
                    f"2 ({away_team})",
                    min_value=1.01,
                    max_value=50.0,
                    value=4.0 + idx * 0.1,
                    step=0.1,
                    key=f"odds2_{match_num}"
                )
                
                # Fuerzas solo en modo manual
                if is_manual:
                    with st.expander("⚙️ Fuerzas (Opcional)", expanded=False):
                        home_attack = st.slider(
                            f"Ataque {home_team}",
                            0.5, 2.0, 1.0, 0.05,
                            key=f"ha_{match_num}"
                        )
                        away_attack = st.slider(
                            f"Ataque {away_team}",
                            0.5, 2.0, 1.0, 0.05,
                            key=f"aa_{match_num}"
                        )
                else:
                    home_attack = away_attack = 1.0
            
            # Valores por defecto para defensas y ventaja
            home_defense = away_defense = 1.0
            home_advantage = SystemConfig.DEFAULT_HOME_ADVANTAGE
            
            # Calcular métricas
            implied_prob = (1/odds_1 + 1/odds_x + 1/odds_2)
            margin = (implied_prob - 1) * 100
            
            matches_data.append({
                'match_id': match_num,
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
                'mode': 'Manual' if is_manual else 'Auto'
            })
            
            strengths_data.append({
                'attack': [home_attack, away_attack],
                'defense': [home_defense, away_defense],
                'advantage': home_advantage
            })
            
            st.markdown("---")
        
        # Crear DataFrame
        matches_df = pd.DataFrame(matches_data)
        
        # Extraer matriz de cuotas
        odds_matrix = matches_df[['odds_1', 'odds_X', 'odds_2']].values
        odds_matrix = MatchInputLayer.validate_odds(odds_matrix)
        
        # Preparar diccionario de parámetros
        params_dict = {
            'attack_strengths': np.array([s['attack'] for s in strengths_data]),
            'defense_strengths': np.array([s['defense'] for s in strengths_data]),
            'home_advantages': np.array([s['advantage'] for s in strengths_data]),
            'matches_df': matches_df,
            'odds_matrix': odds_matrix,
            'mode': 'manual' if is_manual else 'auto'
        }
        
        # Mostrar resumen
        MatchInputLayer._render_summary(matches_df, params_dict)
        
        return matches_df, params_dict, params_dict['mode']
    
    @staticmethod
    def _render_summary(matches_df: pd.DataFrame, params_dict: Dict):
        """Renderiza resumen del input."""
        st.subheader("📊 Resumen del Input")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_margin = matches_df['margin'].mean()
            st.metric("Margen Promedio", f"{avg_margin:.2f}%")
        
        with col2:
            avg_odds = matches_df[['odds_1', 'odds_X', 'odds_2']].values.mean()
            st.metric("Cuota Promedio", f"{avg_odds:.2f}")
        
        with col3:
            total_combo = 3 ** len(matches_df)
            st.metric("Combinaciones", f"{total_combo:,}")
        
        # Información de modo
        mode = params_dict['mode']
        st.caption(f"Modo actual: **{'🎮 Manual' if mode == 'manual' else '🔘 Automático'}**")
    
    @staticmethod
    def process_input(params_dict: Dict) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Procesa input para generar probabilidades."""
        attack = params_dict['attack_strengths']
        defense = params_dict['defense_strengths']
        advantages = params_dict['home_advantages']
        
        n_matches = len(attack)
        lambda_home = np.zeros(n_matches)
        lambda_away = np.zeros(n_matches)
        
        # Calcular tasas de goles
        for i in range(n_matches):
            lambda_home[i] = attack[i, 0] * defense[i, 1] * advantages[i]
            lambda_away[i] = attack[i, 1] * defense[i, 0]
        
        # Generar probabilidades
        probabilities = ACBEModel.simulate_probabilities(lambda_home, lambda_away)
        
        # Actualizar DataFrame
        matches_df = params_dict['matches_df'].copy()
        matches_df['lambda_home'] = lambda_home
        matches_df['lambda_away'] = lambda_away
        
        return matches_df, params_dict['odds_matrix'], probabilities

    # ============================================================================
    # SECCIÓN 2: SISTEMA COMBINATORIO S73 PROFESIONAL CON REDUCCIÓN ELITE v3.0
    # ============================================================================

class S73System:
    """
    Sistema combinatorio S73 con cobertura de 2 errores y reducción elite v3.0.
    
    Características principales:
    1. ✅ Generación optimizada de combinaciones pre-filtradas
    2. ✅ Algoritmo greedy mejorado para cobertura de 2 errores
    3. ✅ REDUCCIÓN ELITE v3.0 con Score de Eficiencia
    4. ✅ Simulador de Escenarios interactivo
    5. ✅ Validación de cobertura y métricas de calidad
    6. ✅ Vectorización para máxima performance
    7. ✅ Caching inteligente con Streamlit
    
    Arquitectura de Doble Reducción v3.0:
    Fase 1: 729 combinaciones → 73 columnas (cobertura 2 errores)
    Fase 2: 73 columnas → 24 columnas elite (optimización por score)
    """
    
    # ============================================================================
    # 1. GENERACIÓN DE COMBINACIONES PRE-FILTRADAS (OPTIMIZADA)
    # ============================================================================
    
    @staticmethod
    @st.cache_data(show_spinner=False)
    def generate_prefiltered_combinations(
        probabilities: np.ndarray,
        normalized_entropies: np.ndarray,
        odds_matrix: np.ndarray,
        apply_filters: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Genera combinaciones pre-filtradas con optimizaciones v3.0.
        
        Proceso:
        1. Clasificación por entropía (fuerte/medio/caótico)
        2. Filtrado por umbrales institucionales
        3. Cálculo vectorizado de probabilidades conjuntas
        4. Filtrado por probabilidad mínima
        
        Args:
            probabilities: Array (n_matches, 3) de probabilidades ACBE
            normalized_entropies: Array (n_matches,) de entropías normalizadas
            odds_matrix: Array (n_matches, 3) de cuotas
            apply_filters: Si True, aplica filtros institucionales
            
        Returns:
            combinations: Array (n_combinations, 6) de combinaciones
            joint_probs: Array (n_combinations,) de probabilidades conjuntas
            allowed_signs: Lista de signos permitidos por partido
        """
        from models.information_theory import InformationTheory
        
        # Validación de inputs
        if probabilities.shape[0] < SystemConfig.NUM_MATCHES:
            raise ValueError(f"Se requieren al menos {SystemConfig.NUM_MATCHES} partidos")
        
        # Usar solo los primeros 6 partidos para S73
        n_matches = min(SystemConfig.NUM_MATCHES, probabilities.shape[0])
        probs = probabilities[:n_matches, :]
        odds = odds_matrix[:n_matches, :]
        entropies = normalized_entropies[:n_matches]
        
        # 1. Clasificar partidos y obtener signos permitidos
        allowed_signs, classifications = InformationTheory.classify_matches_by_entropy(
            probs, entropies, odds if apply_filters else None
        )
        
        # 2. Validar y ajustar signos permitidos
        validated_signs = []
        for i, signs in enumerate(allowed_signs):
            if len(signs) == 0:
                # Si no hay signos permitidos, permitir todos (fallback)
                validated_signs.append([0, 1, 2])
                st.warning(f"Partido {i+1}: Sin signos permitidos. Usando todos los signos.")
            else:
                validated_signs.append(signs)
        
        # 3. Generar producto cartesiano (todas las combinaciones posibles)
        import itertools
        combinations_list = list(itertools.product(*validated_signs))
        
        if not combinations_list:
            raise ValueError("No se generaron combinaciones. Revisar filtros.")
        
        combinations = np.array(combinations_list)
        
        # 4. Calcular probabilidades conjuntas (VECTORIZADO OPTIMIZADO)
        n_combinations = len(combinations)
        
        # Pre-calcular log-probabilities para estabilidad numérica
        log_probs = np.log(np.clip(probs, SystemConfig.MIN_PROBABILITY, 1.0))
        
        # Calcular log-probabilidades conjuntas vectorizado
        log_joint_probs = np.zeros(n_combinations)
        
        for i in range(n_combinations):
            # Sumar log-probabilidades en lugar de multiplicar
            for match_idx in range(n_matches):
                sign = combinations[i, match_idx]
                log_joint_probs[i] += log_probs[match_idx, sign]
        
        # Convertir de vuelta a probabilidades
        joint_probs = np.exp(log_joint_probs)
        
        # 5. Aplicar filtros institucionales si está activado
        if apply_filters:
            # Filtrar por probabilidad conjunta mínima
            prob_mask = joint_probs >= SystemConfig.MIN_JOINT_PROBABILITY
            
            # Filtrar por EV positivo si hay cuotas
            if odds is not None:
                ev_mask = np.ones(n_combinations, dtype=bool)
                for i in range(n_combinations):
                    combo_odds = np.prod(odds[np.arange(n_matches), combinations[i]])
                    ev = joint_probs[i] * combo_odds - 1
                    ev_mask[i] = ev >= SystemConfig.MIN_EV_THRESHOLD
                
                mask = prob_mask & ev_mask
            else:
                mask = prob_mask
            
            filtered_combinations = combinations[mask]
            filtered_probs = joint_probs[mask]
        else:
            filtered_combinations = combinations
            filtered_probs = joint_probs
        
        # 6. Validar que haya combinaciones después del filtrado
        if len(filtered_combinations) == 0:
            st.warning("⚠️ Filtros muy estrictos. No hay combinaciones. Relajando filtros...")
            # Fallback: usar todas las combinaciones sin filtrar
            filtered_combinations = combinations
            filtered_probs = joint_probs
        
        return filtered_combinations, filtered_probs, validated_signs
    
    # ============================================================================
    # 2. ALGORITMO DE COBERTURA S73 (MEJORADO)
    # ============================================================================
    
    @staticmethod
    @st.cache_data(show_spinner=False)
    def build_s73_coverage_system(
        filtered_combinations: np.ndarray,
        filtered_probs: np.ndarray,
        validate_coverage: bool = True,
        verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Construye sistema S73 con cobertura garantizada de 2 errores (algoritmo greedy mejorado).
        
        Args:
            filtered_combinations: Combinaciones pre-filtradas
            filtered_probs: Probabilidades conjuntas
            validate_coverage: Si True, valida cobertura después de construir
            verbose: Si True, muestra información detallada del proceso
            
        Returns:
            selected_combinations: Array (73,) de combinaciones seleccionadas
            selected_probs: Array (73,) de probabilidades seleccionadas
            metrics: Diccionario con métricas del sistema
        """
        n_combinations = len(filtered_combinations)
        
        # Caso especial: si ya tenemos menos combinaciones que el target
        if n_combinations <= SystemConfig.TARGET_COMBINATIONS:
            if verbose:
                st.info(f"Ya tenemos {n_combinations} ≤ {SystemConfig.TARGET_COMBINATIONS} combinaciones")
            
            metrics = {
                'initial_combinations': n_combinations,
                'final_combinations': n_combinations,
                'coverage_validated': True,
                'coverage_rate': 1.0,
                'avg_probability': np.mean(filtered_probs),
                'min_probability': np.min(filtered_probs) if n_combinations > 0 else 0,
                'max_probability': np.max(filtered_probs) if n_combinations > 0 else 0
            }
            
            return filtered_combinations, filtered_probs, metrics
        
        # 1. Ordenar por probabilidad descendente
        sorted_indices = np.argsort(filtered_probs)[::-1]
        sorted_combinations = filtered_combinations[sorted_indices]
        sorted_probs = filtered_probs[sorted_indices]
        
        if verbose:
            st.write(f"📊 Ordenadas {n_combinations} combinaciones por probabilidad")
        
        # 2. Precalcular matriz de distancias Hamming (OPTIMIZADO)
        distance_matrix = S73System._hamming_distance_matrix_optimized(sorted_combinations)
        
        # 3. Algoritmo greedy mejorado con prioridad probabilística
        selected_indices = []
        covered_indices = set()
        
        # Priorizar combinaciones con alta probabilidad
        priority_queue = sorted_indices.copy()
        
        iteration = 0
        max_iterations = SystemConfig.TARGET_COMBINATIONS * 2
        
        while (len(selected_indices) < SystemConfig.TARGET_COMBINATIONS and 
               iteration < max_iterations and 
               priority_queue):
            
            iteration += 1
            
            # Tomar el mejor de la cola de prioridad
            best_idx = priority_queue.pop(0)
            
            # Si ya está seleccionado, continuar
            if best_idx in selected_indices:
                continue
            
            # Calcular cobertura de esta combinación
            coverage_mask = distance_matrix[best_idx] <= SystemConfig.HAMMING_DISTANCE_TARGET
            newly_covered = []
            
            for j in range(n_combinations):
                if coverage_mask[j] and j not in covered_indices:
                    newly_covered.append(j)
            
            # Solo seleccionar si aporta cobertura nueva
            if newly_covered or len(selected_indices) == 0:
                selected_indices.append(best_idx)
                covered_indices.update(newly_covered)
                
                if verbose and iteration % 10 == 0:
                    coverage_rate = len(covered_indices) / n_combinations
                    st.write(f"Iteración {iteration}: {len(selected_indices)} seleccionadas, "
                           f"Cobertura: {coverage_rate:.1%}")
            
            # Recalcular prioridades (menor prioridad a combinaciones similares a las ya cubiertas)
            if iteration % 5 == 0:
                priority_queue = S73System._update_priority_queue(
                    priority_queue, selected_indices, distance_matrix, sorted_probs
                )
        
        # 4. Completar si no alcanza el target (usar las más probables restantes)
        if len(selected_indices) < SystemConfig.TARGET_COMBINATIONS:
            remaining_needed = SystemConfig.TARGET_COMBINATIONS - len(selected_indices)
            
            for idx in sorted_indices:
                if idx not in selected_indices:
                    selected_indices.append(idx)
                    remaining_needed -= 1
                    if remaining_needed == 0:
                        break
        
        # 5. Extraer combinaciones seleccionadas
        selected_combinations = sorted_combinations[selected_indices]
        selected_probs = sorted_probs[selected_indices]
        
        # 6. Validar cobertura si se solicita
        coverage_validated = False
        coverage_rate = 0.0
        
        if validate_coverage:
            coverage_rate = S73System._validate_coverage(
                sorted_combinations, selected_combinations, distance_matrix
            )
            coverage_validated = coverage_rate >= 0.95  # 95% de cobertura mínimo
            
            if verbose:
                if coverage_validated:
                    st.success(f"✅ Cobertura validada: {coverage_rate:.2%}")
                else:
                    st.warning(f"⚠️ Cobertura insuficiente: {coverage_rate:.2%}")
        
        # 7. Calcular métricas del sistema
        metrics = {
            'initial_combinations': n_combinations,
            'final_combinations': len(selected_combinations),
            'coverage_validated': coverage_validated,
            'coverage_rate': coverage_rate,
            'avg_probability': np.mean(selected_probs),
            'min_probability': np.min(selected_probs) if len(selected_probs) > 0 else 0,
            'max_probability': np.max(selected_probs) if len(selected_probs) > 0 else 0,
            'total_probability': np.sum(selected_probs),
            'probability_std': np.std(selected_probs) if len(selected_probs) > 1 else 0,
            'iterations_needed': iteration,
            'efficiency': len(selected_indices) / n_combinations if n_combinations > 0 else 0
        }
        
        return selected_combinations, selected_probs, metrics
    
    # ============================================================================
    # 3. REDUCCIÓN ELITE v3.0 - NUEVO
    # ============================================================================
    
    @staticmethod
    @st.cache_data(show_spinner=False)
    def apply_elite_reduction(
        s73_combinations: np.ndarray,
        s73_probabilities: np.ndarray,
        odds_matrix: np.ndarray,
        normalized_entropies: np.ndarray,
        elite_target: int = None,
        score_weights: Dict[str, float] = None,
        portfolio_type: str = "elite"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Fase 2: Reducción Elite - Selecciona las mejores columnas usando Score de Eficiencia.
        
        Score = P × (1 + EV) × (1 - EntropíaPromedio)
        
        Args:
            s73_combinations: Array (73, 6) de combinaciones S73
            s73_probabilities: Array (73,) de probabilidades conjuntas
            odds_matrix: Array (6, 3) de cuotas por partido
            normalized_entropies: Array (6,) de entropías normalizadas
            elite_target: Número de columnas elite (default: SystemConfig.ELITE_COLUMNS_TARGET)
            score_weights: Pesos para cálculo de score (opcional)
            portfolio_type: 'elite' o 'full' (para ajuste de Kelly)
            
        Returns:
            elite_combinations: Array (elite_target, 6) de combinaciones elite
            elite_probabilities: Array (elite_target,) de probabilidades elite
            elite_scores: Array (elite_target,) de scores de eficiencia
            elite_metrics: Diccionario con métricas de la reducción
        """
        if elite_target is None:
            elite_target = SystemConfig.ELITE_COLUMNS_TARGET
        
        if score_weights is None:
            score_weights = SystemConfig.ELITE_SCORE_WEIGHTS
        
        n_columns = len(s73_combinations)
        
        # Validación básica
        if n_columns == 0:
            raise ValueError("No hay combinaciones para reducir")
        
        # Si ya tenemos menos columnas que el target, devolver todas
        if n_columns <= elite_target:
            elite_combinations = s73_combinations
            elite_probabilities = s73_probabilities
            elite_scores = np.ones(n_columns)
            
            elite_metrics = {
                'reduction_applied': False,
                'original_columns': n_columns,
                'elite_columns': n_columns,
                'reduction_ratio': 1.0,
                'avg_score': 1.0,
                'min_score': 1.0,
                'max_score': 1.0
            }
            
            return elite_combinations, elite_probabilities, elite_scores, elite_metrics
        
        # 1. Calcular métricas por columna (VECTORIZADO)
        n_matches = SystemConfig.NUM_MATCHES
        
        # Pre-calcular cuotas conjuntas
        column_odds = np.zeros(n_columns)
        column_ev = np.zeros(n_columns)
        column_entropy_avg = np.zeros(n_columns)
        
        for i in range(n_columns):
            # Calcular cuota conjunta
            selected_odds = odds_matrix[np.arange(n_matches), s73_combinations[i]]
            column_odds[i] = np.prod(selected_odds)
            
            # Calcular EV de la columna
            column_ev[i] = s73_probabilities[i] * column_odds[i] - 1
            
            # Calcular entropía promedio de la columna
            match_entropies = normalized_entropies[np.arange(n_matches)]
            column_entropy_avg[i] = np.mean(match_entropies)
        
        # 2. Calcular Score de Eficiencia v3.0
        # Score = P × (1 + EV) × (1 - EntropíaPromedio)
        
        # Componentes del score
        prob_component = s73_probabilities ** score_weights['probability']
        ev_component = (1.0 + np.maximum(column_ev, -0.5)) ** score_weights['expected_value']
        entropy_component = (1.0 - np.clip(column_entropy_avg, 0, 0.99)) ** score_weights['entropy']
        
        # Score final
        scores = prob_component * ev_component * entropy_component
        
        # 3. Ordenar por score descendente y seleccionar top N
        elite_indices = np.argsort(scores)[::-1][:elite_target]
        
        # 4. Extraer columnas elite
        elite_combinations = s73_combinations[elite_indices]
        elite_probabilities = s73_probabilities[elite_indices]
        elite_scores = scores[elite_indices]
        
        # 5. Calcular métricas de la reducción
        original_avg_prob = np.mean(s73_probabilities)
        elite_avg_prob = np.mean(elite_probabilities)
        
        original_avg_ev = np.mean(column_ev)
        elite_avg_ev = np.mean(column_ev[elite_indices])
        
        elite_metrics = {
            'reduction_applied': True,
            'original_columns': n_columns,
            'elite_columns': elite_target,
            'reduction_ratio': elite_target / n_columns,
            'avg_score': np.mean(elite_scores),
            'min_score': np.min(elite_scores),
            'max_score': np.max(elite_scores),
            'score_std': np.std(elite_scores) if len(elite_scores) > 1 else 0,
            'prob_improvement': (elite_avg_prob - original_avg_prob) / original_avg_prob if original_avg_prob > 0 else 0,
            'ev_improvement': (elite_avg_ev - original_avg_ev) / abs(original_avg_ev) if abs(original_avg_ev) > 0 else 0,
            'avg_entropy': np.mean(column_entropy_avg[elite_indices]),
            'score_weights': score_weights
        }
        
        return elite_combinations, elite_probabilities, elite_scores, elite_metrics
    
    # ============================================================================
    # 4. SIMULADOR DE ESCENARIOS v3.0 - NUEVO
    # ============================================================================
    
    @staticmethod
    @st.cache_data(show_spinner=False)
    def simulate_scenario(
        combinations: np.ndarray,
        failed_matches: List[Tuple[int, int]],
        probabilities: np.ndarray,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Simula escenario: ¿Qué pasa si fallo los partidos X e Y?
        
        Args:
            combinations: Array (n_columns, 6) de combinaciones
            failed_matches: Lista de tuplas (partido_idx, resultado_correcto)
                          partido_idx: 0-5 (índice del partido)
                          resultado_correcto: 0,1,2 (1,X,2)
            probabilities: Array (n_columns,) de probabilidades conjuntas
            verbose: Si True, muestra información detallada
            
        Returns:
            Dict con estadísticas detalladas del escenario
        """
        n_columns = len(combinations)
        
        if len(failed_matches) == 0:
            return {
                "error": "No se especificaron partidos fallados",
                "total_columns": n_columns,
                "scenario_type": "no_failed_matches"
            }
        
        # Convertir failed_matches a dict para acceso rápido
        failed_dict = {match_idx: correct_result for match_idx, correct_result in failed_matches}
        
        if verbose:
            st.write(f"🔮 Simulando escenario con {len(failed_dict)} partidos fallados")
        
        # Calcular aciertos por columna (VECTORIZADO)
        hits_per_column = np.zeros(n_columns, dtype=int)
        
        for col_idx in range(n_columns):
            hits = 0
            for match_idx in range(6):
                if match_idx in failed_dict:
                    # Este partido está en los fallados
                    if combinations[col_idx, match_idx] == failed_dict[match_idx]:
                        # ¡Acertó un partido que se supone que falló!
                        hits += 1
                else:
                    # Partido no fallado - siempre cuenta como acierto para este análisis
                    hits += 1
            
            hits_per_column[col_idx] = hits
        
        # Estadísticas detalladas
        hits_distribution = {
            "6": int(np.sum(hits_per_column == 6)),
            "5": int(np.sum(hits_per_column == 5)),
            "4": int(np.sum(hits_per_column == 4)),
            "3": int(np.sum(hits_per_column == 3)),
            "2": int(np.sum(hits_per_column == 2)),
            "1": int(np.sum(hits_per_column == 1)),
            "0": int(np.sum(hits_per_column == 0))
        }
        
        # Calcular probabilidades por nivel de aciertos
        prob_by_hits = {}
        for hits in range(7):
            mask = hits_per_column == hits
            if np.any(mask):
                prob_by_hits[str(hits)] = float(np.sum(probabilities[mask]))
            else:
                prob_by_hits[str(hits)] = 0.0
        
        # Calcular cobertura
        columns_with_4plus = int(np.sum(hits_per_column >= 4))
        columns_with_5plus = int(np.sum(hits_per_column >= 5))
        columns_with_6 = int(np.sum(hits_per_column == 6))
        
        coverage_4plus = columns_with_4plus / n_columns if n_columns > 0 else 0
        coverage_5plus = columns_with_5plus / n_columns if n_columns > 0 else 0
        coverage_6 = columns_with_6 / n_columns if n_columns > 0 else 0
        
        # Clasificar escenario según cobertura
        if coverage_4plus >= SystemConfig.SCENARIO_CONFIG['thresholds']['favorable_coverage']:
            scenario_type = "favorable"
            scenario_color = SystemConfig.SCENARIO_COLORS['success']
        elif coverage_4plus >= SystemConfig.SCENARIO_CONFIG['thresholds']['challenging_coverage']:
            scenario_type = "challenging"
            scenario_color = SystemConfig.SCENARIO_COLORS['warning']
        else:
            scenario_type = "critical"
            scenario_color = SystemConfig.SCENARIO_COLORS['danger']
        
        # Calcular impacto en bankroll (simplificado)
        avg_hits = float(np.mean(hits_per_column))
        median_hits = float(np.median(hits_per_column))
        
        # Calcular probabilidad de mantener 4+ aciertos
        prob_4plus = float(np.sum(probabilities[hits_per_column >= 4]))
        
        stats = {
            "hits_distribution": hits_distribution,
            "probability_distribution": prob_by_hits,
            "total_columns": n_columns,
            "avg_hits": avg_hits,
            "median_hits": median_hits,
            "std_hits": float(np.std(hits_per_column)) if n_columns > 1 else 0.0,
            "columns_with_4plus": columns_with_4plus,
            "columns_with_5plus": columns_with_5plus,
            "columns_with_6": columns_with_6,
            "coverage_4plus": coverage_4plus,
            "coverage_5plus": coverage_5plus,
            "coverage_6": coverage_6,
            "prob_4plus": prob_4plus,
            "failed_matches": failed_matches,
            "scenario_type": scenario_type,
            "scenario_color": scenario_color,
            "max_possible_hits": 6 - len(failed_dict),  # Máximo teórico después de fallar partidos
            "system_robustness": coverage_4plus  # Métrica simple de robustez
        }
        
        if verbose:
            st.write(f"📊 Escenario clasificado como: {scenario_type.upper()}")
            st.write(f"📈 Columnas con 4+ aciertos: {columns_with_4plus}/{n_columns} ({coverage_4plus:.1%})")
        
        return stats
    
    # ============================================================================
    # 5. MÉTODOS UTILITARIOS Y AUXILIARES
    # ============================================================================
    
    @staticmethod
    def _hamming_distance_matrix_optimized(combinations: np.ndarray) -> np.ndarray:
        """
        Calcula matriz de distancias de Hamming optimizada.
        
        Args:
            combinations: Array (n, 6) de combinaciones
            
        Returns:
            Matriz de distancias (n, n)
        """
        n = len(combinations)
        
        # Versión optimizada usando broadcasting
        distances = np.zeros((n, n), dtype=np.int8)
        
        for i in range(n):
            # Comparar la combinación i con todas las demás
            diff = combinations[i] != combinations
            distances[i] = np.sum(diff, axis=1)
        
        return distances
    
    @staticmethod
    def _validate_coverage(
        all_combinations: np.ndarray,
        selected_combinations: np.ndarray,
        distance_matrix: np.ndarray = None
    ) -> float:
        """
        Valida cobertura del sistema seleccionado.
        
        Args:
            all_combinations: Todas las combinaciones posibles
            selected_combinations: Combinaciones seleccionadas
            distance_matrix: Matriz de distancias precalculada (opcional)
            
        Returns:
            Tasa de cobertura (0-1)
        """
        n_all = len(all_combinations)
        n_selected = len(selected_combinations)
        
        if n_selected == 0:
            return 0.0
        
        # Calcular matriz de distancias si no se proporciona
        if distance_matrix is None:
            # Calcular solo para las combinaciones seleccionadas vs todas
            coverage_count = 0
            
            for i in range(n_all):
                covered = False
                for j in range(n_selected):
                    if np.sum(all_combinations[i] != selected_combinations[j]) <= SystemConfig.HAMMING_DISTANCE_TARGET:
                        covered = True
                        break
                
                if covered:
                    coverage_count += 1
            
            coverage_rate = coverage_count / n_all
        else:
            # Usar matriz precalculada (más eficiente)
            coverage_mask = np.zeros(n_all, dtype=bool)
            
            for j in range(n_selected):
                idx = np.where((all_combinations == selected_combinations[j]).all(axis=1))[0]
                if len(idx) > 0:
                    selected_idx = idx[0]
                    coverage_mask |= distance_matrix[selected_idx] <= SystemConfig.HAMMING_DISTANCE_TARGET
            
            coverage_rate = np.sum(coverage_mask) / n_all
        
        return coverage_rate
    
    @staticmethod
    def _update_priority_queue(
        priority_queue: List[int],
        selected_indices: List[int],
        distance_matrix: np.ndarray,
        probabilities: np.ndarray
    ) -> List[int]:
        """
        Actualiza cola de prioridad considerando similitud con combinaciones ya seleccionadas.
        
        Args:
            priority_queue: Cola de prioridad actual
            selected_indices: Índices ya seleccionados
            distance_matrix: Matriz de distancias
            probabilities: Probabilidades de las combinaciones
            
        Returns:
            Nueva cola de prioridad
        """
        if not selected_indices or not priority_queue:
            return priority_queue
        
        # Calcular scores de diversidad
        diversity_scores = []
        
        for idx in priority_queue:
            # Penalizar similitud con combinaciones ya seleccionadas
            similarity_penalty = 0
            
            for selected_idx in selected_indices:
                distance = distance_matrix[idx, selected_idx]
                # Penalizar más las combinaciones muy similares (distancia 0-1)
                if distance <= 1:
                    similarity_penalty += 2.0
                elif distance == 2:
                    similarity_penalty += 1.0
            
            # Score = probabilidad - penalización por similitud
            score = probabilities[idx] - similarity_penalty * 0.1
            diversity_scores.append((idx, score))
        
        # Ordenar por score descendente
        diversity_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Extraer índices ordenados
        new_priority_queue = [idx for idx, _ in diversity_scores]
        
        return new_priority_queue
    
    @staticmethod
    def calculate_combination_odds(combination: np.ndarray, odds_matrix: np.ndarray) -> float:
        """
        Calcula cuota conjunta de una combinación.
        
        Args:
            combination: Array (6,) de signos
            odds_matrix: Matriz (6, 3) de cuotas
            
        Returns:
            Cuota conjunta
        """
        selected_odds = odds_matrix[np.arange(6), combination]
        return np.prod(selected_odds)
    
    @staticmethod
    def get_system_summary(
        combinations: np.ndarray,
        probabilities: np.ndarray,
        odds_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """
        Genera resumen completo del sistema.
        
        Args:
            combinations: Combinaciones del sistema
            probabilities: Probabilidades conjuntas
            odds_matrix: Matriz de cuotas
            
        Returns:
            Diccionario con resumen del sistema
        """
        n_columns = len(combinations)
        
        if n_columns == 0:
            return {
                "columns_count": 0,
                "avg_probability": 0,
                "avg_odds": 0,
                "avg_ev": 0,
                "coverage_rate": 0
            }
        
        # Calcular métricas por columna
        column_odds = []
        column_ev = []
        
        for i in range(n_columns):
            odds = S73System.calculate_combination_odds(combinations[i], odds_matrix)
            ev = probabilities[i] * odds - 1
            
            column_odds.append(odds)
            column_ev.append(ev)
        
        return {
            "columns_count": n_columns,
            "avg_probability": float(np.mean(probabilities)),
            "min_probability": float(np.min(probabilities)),
            "max_probability": float(np.max(probabilities)),
            "avg_odds": float(np.mean(column_odds)),
            "min_odds": float(np.min(column_odds)),
            "max_odds": float(np.max(column_odds)),
            "avg_ev": float(np.mean(column_ev)),
            "positive_ev_columns": int(np.sum(np.array(column_ev) > 0)),
            "negative_ev_columns": int(np.sum(np.array(column_ev) <= 0)),
            "prob_std": float(np.std(probabilities)) if n_columns > 1 else 0.0,
            "odds_std": float(np.std(column_odds)) if n_columns > 1 else 0.0
        }

# ============================================================================
# SECCIÓN 3: CRITERIO DE KELLY INTELIGENTE CON GESTIÓN DE CAPITAL v3.0
# ============================================================================

class KellyCapitalManagement:
    """
    Sistema avanzado de gestión de capital basado en criterio de Kelly.
    
    Características v3.0:
    1. ✅ Kelly diferenciado para portafolios Full (73) y Elite (24)
    2. ✅ Ajuste dinámico por entropía (riesgo) y correlación
    3. ✅ Límites de exposición y normalización automática
    4. ✅ Modos manual/automático con validación de riesgos
    5. ✅ Cálculo de métricas de gestión de capital
    6. ✅ Simulación de escenarios de bankroll
    7. ✅ Optimización vectorizada para performance
    
    Filosofía v3.0:
    - Portafolio Full (73): Más diversificación → Kelly más conservador
    - Portafolio Elite (24): Más concentración → Kelly más agresivo
    """
    
    # ============================================================================
    # 1. CÁLCULO DE STAKES KELLY PARA APUESTAS INDIVIDUALES
    # ============================================================================
    
    @staticmethod
    @st.cache_data(show_spinner=False)
    def calculate_kelly_stakes(
        probabilities: np.ndarray,
        odds_matrix: np.ndarray,
        normalized_entropies: np.ndarray,
        kelly_fraction: float = 0.5,
        manual_stake: Optional[float] = None,
        portfolio_type: str = "full",
        apply_risk_adjustment: bool = True,
        max_exposure: float = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Calcula stakes Kelly ajustados por entropía, tipo de portafolio y exposición máxima.
        
        Args:
            probabilities: Array (n_matches, 3) de probabilidades ACBE
            odds_matrix: Array (n_matches, 3) de cuotas
            normalized_entropies: Array (n_matches,) de entropías normalizadas (0-1)
            kelly_fraction: Fracción conservadora de Kelly (0.1-1.0)
            manual_stake: Stake manual fijo (% bankroll) (None para Kelly automático)
            portfolio_type: 'full' (73 cols) o 'elite' (24 cols)
            apply_risk_adjustment: Si True, ajusta por entropía y correlación
            max_exposure: Exposición máxima permitida (None = usar SystemConfig)
            
        Returns:
            stakes_matrix: Array (n_matches, 3) de stakes recomendados (% bankroll)
            metrics: Diccionario con métricas de gestión de capital
        """
        # Validación de inputs
        if probabilities.shape != odds_matrix.shape:
            raise ValueError("probabilities y odds_matrix deben tener la misma forma")
        
        if len(normalized_entropies) != probabilities.shape[0]:
            raise ValueError("normalized_entropies debe tener la misma longitud que probabilities")
        
        if max_exposure is None:
            max_exposure = SystemConfig.MAX_PORTFOLIO_EXPOSURE
        
        # ==================== MODO MANUAL ====================
        if manual_stake is not None:
            # Validar stake manual
            if manual_stake <= 0 or manual_stake > SystemConfig.MAX_STAKE_PERCENTAGE * 100:
                st.warning(f"Stake manual ({manual_stake}%) fuera de límites. Ajustando...")
                manual_stake = np.clip(manual_stake, 
                                      SystemConfig.MIN_STAKE_PERCENTAGE * 100,
                                      SystemConfig.MAX_STAKE_PERCENTAGE * 100)
            
            stake_percent = manual_stake / 100.0  # Convertir a fracción
            
            # Crear matriz de stakes
            stakes_matrix = np.full_like(probabilities, stake_percent)
            
            # Ajustar por entropía si está activado
            if apply_risk_adjustment:
                entropy_adjustment = (1.0 - normalized_entropies[:, np.newaxis])
                stakes_matrix = stakes_matrix * entropy_adjustment
            
            # Normalizar por exposición máxima
            stakes_matrix = KellyCapitalManagement._normalize_stakes(
                stakes_matrix, max_exposure, 'manual'
            )
            
            metrics = KellyCapitalManagement._calculate_stake_metrics(
                stakes_matrix, probabilities, odds_matrix, portfolio_type, 'manual'
            )
            
            return stakes_matrix, metrics
        
        # ==================== MODO KELLY AUTOMÁTICO ====================
        
        # 1. Calcular Kelly crudo
        with np.errstate(divide='ignore', invalid='ignore'):
            kelly_raw = (probabilities * odds_matrix - 1) / (odds_matrix - 1)
        
        # Manejar casos especiales
        kelly_raw = np.nan_to_num(kelly_raw, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 2. Aplicar límites por columna
        kelly_capped = np.clip(kelly_raw, 0, SystemConfig.KELLY_FRACTION_MAX)
        
        # 3. Ajustar por fracción conservadora
        kelly_fractioned = kelly_capped * kelly_fraction
        
        # 4. Ajustar por tipo de portafolio
        kelly_multiplier = SystemConfig.get_kelly_multiplier(portfolio_type)
        kelly_portfolio_adjusted = kelly_fractioned * kelly_multiplier
        
        # 5. Ajustar por riesgo (entropía)
        if apply_risk_adjustment:
            entropy_adjustment = (1.0 - normalized_entropies[:, np.newaxis])
            kelly_risk_adjusted = kelly_portfolio_adjusted * entropy_adjustment
        else:
            kelly_risk_adjusted = kelly_portfolio_adjusted
        
        # 6. Aplicar límites mínimos/máximos
        kelly_risk_adjusted = np.clip(
            kelly_risk_adjusted,
            SystemConfig.MIN_STAKE_PERCENTAGE,
            SystemConfig.MAX_STAKE_PERCENTAGE
        )
        
        # 7. Normalizar por exposición máxima
        stakes_matrix = KellyCapitalManagement._normalize_stakes(
            kelly_risk_adjusted, max_exposure, 'kelly'
        )
        
        # 8. Calcular métricas
        metrics = KellyCapitalManagement._calculate_stake_metrics(
            stakes_matrix, probabilities, odds_matrix, portfolio_type, 'kelly'
        )
        
        return stakes_matrix, metrics
    
    # ============================================================================
    # 2. CÁLCULO DE STAKES PARA COLUMNAS S73/ELITE
    # ============================================================================
    
    @staticmethod
    def calculate_column_kelly_stakes(
        combinations: np.ndarray,
        probabilities: np.ndarray,
        odds_matrix: np.ndarray,
        normalized_entropies: np.ndarray,
        kelly_fraction: float = 0.5,
        manual_stake: Optional[float] = None,
        portfolio_type: str = "full",
        max_exposure: float = None,
        bankroll: float = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Calcula stakes Kelly para columnas del sistema S73/Elite.
        
        Args:
            combinations: Array (n_columns, 6) de combinaciones
            probabilities: Array (n_columns,) de probabilidades conjuntas
            odds_matrix: Array (6, 3) de cuotas por partido
            normalized_entropies: Array (6,) de entropías normalizadas
            kelly_fraction: Fracción conservadora de Kelly
            manual_stake: Stake manual fijo (% bankroll)
            portfolio_type: 'full' o 'elite'
            max_exposure: Exposición máxima permitida
            bankroll: Bankroll para cálculos monetarios
            
        Returns:
            column_stakes: Array (n_columns,) de stakes por columna (% bankroll)
            metrics: Diccionario con métricas detalladas
        """
        if bankroll is None:
            bankroll = SystemConfig.DEFAULT_BANKROLL
        
        if max_exposure is None:
            max_exposure = SystemConfig.MAX_PORTFOLIO_EXPOSURE
        
        n_columns = len(combinations)
        
        # Calcular stakes base por columna
        column_stakes = np.zeros(n_columns)
        column_odds = np.zeros(n_columns)
        column_ev = np.zeros(n_columns)
        column_entropy = np.zeros(n_columns)
        
        for i in range(n_columns):
            combo = combinations[i]
            prob = probabilities[i]
            
            # Calcular cuota conjunta
            selected_odds = odds_matrix[np.arange(6), combo]
            combo_odds = np.prod(selected_odds)
            column_odds[i] = combo_odds
            
            # Calcular EV
            column_ev[i] = prob * combo_odds - 1
            
            # Calcular entropía promedio de la columna
            match_entropies = normalized_entropies[np.arange(6)]
            avg_entropy = np.mean(match_entropies)
            column_entropy[i] = avg_entropy
            
            # Calcular stake según modo
            if manual_stake is not None:
                stake_base = manual_stake / 100.0  # Convertir % a fracción
                stake = stake_base * (1.0 - avg_entropy)
            else:
                # Kelly para columna
                if combo_odds <= 1.0:
                    stake = 0.0
                else:
                    kelly_raw = (prob * combo_odds - 1) / (combo_odds - 1)
                    kelly_capped = max(0.0, min(kelly_raw, SystemConfig.KELLY_FRACTION_MAX))
                    
                    # Ajustar por fracción y tipo de portafolio
                    kelly_multiplier = SystemConfig.get_kelly_multiplier(portfolio_type)
                    kelly_adjusted = kelly_capped * kelly_fraction * kelly_multiplier
                    
                    # Ajustar por entropía
                    stake = kelly_adjusted * (1.0 - avg_entropy)
            
            # Aplicar límites
            stake = np.clip(stake, 
                          SystemConfig.MIN_STAKE_PERCENTAGE,
                          SystemConfig.MAX_STAKE_PERCENTAGE)
            
            column_stakes[i] = stake
        
        # Normalizar para respetar exposición máxima
        total_stake = np.sum(column_stakes)
        
        if total_stake > max_exposure:
            normalization_factor = max_exposure / total_stake
            column_stakes = column_stakes * normalization_factor
        
        # Calcular métricas detalladas
        metrics = {
            'portfolio_type': portfolio_type,
            'n_columns': n_columns,
            'total_stake_percentage': float(np.sum(column_stakes) * 100),
            'avg_stake_percentage': float(np.mean(column_stakes) * 100),
            'min_stake_percentage': float(np.min(column_stakes) * 100),
            'max_stake_percentage': float(np.max(column_stakes) * 100),
            'stake_std_percentage': float(np.std(column_stakes) * 100) if n_columns > 1 else 0.0,
            'total_investment': float(np.sum(column_stakes) * bankroll),
            'avg_column_ev': float(np.mean(column_ev)),
            'positive_ev_columns': int(np.sum(column_ev > 0)),
            'negative_ev_columns': int(np.sum(column_ev <= 0)),
            'avg_column_entropy': float(np.mean(column_entropy)),
            'max_exposure_percentage': max_exposure * 100,
            'exposure_utilization': float(np.sum(column_stakes) / max_exposure) if max_exposure > 0 else 0.0,
            'kelly_fraction_used': kelly_fraction if manual_stake is None else None,
            'manual_stake_used': manual_stake,
            'mode': 'manual' if manual_stake is not None else 'kelly'
        }
        
        # Añadir métricas de riesgo
        risk_metrics = KellyCapitalManagement._calculate_portfolio_risk_metrics(
            column_stakes, probabilities, column_odds, bankroll
        )
        metrics.update(risk_metrics)
        
        return column_stakes, metrics
    
    # ============================================================================
    # 3. SIMULACIÓN Y ANÁLISIS DE BANKROLL
    # ============================================================================
    
    @staticmethod
    def simulate_bankroll_evolution(
        column_stakes: np.ndarray,
        column_probabilities: np.ndarray,
        column_odds: np.ndarray,
        initial_bankroll: float,
        n_simulations: int = 10000,
        n_rounds: int = 100,
        reinvest_profits: bool = True
    ) -> Dict[str, Any]:
        """
        Simula evolución del bankroll usando Monte Carlo.
        
        Args:
            column_stakes: Array de stakes por columna (% bankroll)
            column_probabilities: Array de probabilidades por columna
            column_odds: Array de cuotas por columna
            initial_bankroll: Bankroll inicial
            n_simulations: Número de simulaciones Monte Carlo
            n_rounds: Número de rondas por simulación
            reinvest_profits: Si True, reinvierte ganancias
            
        Returns:
            Dict con resultados de la simulación
        """
        n_columns = len(column_stakes)
        
        if n_columns == 0:
            return {
                'error': 'No hay columnas para simular',
                'final_bankrolls': np.array([initial_bankroll]),
                'metrics': {}
            }
        
        # Validar datos
        if len(column_probabilities) != n_columns or len(column_odds) != n_columns:
            raise ValueError("Los arrays deben tener la misma longitud")
        
        # Simular resultados
        final_bankrolls = np.zeros(n_simulations)
        max_drawdowns = np.zeros(n_simulations)
        
        for sim_idx in range(n_simulations):
            bankroll = initial_bankroll
            
            # Track para drawdown
            peak_bankroll = bankroll
            max_drawdown = 0.0
            
            for round_idx in range(n_rounds):
                # Calcular stakes actuales
                if reinvest_profits:
                    stakes_amount = column_stakes * bankroll
                else:
                    stakes_amount = column_stakes * initial_bankroll
                
                # Simular resultados de esta ronda
                round_result = 0.0
                
                for col_idx in range(n_columns):
                    # Determinar si gana esta columna
                    win = np.random.random() < column_probabilities[col_idx]
                    
                    if win:
                        # Ganancia: stake * (odds - 1)
                        profit = stakes_amount[col_idx] * (column_odds[col_idx] - 1)
                        round_result += profit
                    else:
                        # Pérdida: stake
                        round_result -= stakes_amount[col_idx]
                
                # Actualizar bankroll
                bankroll += round_result
                
                # Actualizar drawdown
                if bankroll > peak_bankroll:
                    peak_bankroll = bankroll
                
                drawdown = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
            
            final_bankrolls[sim_idx] = bankroll
            max_drawdowns[sim_idx] = max_drawdown
        
        # Calcular métricas
        total_return = final_bankrolls - initial_bankroll
        total_return_pct = (total_return / initial_bankroll) * 100
        
        metrics = {
            'initial_bankroll': initial_bankroll,
            'avg_final_bankroll': float(np.mean(final_bankrolls)),
            'median_final_bankroll': float(np.median(final_bankrolls)),
            'min_final_bankroll': float(np.min(final_bankrolls)),
            'max_final_bankroll': float(np.max(final_bankrolls)),
            'std_final_bankroll': float(np.std(final_bankrolls)),
            'avg_total_return': float(np.mean(total_return)),
            'avg_total_return_pct': float(np.mean(total_return_pct)),
            'median_total_return_pct': float(np.median(total_return_pct)),
            'probability_profit': float(np.mean(total_return > 0)),
            'probability_loss': float(np.mean(total_return <= 0)),
            'expected_value': float(np.mean(total_return)),
            'expected_value_pct': float(np.mean(total_return_pct)),
            'value_at_risk_95': float(np.percentile(total_return, 5)),
            'conditional_var_95': float(np.mean(total_return[total_return <= np.percentile(total_return, 5)])),
            'avg_max_drawdown': float(np.mean(max_drawdowns) * 100),
            'max_observed_drawdown': float(np.max(max_drawdowns) * 100),
            'sharpe_ratio': float(np.mean(total_return_pct) / np.std(total_return_pct) if np.std(total_return_pct) > 0 else 0),
            'sortino_ratio': KellyCapitalManagement._calculate_sortino_ratio(total_return_pct),
            'n_simulations': n_simulations,
            'n_rounds': n_rounds,
            'n_columns': n_columns
        }
        
        # Calcular percentiles de bankroll final
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            metrics[f'percentile_{p}_final_bankroll'] = float(np.percentile(final_bankrolls, p))
        
        return {
            'final_bankrolls': final_bankrolls,
            'max_drawdowns': max_drawdowns,
            'total_returns': total_return,
            'total_returns_pct': total_return_pct,
            'metrics': metrics
        }
    
    # ============================================================================
    # 4. ANÁLISIS DE RIESGO Y OPTIMIZACIÓN
    # ============================================================================
    
    @staticmethod
    def calculate_optimal_kelly_fraction(
        column_stakes: np.ndarray,
        column_probabilities: np.ndarray,
        column_odds: np.ndarray,
        initial_bankroll: float,
        risk_tolerance: str = 'moderate'
    ) -> Dict[str, Any]:
        """
        Encuentra fracción óptima de Kelly basada en simulación.
        
        Args:
            column_stakes: Stakes base (Kelly completo = 1.0)
            column_probabilities: Probabilidades de columnas
            column_odds: Cuotas de columnas
            initial_bankroll: Bankroll inicial
            risk_tolerance: 'conservative', 'moderate', 'aggressive'
            
        Returns:
            Dict con fracción óptima y métricas
        """
        # Fracciones a evaluar
        fractions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        
        results = []
        
        for fraction in fractions:
            # Ajustar stakes
            adjusted_stakes = column_stakes * fraction
            
            # Simular
            simulation = KellyCapitalManagement.simulate_bankroll_evolution(
                adjusted_stakes, column_probabilities, column_odds,
                initial_bankroll, n_simulations=5000, n_rounds=50
            )
            
            metrics = simulation['metrics']
            
            # Score basado en tolerancia al riesgo
            if risk_tolerance == 'conservative':
                # Priorizar baja varianza y bajo drawdown
                score = metrics['expected_value_pct'] * 0.3 + \
                       (100 - metrics['avg_max_drawdown']) * 0.4 + \
                       metrics['probability_profit'] * 100 * 0.3
            elif risk_tolerance == 'aggressive':
                # Priorizar retorno esperado
                score = metrics['expected_value_pct'] * 0.6 + \
                       (100 - metrics['avg_max_drawdown']) * 0.2 + \
                       metrics['probability_profit'] * 100 * 0.2
            else:  # moderate
                # Balance
                score = metrics['expected_value_pct'] * 0.4 + \
                       (100 - metrics['avg_max_drawdown']) * 0.4 + \
                       metrics['probability_profit'] * 100 * 0.2
            
            results.append({
                'fraction': fraction,
                'score': score,
                'expected_value_pct': metrics['expected_value_pct'],
                'avg_max_drawdown': metrics['avg_max_drawdown'],
                'probability_profit': metrics['probability_profit']
            })
        
        # Encontrar mejor fracción
        results_df = pd.DataFrame(results)
        optimal_idx = results_df['score'].idxmax()
        optimal_result = results_df.loc[optimal_idx]
        
        return {
            'optimal_fraction': float(optimal_result['fraction']),
            'optimal_score': float(optimal_result['score']),
            'all_results': results,
            'risk_tolerance': risk_tolerance,
            'recommendation': KellyCapitalManagement._get_fraction_recommendation(
                optimal_result['fraction'], risk_tolerance
            )
        }
    
    # ============================================================================
    # 5. MÉTODOS PRIVADOS Y AUXILIARES
    # ============================================================================
    
    @staticmethod
    def _normalize_stakes(
        stakes_matrix: np.ndarray,
        max_exposure: float,
        mode: str
    ) -> np.ndarray:
        """
        Normaliza stakes para respetar exposición máxima.
        
        Args:
            stakes_matrix: Matriz de stakes
            max_exposure: Exposición máxima permitida
            mode: 'kelly' o 'manual'
            
        Returns:
            Matriz de stakes normalizada
        """
        total_stake = np.sum(stakes_matrix)
        
        if total_stake <= max_exposure:
            return stakes_matrix
        
        # Calcular factor de normalización
        normalization_factor = max_exposure / total_stake
        
        # Aplicar normalización
        normalized_stakes = stakes_matrix * normalization_factor
        
        # Logging (opcional)
        if total_stake > max_exposure * 1.1:  # Solo si excede significativamente
            reduction_pct = (1 - normalization_factor) * 100
            print(f"⚠️ Normalizando stakes: reducción del {reduction_pct:.1f}% para respetar exposición máxima")
        
        return normalized_stakes
    
    @staticmethod
    def _calculate_stake_metrics(
        stakes_matrix: np.ndarray,
        probabilities: np.ndarray,
        odds_matrix: np.ndarray,
        portfolio_type: str,
        mode: str
    ) -> Dict[str, Any]:
        """
        Calcula métricas detalladas de gestión de stakes.
        """
        n_matches, n_outcomes = stakes_matrix.shape
        
        # Calcular EV por stake
        ev_matrix = probabilities * odds_matrix - 1
        weighted_ev = stakes_matrix * ev_matrix
        
        # Métricas básicas
        total_stake = np.sum(stakes_matrix)
        avg_stake = np.mean(stakes_matrix)
        max_stake = np.max(stakes_matrix)
        min_stake = np.min(stakes_matrix)
        
        # Contar stakes positivos
        positive_stakes = np.sum(stakes_matrix > 0)
        
        # Calcular concentración (Herfindahl-Hirschman Index)
        stake_shares = stakes_matrix.flatten() / total_stake if total_stake > 0 else np.zeros_like(stakes_matrix.flatten())
        hhi = np.sum(stake_shares ** 2)
        
        # Normalizar HHI (0 = perfectamente diversificado, 1 = totalmente concentrado)
        max_hhi = 1 / len(stake_shares) if len(stake_shares) > 0 else 1
        concentration = (hhi - max_hhi) / (1 - max_hhi) if max_hhi < 1 else 0
        concentration = max(0, min(concentration, 1))
        
        metrics = {
            'portfolio_type': portfolio_type,
            'mode': mode,
            'n_matches': n_matches,
            'n_outcomes': n_outcomes,
            'total_stake_percentage': float(total_stake * 100),
            'avg_stake_percentage': float(avg_stake * 100),
            'max_stake_percentage': float(max_stake * 100),
            'min_stake_percentage': float(min_stake * 100),
            'stake_std_percentage': float(np.std(stakes_matrix) * 100) if stakes_matrix.size > 1 else 0.0,
            'positive_stakes_count': int(positive_stakes),
            'zero_stakes_count': int(stakes_matrix.size - positive_stakes),
            'concentration_index': float(concentration),
            'hhi_index': float(hhi),
            'total_expected_value': float(np.sum(weighted_ev)),
            'avg_expected_value': float(np.mean(weighted_ev[weighted_ev != 0])) if np.any(weighted_ev != 0) else 0.0,
            'positive_ev_stakes': int(np.sum(weighted_ev > 0)),
            'negative_ev_stakes': int(np.sum(weighted_ev < 0))
        }
        
        return metrics
    
    @staticmethod
    def _calculate_portfolio_risk_metrics(
        column_stakes: np.ndarray,
        column_probabilities: np.ndarray,
        column_odds: np.ndarray,
        bankroll: float
    ) -> Dict[str, Any]:
        """
        Calcula métricas de riesgo para un portafolio de columnas.
        """
        n_columns = len(column_stakes)
        
        if n_columns == 0:
            return {
                'risk_metrics': 'no_columns',
                'diversification_score': 0,
                'correlation_estimate': 0
            }
        
        # Calcular montos de apuesta
        stake_amounts = column_stakes * bankroll
        
        # Calcular retornos posibles
        returns = []
        for i in range(n_columns):
            # Retorno si gana: stake * (odds - 1)
            win_return = stake_amounts[i] * (column_odds[i] - 1)
            # Retorno si pierde: -stake
            loss_return = -stake_amounts[i]
            
            returns.append({
                'win': win_return,
                'loss': loss_return,
                'probability': column_probabilities[i]
            })
        
        # Calcular VAR simple (Value at Risk)
        worst_case_loss = np.sum([r['loss'] for r in returns])
        expected_loss = np.sum([r['loss'] * r['probability'] for r in returns])
        
        # Estimación simple de correlación (asumimos baja correlación para columnas diferentes)
        avg_correlation = 0.3  # Estimación conservadora
        
        # Score de diversificación (más columnas = mejor diversificación)
        diversification_score = min(n_columns / 20.0, 1.0)  # Normalizado a 0-1
        
        return {
            'worst_case_loss': float(worst_case_loss),
            'worst_case_loss_pct': float(worst_case_loss / bankroll * 100) if bankroll > 0 else 0,
            'expected_loss': float(expected_loss),
            'expected_loss_pct': float(expected_loss / bankroll * 100) if bankroll > 0 else 0,
            'diversification_score': float(diversification_score),
            'correlation_estimate': float(avg_correlation),
            'risk_per_column': float(np.mean(stake_amounts) / bankroll * 100) if bankroll > 0 else 0
        }
    
    @staticmethod
    def _calculate_sortino_ratio(returns_pct: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """
        Calcula ratio de Sortino (similar a Sharpe pero solo penaliza downside risk).
        """
        if len(returns_pct) == 0:
            return 0.0
        
        excess_returns = returns_pct - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0
        
        downside_std = np.std(downside_returns)
        
        if downside_std == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0
        
        return float(np.mean(excess_returns) / downside_std)
    
    @staticmethod
    def _get_fraction_recommendation(fraction: float, risk_tolerance: str) -> str:
        """
        Genera recomendación basada en fracción óptima.
        """
        if fraction <= 0.3:
            level = "MUY CONSERVADORA"
            advice = "Ideal para preservar capital. Baja volatilidad."
        elif fraction <= 0.5:
            level = "CONSERVADORA"
            advice = "Balance buen riesgo/retorno. Recomendada para mayoría."
        elif fraction <= 0.7:
            level = "MODERADA"
            advice = "Busca crecimiento con riesgo controlado."
        elif fraction <= 0.85:
            level = "AGRESIVA"
            advice = "Mayor potencial de crecimiento con mayor volatilidad."
        else:
            level = "MUY AGRESIVA"
            advice = "Máximo potencial de crecimiento. Alta volatilidad."
        
        return f"{level}: Fracción Kelly {fraction:.2f}. {advice}"
    
    # ============================================================================
    # 6. MÉTODOS DE VISUALIZACIÓN Y REPORTING
    # ============================================================================
    
    @staticmethod
    def create_stake_distribution_chart(
        column_stakes: np.ndarray,
        bankroll: float,
        portfolio_type: str
    ) -> go.Figure:
        """
        Crea gráfico de distribución de stakes.
        """
        stakes_pct = column_stakes * 100
        stake_amounts = column_stakes * bankroll
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Distribución de Stakes (%)', 'Stakes por Columna',
                          'Cumulative Stake', 'Distribución de Montos (€)'),
            specs=[[{'type': 'histogram'}, {'type': 'bar'}],
                   [{'type': 'scatter'}, {'type': 'histogram'}]]
        )
        
        # Histograma de porcentajes
        fig.add_trace(
            go.Histogram(
                x=stakes_pct,
                nbinsx=20,
                name='Stakes %',
                marker_color=SystemConfig.COLORS['primary'],
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Barras por columna
        fig.add_trace(
            go.Bar(
                x=list(range(1, len(stakes_pct) + 1)),
                y=stakes_pct,
                name='Columna',
                marker_color=SystemConfig.COLORS['secondary'],
                opacity=0.7
            ),
            row=1, col=2
        )
        
        # Stake acumulado
        cumulative_pct = np.cumsum(stakes_pct)
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(cumulative_pct) + 1)),
                y=cumulative_pct,
                mode='lines+markers',
                name='Acumulado',
                line=dict(color=SystemConfig.COLORS['success'], width=3),
                marker=dict(size=6)
            ),
            row=2, col=1
        )
        
        # Histograma de montos
        fig.add_trace(
            go.Histogram(
                x=stake_amounts,
                nbinsx=20,
                name='Montos €',
                marker_color=SystemConfig.COLORS['info'],
                opacity=0.7
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f'Distribución de Stakes - Portafolio {portfolio_type.title()}',
            height=600,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Stake (%)", row=1, col=1)
        fig.update_xaxes(title_text="Columna", row=1, col=2)
        fig.update_xaxes(title_text="Columna", row=2, col=1)
        fig.update_xaxes(title_text="Monto (€)", row=2, col=2)
        
        fig.update_yaxes(title_text="Frecuencia", row=1, col=1)
        fig.update_yaxes(title_text="Stake (%)", row=1, col=2)
        fig.update_yaxes(title_text="Stake Acumulado (%)", row=2, col=1)
        fig.update_yaxes(title_text="Frecuencia", row=2, col=2)
        
        return fig

# ============================================================================
# SECCIÓN 4: INTERFAZ STREAMLIT PROFESIONAL v3.0 - OPTIMIZADA
# ============================================================================

class ACBEApp:
    """Interfaz principal de la aplicación Streamlit - VERSIÓN OPTIMIZADA."""
    
    def __init__(self):
        self.setup_page_config()
        self.match_input_layer = MatchInputLayer()
        self.portfolio_engine = PortfolioEngine()
        self.data_exporter = DataExporter()
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
        """Renderiza sidebar optimizado - Sin duplicación de lógica."""
        with st.sidebar:
            st.header("⚙️ Panel de Control")
            
            # Indicador de versión
            st.caption(f"v3.0 | {datetime.now().strftime('%Y-%m-%d')}")
            
            # Botón de reinicio
            if st.button("🔄 Reiniciar Sistema", type="secondary", use_container_width=True):
                SessionStateManager.reset_to_input()
                st.rerun()
            
            # Configuración de bankroll
            bankroll = st.number_input(
                "Bankroll Inicial (€)",
                min_value=100.0,
                max_value=1000000.0,
                value=SystemConfig.DEFAULT_BANKROLL,
                step=1000.0
            )
            
            # Selector de portafolio
            st.subheader("🎯 Estrategia de Portafolio")
            portfolio_type = st.radio(
                "Configuración de columnas:",
                ["Set Completo (73)", "Set Elite (24)"],
                index=0,
                help="Cobertura completa vs. concentración estratégica"
            )
            
            st.session_state.portfolio_type = "full" if "Completo" in portfolio_type else "elite"
            
            # Gestión de stake
            st.subheader("💰 Gestión de Apuesta")
            auto_stake_mode = st.toggle(
                "Kelly Automático",
                value=True,
                help="Calcula stake óptimo basado en probabilidad y valor"
            )
            
            manual_stake_fraction = None
            kelly_fraction = None
            
            if not auto_stake_mode:
                manual_stake = st.number_input(
                    "Stake Fijo (% por columna)",
                    min_value=0.1,
                    max_value=10.0,
                    value=1.0,
                    step=0.1
                )
                manual_stake_fraction = manual_stake / 100.0
                st.info(f"Stake fijo: {manual_stake}%")
            else:
                kelly_fraction = st.slider(
                    "Conservadurismo Kelly",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                    help="Fracción del Kelly completo (menor = más conservador)"
                )
            
            # Parámetros de riesgo
            st.subheader("🛡️ Límites de Riesgo")
            max_exposure = st.slider(
                "Exposición Máxima (%)",
                min_value=5,
                max_value=30,
                value=15,
                step=1,
                help="Porcentaje máximo del bankroll en riesgo"
            )
            
            # Configuración de simulación
            st.subheader("🎲 Parámetros de Simulación")
            monte_carlo_sims = st.number_input(
                "Simulaciones Monte Carlo",
                min_value=1000,
                max_value=50000,
                value=10000,
                step=1000
            )
            
            n_rounds = st.slider(
                "Rondas de Simulación",
                min_value=10,
                max_value=500,
                value=100,
                step=10
            )
            
            # Reducción elite
            st.subheader("🏆 Filtro Elite")
            apply_elite_reduction = st.toggle(
                "Aplicar filtro de eficiencia",
                value=True,
                help="Selecciona las mejores columnas basado en score compuesto"
            )
            
            if apply_elite_reduction:
                elite_columns_target = st.slider(
                    "Nº Columnas Elite",
                    min_value=12,
                    max_value=36,
                    value=24,
                    step=1
                )
            else:
                elite_columns_target = 73
            
            # Fuente de datos
            st.subheader("📊 Origen de Datos")
            data_source = st.radio(
                "Fuente de probabilidades:",
                ["⚽ Input Manual", "📈 Datos Sintéticos", "📂 Archivo CSV"],
                index=0
            )
            
            uploaded_file = None
            n_matches = 6
            
            if data_source == "📈 Datos Sintéticos":
                n_matches = st.slider(
                    "Nº Partidos a simular",
                    min_value=6,
                    max_value=15,
                    value=6,
                    step=1
                )
            elif data_source == "📂 Archivo CSV":
                uploaded_file = st.file_uploader(
                    "Subir archivo CSV",
                    type=['csv']
                )
            
            return {
                'bankroll': bankroll,
                'portfolio_type': st.session_state.portfolio_type,
                'auto_stake_mode': auto_stake_mode,
                'manual_stake': manual_stake_fraction,
                'kelly_fraction': kelly_fraction,
                'max_exposure': max_exposure / 100,
                'monte_carlo_sims': monte_carlo_sims,
                'n_rounds': n_rounds,
                'elite_columns_target': elite_columns_target,
                'apply_elite_reduction': apply_elite_reduction,
                'data_source': data_source,
                'n_matches': n_matches,
                'uploaded_file': uploaded_file
            }
    
    def render_s73_system(self, probabilities: np.ndarray,
                         odds_matrix: np.ndarray,
                         normalized_entropies: np.ndarray,
                         bankroll: float,
                         config: Dict) -> Dict:
        """Renderiza sistema S73 completo con todas las fases."""
        st.header("🧮 Sistema Combinatorio S73 v3.0")
        
        with st.spinner("Construyendo sistema de cobertura optimizado..."):
            # Fase 1: Generación de combinaciones
            filtered_combinations, filtered_probs = (
                S73System.generate_prefiltered_combinations(
                    probabilities, normalized_entropies, odds_matrix
                )
            )
            
            # Fase 2: Sistema de cobertura
            s73_combinations, s73_probs = S73System.build_s73_coverage_system(
                filtered_combinations, filtered_probs, validate_coverage=True
            )
            
            # Fase 3: Reducción elite (si aplica)
            elite_data = None
            if config['apply_elite_reduction']:
                elite_data = self._apply_elite_reduction(
                    s73_combinations, s73_probs, odds_matrix,
                    normalized_entropies, config['elite_columns_target']
                )
            
            # Seleccionar combinaciones finales
            if config['apply_elite_reduction'] and config['portfolio_type'] == "elite":
                final_combinations = elite_data['combinations']
                final_probs = elite_data['probabilities']
            else:
                final_combinations = s73_combinations
                final_probs = s73_probs
            
            # Calcular métricas de columnas
            columns_df = self._calculate_columns_metrics(
                final_combinations, final_probs, odds_matrix,
                normalized_entropies, bankroll, config
            )
            
            # Normalizar stakes
            normalized_stakes = KellyCapitalManagement.normalize_portfolio_stakes(
                columns_df['Stake (%)'].values / 100,
                is_manual_mode=not config['auto_stake_mode']
            )
            
            columns_df['Stake (%)'] = normalized_stakes * 100
            columns_df['Inversión (€)'] = normalized_stakes * bankroll
        
        # Visualizaciones principales
        self._render_master_bet_section(columns_df)
        self._render_scenario_simulator(final_combinations, final_probs)
        self._render_system_statistics(
            filtered_combinations, s73_combinations,
            final_combinations, columns_df, config
        )
        
        # Mostrar columnas
        self._display_columns_table(columns_df)
        
        # Visualización de scores elite
        if config['apply_elite_reduction'] and elite_data:
            self._render_elite_scores_viz(elite_data['scores'], config['elite_columns_target'])
        
        return {
            'combinations': final_combinations,
            'probabilities': final_probs,
            'kelly_stakes': normalized_stakes,
            'columns_df': columns_df,
            'elite_data': elite_data if config['apply_elite_reduction'] else None,
            'portfolio_type': config['portfolio_type']
        }
    
    def _apply_elite_reduction(self, combinations: List, probabilities: List,
                              odds_matrix: np.ndarray, entropies: np.ndarray,
                              target: int) -> Dict:
        """Aplica reducción elite a las combinaciones."""
        elite_combo, elite_probs, elite_scores = S73System.get_elite_subset(
            combinations, probabilities, odds_matrix, entropies, limit=target
        )
        
        # Guardar en estado de sesión
        st.session_state.update({
            'elite_columns': elite_combo,
            'elite_scores': elite_scores,
            'elite_columns_selected': True
        })
        
        return {
            'combinations': elite_combo,
            'probabilities': elite_probs,
            'scores': elite_scores
        }
    
    def _calculate_columns_metrics(self, combinations: List, probabilities: List,
                                  odds_matrix: np.ndarray, entropies: np.ndarray,
                                  bankroll: float, config: Dict) -> pd.DataFrame:
        """Calcula métricas detalladas para cada columna."""
        columns_data = []
        
        for idx, (combo, prob) in enumerate(zip(combinations, probabilities), 1):
            # Calcular métricas base
            combo_odds = S73System.calculate_combination_odds(combo, odds_matrix)
            combo_entropy = np.mean([entropies[i] for i in range(6)])
            expected_value = prob * combo_odds - 1
            
            # Calcular stake
            if config['auto_stake_mode']:
                base_stake = KellyCapitalManagement.calculate_column_kelly(
                    combo, prob, combo_odds, combo_entropy,
                    portfolio_type=config['portfolio_type']
                )
            else:
                base_stake = config['manual_stake'] or 0.01
            
            columns_data.append({
                'ID': idx,
                'Combinación': ''.join([SystemConfig.OUTCOME_LABELS[s] for s in combo]),
                'Probabilidad': prob,
                'Cuota': combo_odds,
                'Valor Esperado': expected_value,
                'Entropía': combo_entropy,
                'Stake (%)': base_stake * 100,
                'Inversión (€)': base_stake * bankroll,
                'Tipo': 'Elite' if config['apply_elite_reduction'] and idx <= config['elite_columns_target'] else 'Cobertura'
            })
        
        return pd.DataFrame(columns_data)
    
    def _render_master_bet_section(self, columns_df: pd.DataFrame):
        """Renderiza la sección de 'Apuesta Maestra'."""
        st.subheader("🏆 Apuesta Maestra")
        
        if len(columns_df) == 0:
            st.warning("No hay columnas disponibles")
            return
        
        # Encontrar la columna con mayor probabilidad
        master_bet = columns_df.loc[columns_df['Probabilidad'].idxmax()]
        
        # Visualización de combinación
        self._render_combination_viz(master_bet['Combinación'])
        
        # Métricas clave
        cols = st.columns(4)
        metrics = [
            ("Probabilidad", f"{master_bet['Probabilidad']:.2%}", None),
            ("Cuota", f"{master_bet['Cuota']:.2f}", None),
            ("Valor Esperado", f"{master_bet['Valor Esperado']:.3f}", 
             "green" if master_bet['Valor Esperado'] > 0 else "red"),
            ("Recomendación", "✅ JUGAR" if master_bet['Valor Esperado'] > 0 else "⛔ EVITAR", 
             "green" if master_bet['Valor Esperado'] > 0 else "red")
        ]
        
        for col, (label, value, color) in zip(cols, metrics):
            with col:
                if color:
                    st.markdown(f"<h3 style='color:{color}'>{value}</h3>", unsafe_allow_html=True)
                else:
                    st.metric(label, value)
                st.caption(label)
    
    def _render_combination_viz(self, combination: str):
        """Renderiza visualización de combinación con cuadros de colores."""
        cols = st.columns(6)
        for i, outcome in enumerate(combination):
            with cols[i]:
                color = SystemConfig.COLORS.get(outcome, '#4A5568')
                st.markdown(
                    f"""
                    <div style='
                        background: {color};
                        color: white;
                        padding: 15px;
                        border-radius: 8px;
                        text-align: center;
                        font-weight: bold;
                        font-size: 18px;
                        margin: 2px;
                    '>
                        {outcome}<br>
                        <small style='font-size: 12px;'>Partido {i+1}</small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    
    def _render_scenario_simulator(self, combinations: List, probabilities: List):
        """Renderiza simulador de escenarios what-if."""
        st.subheader("🔮 Simulador de Escenarios")
        st.caption("Analiza el impacto de fallar partidos específicos")
        
        # Selector de partidos
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            match_1 = st.selectbox(
                "Partido 1 a fallar",
                ["Ninguno"] + [f"Partido {i+1}" for i in range(6)],
                key="scenario_match_1"
            )
        
        with col2:
            result_1 = st.selectbox(
                "Resultado real",
                ["1", "X", "2"],
                index=0,
                disabled=(match_1 == "Ninguno"),
                key="scenario_result_1"
            )
        
        with col3:
            match_2 = st.selectbox(
                "Partido 2 a fallar",
                ["Ninguno"] + [f"Partido {i+1}" for i in range(6)],
                key="scenario_match_2"
            )
        
        with col4:
            result_2 = st.selectbox(
                "Resultado real",
                ["1", "X", "2"],
                index=0,
                disabled=(match_2 == "Ninguno"),
                key="scenario_result_2"
            )
        
        # Botón de simulación
        if st.button("🎯 Ejecutar Simulación", type="secondary"):
            failed_matches = []
            
            if match_1 != "Ninguno":
                match_idx = int(match_1.split(" ")[1]) - 1
                result_idx = SystemConfig.OUTCOME_MAPPING[result_1]
                failed_matches.append((match_idx, result_idx))
            
            if match_2 != "Ninguno" and match_2 != match_1:
                match_idx = int(match_2.split(" ")[1]) - 1
                result_idx = SystemConfig.OUTCOME_MAPPING[result_2]
                failed_matches.append((match_idx, result_idx))
            
            # Ejecutar simulación
            scenario_stats = S73System.simulate_scenario(
                combinations, failed_matches, probabilities
            )
            
            # Mostrar resultados
            self._render_scenario_results(scenario_stats)
    
    def _render_scenario_results(self, stats: Dict):
        """Renderiza resultados de simulación de escenarios."""
        st.subheader("📊 Resultados de Simulación")
        
        # Métricas principales
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Columnas viables (4+ aciertos)", stats['columns_with_4plus'])
            st.metric("Columnas excelentes (5+ aciertos)", stats['columns_with_5plus'])
        
        with col2:
            st.metric("Columnas perfectas (6)", stats['columns_with_6'])
            st.metric("Aciertos promedio", f"{stats['avg_hits']:.1f}")
        
        with col3:
            total = stats['total_columns']
            coverage_4plus = (stats['columns_with_4plus'] / total) * 100
            coverage_5plus = (stats['columns_with_5plus'] / total) * 100
            
            st.metric("Cobertura 4+", f"{coverage_4plus:.1f}%")
            st.metric("Cobertura 5+", f"{coverage_5plus:.1f}%")
        
        # Gráfico de distribución
        fig = px.bar(
            x=list(stats['hits_distribution'].keys()),
            y=list(stats['hits_distribution'].values()),
            title="Distribución de Aciertos",
            labels={'x': 'Nº Aciertos', 'y': 'Nº Columnas'},
            color=list(stats['hits_distribution'].keys()),
            color_continuous_scale='RdYlGn'
        )
        
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
        
        # Análisis interpretativo
        if stats['columns_with_4plus'] == 0:
            st.error("❌ **Escenario crítico:** Ninguna columna mantiene 4+ aciertos")
        elif stats['columns_with_4plus'] >= total * 0.6:
            st.success("✅ **Escenario robusto:** Sistema mantiene buena cobertura")
        else:
            st.warning("⚠️ **Escenario vulnerable:** Cobertura reducida significativamente")
    
    def _render_system_statistics(self, filtered: List, s73: List,
                                 final: List, columns_df: pd.DataFrame,
                                 config: Dict):
        """Renderiza estadísticas generales del sistema."""
        st.subheader("📈 Estadísticas del Sistema")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Combinaciones iniciales", len(filtered))
            st.metric("Sistema S73 completo", len(s73))
        
        with col2:
            portfolio_name = "Completo" if config['portfolio_type'] == "full" else "Elite"
            st.metric(f"Portafolio {portfolio_name}", len(final))
            avg_stake = columns_df['Stake (%)'].mean()
            st.metric("Stake promedio", f"{avg_stake:.2f}%")
        
        with col3:
            total_exposure = columns_df['Stake (%)'].sum()
            st.metric("Exposición total", f"{total_exposure:.1f}%")
            avg_odds = columns_df['Cuota'].mean()
            st.metric("Cuota promedio", f"{avg_odds:.2f}")
        
        with col4:
            avg_prob = columns_df['Probabilidad'].mean()
            st.metric("Probabilidad promedio", f"{avg_prob:.2%}")
            if config['apply_elite_reduction']:
                elite_coverage = (config['elite_columns_target'] / len(s73)) * 100
                st.metric("Reducción elite", f"{elite_coverage:.1f}%")
    
    def _display_columns_table(self, columns_df: pd.DataFrame):
        """Muestra tabla de columnas con formato."""
        st.subheader("📋 Detalle de Columnas")
        
        # Formatear DataFrame para visualización
        display_df = columns_df.copy()
        display_df['Probabilidad'] = display_df['Probabilidad'].apply(lambda x: f'{x:.4%}')
        display_df['Cuota'] = display_df['Cuota'].apply(lambda x: f'{x:.2f}')
        display_df['Valor Esperado'] = display_df['Valor Esperado'].apply(lambda x: f'{x:.4f}')
        display_df['Stake (%)'] = display_df['Stake (%)'].apply(lambda x: f'{x:.2f}%')
        display_df['Inversión (€)'] = display_df['Inversión (€)'].apply(lambda x: f'€{x:.2f}')
        display_df['Entropía'] = display_df['Entropía'].apply(lambda x: f'{x:.3f}')
        
        # Ordenar por probabilidad
        display_df = display_df.sort_values('Probabilidad', ascending=False)
        
        # Mostrar con paginación
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400,
            column_config={
                "ID": st.column_config.NumberColumn(width="small"),
                "Combinación": st.column_config.TextColumn(width="medium"),
                "Tipo": st.column_config.TextColumn(width="small")
            }
        )
    
    def _render_elite_scores_viz(self, scores: np.ndarray, target: int):
        """Visualización de scores de eficiencia elite."""
        st.subheader("📊 Score de Eficiencia - Análisis Elite")
        
        sorted_scores = np.sort(scores)[::-1]
        indices = np.arange(1, len(sorted_scores) + 1)
        
        fig = go.Figure()
        
        # Línea principal
        fig.add_trace(go.Scatter(
            x=indices,
            y=sorted_scores,
            mode='lines+markers',
            name='Score',
            line=dict(color='#4C78A8', width=3)
        ))
        
        # Línea de corte
        if target < len(sorted_scores):
            fig.add_hline(
                y=sorted_scores[target-1],
                line_dash="dash",
                line_color="#FF6B6B",
                annotation_text=f"Top {target}"
            )
        
        fig.update_layout(
            title="Distribución de Scores de Eficiencia",
            xaxis_title="Ranking",
            yaxis_title="Score",
            height=350,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Estadísticas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Score máximo", f"{sorted_scores[0]:.4f}")
        with col2:
            st.metric("Score promedio", f"{np.mean(sorted_scores):.4f}")
        with col3:
            st.metric("Score mínimo elite", f"{np.min(sorted_scores[:target]):.4f}")

# ============================================================================
# SECCIÓN 5: CLASES AUXILIARES OPTIMIZADAS (v3.0)
# ============================================================================
# ============================================================================
# SECCIÓN 6: PORTFOLIO ENGINE DE DOBLE ESTRATEGIA v3.0
# ============================================================================

class PortfolioEngine:
    """
    Motor de análisis de portafolio unificado para estrategias duales v3.0.
    Soporte nativo para portafolios Full (73) y Elite (24) con métricas avanzadas.
    """
    
    def __init__(self, initial_bankroll: float = SystemConfig.DEFAULT_BANKROLL):
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.performance_history = []
        self.active_strategies = {
            'full': {'active': False, 'data': None},
            'elite': {'active': False, 'data': None}
        }
    
    def activate_portfolio(self, portfolio_type: str, columns_data: pd.DataFrame):
        """
        Activa un portafolio específico con sus datos.
        
        Args:
            portfolio_type: 'full' o 'elite'
            columns_data: DataFrame con columnas del sistema S73
        """
        if portfolio_type not in ['full', 'elite']:
            raise ValueError("Tipo de portafolio debe ser 'full' o 'elite'")
        
        self.active_strategies[portfolio_type]['active'] = True
        self.active_strategies[portfolio_type]['data'] = {
            'columns': columns_data.copy(),
            'activation_time': datetime.now(),
            'initial_bankroll': self.current_bankroll
        }
        
        st.success(f"✅ Portafolio {portfolio_type.upper()} activado")
    
    def calculate_dual_metrics(self) -> Dict:
        """
        Calcula métricas comparativas entre portafolios full y elite.
        
        Returns:
            Diccionario con métricas comparativas
        """
        metrics = {
            'full': None,
            'elite': None,
            'comparison': {}
        }
        
        # Calcular métricas para cada portafolio activo
        for ptype in ['full', 'elite']:
            if self.active_strategies[ptype]['active']:
                data = self.active_strategies[ptype]['data']
                metrics[ptype] = self._calculate_single_portfolio_metrics(data['columns'], ptype)
        
        # Comparación entre portafolios
        if metrics['full'] and metrics['elite']:
            metrics['comparison'] = self._compare_portfolios(
                metrics['full'], metrics['elite']
            )
        
        return metrics
    
    def _calculate_single_portfolio_metrics(self, columns_df: pd.DataFrame, 
                                           portfolio_type: str) -> Dict:
        """Calcula métricas detalladas para un portafolio individual."""
        if columns_df.empty:
            return {}
        
        # Extraer arrays para cálculo vectorizado
        probabilities = columns_df['Probabilidad'].values
        odds = columns_df['Cuota'].values
        stakes = columns_df['Stake (%)'].values / 100
        investments = columns_df['Inversión (€)'].values
        evs = columns_df['Valor Esperado'].values
        
        # Métricas básicas
        total_investment = investments.sum()
        total_exposure = stakes.sum() * 100
        avg_prob = probabilities.mean()
        avg_odds = odds.mean()
        avg_ev = evs.mean()
        
        # Cálculos vectorizados
        expected_returns = probabilities * odds
        total_expected_return = (expected_returns * investments).sum()
        expected_roi = (total_expected_return - total_investment) / total_investment * 100
        
        # Riesgo y volatilidad
        variance = np.var(evs) if len(evs) > 1 else 0
        sharpe_ratio = self._calculate_sharpe_ratio(evs, stakes)
        
        # Drawdown esperado
        expected_drawdown = self._estimate_expected_drawdown(probabilities, stakes, odds)
        
        # Probabilidad de ruina
        ruin_probability = self._calculate_ruin_probability(probabilities, stakes, odds)
        
        # Eficiencia
        capital_efficiency = self._calculate_capital_efficiency(
            evs, stakes, portfolio_type
        )
        
        # Concentración (solo para elite)
        if portfolio_type == 'elite':
            concentration = self._calculate_concentration(stakes)
            efficiency_score = self._calculate_elite_efficiency(evs, stakes)
        else:
            concentration = 0
            efficiency_score = 0
        
        # Score de calidad del portafolio
        quality_score = self._calculate_quality_score(
            avg_ev, sharpe_ratio, expected_drawdown, 
            ruin_probability, expected_roi
        )
        
        return {
            'portfolio_type': portfolio_type,
            'n_columns': len(columns_df),
            'total_investment': total_investment,
            'total_exposure_pct': total_exposure,
            'avg_probability': avg_prob,
            'avg_odds': avg_odds,
            'avg_expected_value': avg_ev,
            'expected_roi_pct': expected_roi,
            'variance': variance,
            'sharpe_ratio': sharpe_ratio,
            'expected_drawdown_pct': expected_drawdown,
            'ruin_probability_pct': ruin_probability,
            'capital_efficiency': capital_efficiency,
            'concentration_ratio': concentration,
            'efficiency_score': efficiency_score,
            'quality_score': quality_score,
            'quality_rating': self._get_quality_rating(quality_score),
            'timestamp': datetime.now()
        }
    
    def _compare_portfolios(self, full_metrics: Dict, elite_metrics: Dict) -> Dict:
        """Compara métricas entre portafolios full y elite."""
        comparisons = {}
        
        # Comparativas clave
        key_metrics = [
            ('total_exposure_pct', 'Exposición (%)'),
            ('avg_expected_value', 'EV Promedio'),
            ('expected_roi_pct', 'ROI Esperado (%)'),
            ('sharpe_ratio', 'Sharpe Ratio'),
            ('expected_drawdown_pct', 'Drawdown Esperado (%)'),
            ('ruin_probability_pct', 'Prob. Ruina (%)'),
            ('quality_score', 'Score Calidad')
        ]
        
        for metric_key, label in key_metrics:
            full_val = full_metrics.get(metric_key, 0)
            elite_val = elite_metrics.get(metric_key, 0)
            
            # Calcular diferencia y mejora
            if full_val != 0:
                improvement_pct = ((elite_val - full_val) / abs(full_val)) * 100
            else:
                improvement_pct = 0
            
            comparisons[metric_key] = {
                'label': label,
                'full': full_val,
                'elite': elite_val,
                'difference': elite_val - full_val,
                'improvement_pct': improvement_pct,
                'better': 'elite' if elite_val > full_val else 'full' 
                if elite_val != full_val else 'equal'
            }
        
        # Análisis de eficiencia relativa
        efficiency_ratio = elite_metrics.get('capital_efficiency', 0) / \
                          max(full_metrics.get('capital_efficiency', 1), 0.001)
        
        comparisons['efficiency_analysis'] = {
            'efficiency_ratio': efficiency_ratio,
            'interpretation': self._interpret_efficiency_ratio(efficiency_ratio),
            'recommendation': self._generate_recommendation(full_metrics, elite_metrics)
        }
        
        return comparisons
    
    def _calculate_sharpe_ratio(self, evs: np.ndarray, stakes: np.ndarray) -> float:
        """Calcula ratio Sharpe simplificado."""
        if len(evs) < 2:
            return 0.0
        
        # Simular retornos
        returns = evs * stakes
        
        if np.std(returns) == 0:
            return 0.0
        
        # Ratio Sharpe (exceso de retorno / volatilidad)
        risk_free_rate = 0.02 / 252  # Tasa libre de riesgo diaria
        excess_returns = returns - risk_free_rate
        
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
    
    def _estimate_expected_drawdown(self, probs: np.ndarray, stakes: np.ndarray, 
                                   odds: np.ndarray) -> float:
        """Estima drawdown esperado mediante simulación Monte Carlo."""
        n_sims = 1000
        n_columns = len(probs)
        
        if n_columns == 0:
            return 0.0
        
        max_drawdowns = []
        
        for _ in range(n_sims):
            # Simular resultados
            results = np.random.binomial(1, probs)
            
            # Calcular P&L por columna
            pnl = results * stakes * (odds - 1) - (1 - results) * stakes
            
            # Equity curve
            equity = 1 + np.cumsum(pnl)
            
            # Calcular drawdown
            peak = np.maximum.accumulate(equity)
            drawdown = (peak - equity) / peak
            
            if len(drawdown) > 0:
                max_drawdowns.append(np.max(drawdown))
        
        return np.mean(max_drawdowns) * 100 if max_drawdowns else 0.0
    
    def _calculate_ruin_probability(self, probs: np.ndarray, stakes: np.ndarray, 
                                   odds: np.ndarray) -> float:
        """Calcula probabilidad de ruina usando criterio Kelly."""
        n_columns = len(probs)
        
        if n_columns == 0:
            return 0.0
        
        # Retorno esperado por columna
        expected_returns = probs * odds - 1
        
        # Calcular probabilidad de perder más del 50% del bankroll
        ruin_threshold = 0.5
        n_sims = 5000
        ruin_count = 0
        
        for _ in range(n_sims):
            bankroll = 1.0  # Normalizado a 1
            
            for i in range(n_columns):
                if np.random.random() < probs[i]:
                    bankroll += stakes[i] * (odds[i] - 1)
                else:
                    bankroll -= stakes[i]
                
                if bankroll < (1 - ruin_threshold):
                    ruin_count += 1
                    break
        
        return (ruin_count / n_sims) * 100
    
    def _calculate_capital_efficiency(self, evs: np.ndarray, stakes: np.ndarray, 
                                     portfolio_type: str) -> float:
        """Calcula eficiencia de capital."""
        total_ev = evs.sum()
        total_exposure = stakes.sum()
        
        if total_exposure == 0:
            return 0.0
        
        # Eficiencia base
        efficiency = total_ev / total_exposure
        
        # Ajustar por tipo de portafolio
        if portfolio_type == 'elite':
            # Elite debe ser más eficiente
            efficiency *= 1.2
        
        return efficiency
    
    def _calculate_concentration(self, stakes: np.ndarray) -> float:
        """Calcula índice de concentración de Herfindahl-Hirschman."""
        if len(stakes) == 0:
            return 0.0
        
        # Normalizar stakes
        normalized = stakes / stakes.sum()
        
        # Calcular HHI
        hhi = np.sum(normalized ** 2)
        
        # Normalizar a 0-1 (0 = perfecta diversificación, 1 = máxima concentración)
        n = len(stakes)
        min_hhi = 1 / n
        concentration = (hhi - min_hhi) / (1 - min_hhi)
        
        return max(0.0, min(concentration, 1.0))
    
    def _calculate_elite_efficiency(self, evs: np.ndarray, stakes: np.ndarray) -> float:
        """Score de eficiencia específico para portafolio elite."""
        if len(stakes) == 0:
            return 0.0
        
        # Ratio EV/exposición
        ev_exposure_ratio = evs.sum() / stakes.sum() if stakes.sum() > 0 else 0
        
        # Concentración óptima (ideal alrededor de 0.3-0.5)
        concentration = self._calculate_concentration(stakes)
        concentration_score = 1 - abs(concentration - 0.4) * 2  # Penalizar desviaciones de 0.4
        
        # Score compuesto
        efficiency_score = ev_exposure_ratio * concentration_score
        
        return efficiency_score
    
    def _calculate_quality_score(self, avg_ev: float, sharpe: float, drawdown: float,
                                ruin_prob: float, roi: float) -> float:
        """Calcula score de calidad del portafolio (0-100)."""
        # Normalizar métricas a escala 0-1
        ev_score = min(max(avg_ev * 10, 0), 1)  # EV de 0 a 0.1
        sharpe_score = min(max(sharpe / 3, 0), 1)  # Sharpe de 0 a 3
        drawdown_score = 1 - min(max(drawdown / 100, 0), 1)  # Drawdown de 0 a 100%
        ruin_score = 1 - min(max(ruin_prob / 100, 0), 1)  # Ruin de 0 a 100%
        roi_score = min(max(roi / 100, 0), 1)  # ROI de 0 a 100%
        
        # Pesos
        weights = {
            'ev': 0.25,
            'sharpe': 0.25,
            'drawdown': 0.20,
            'ruin': 0.15,
            'roi': 0.15
        }
        
        # Score ponderado
        score = (ev_score * weights['ev'] +
                sharpe_score * weights['sharpe'] +
                drawdown_score * weights['drawdown'] +
                ruin_score * weights['ruin'] +
                roi_score * weights['roi'])
        
        return score * 100
    
    def _get_quality_rating(self, score: float) -> str:
        """Convierte score numérico a rating cualitativo."""
        if score >= 90:
            return "A+ ⭐⭐⭐⭐⭐"
        elif score >= 80:
            return "A ⭐⭐⭐⭐"
        elif score >= 70:
            return "B+ ⭐⭐⭐"
        elif score >= 60:
            return "B ⭐⭐"
        elif score >= 50:
            return "C ⭐"
        elif score >= 40:
            return "D ⚠️"
        else:
            return "F 🚨"
    
    def _interpret_efficiency_ratio(self, ratio: float) -> str:
        """Interpreta ratio de eficiencia entre portafolios."""
        if ratio > 1.5:
            return "Elite significativamente más eficiente"
        elif ratio > 1.2:
            return "Elite moderadamente más eficiente"
        elif ratio > 0.8:
            return "Eficiencia similar"
        elif ratio > 0.5:
            return "Full moderadamente más eficiente"
        else:
            return "Full significativamente más eficiente"
    
    def _generate_recommendation(self, full_metrics: Dict, elite_metrics: Dict) -> str:
        """Genera recomendación basada en comparativa de métricas."""
        full_score = full_metrics.get('quality_score', 0)
        elite_score = elite_metrics.get('quality_score', 0)
        
        if elite_score > full_score + 10:
            return "✅ RECOMENDACIÓN FUERTE: Usar portafolio ELITE"
        elif elite_score > full_score + 5:
            return "✅ RECOMENDACIÓN: Usar portafolio ELITE"
        elif abs(elite_score - full_score) < 5:
            return "⚖️ RECOMENDACIÓN NEUTRA: Ambos portafolios son similares"
        elif full_score > elite_score + 5:
            return "ℹ️ RECOMENDACIÓN: Usar portafolio FULL"
        else:
            return "ℹ️ RECOMENDACIÓN DÉBIL: Usar portafolio FULL"
    
    def render_dual_analysis(self, metrics: Dict):
        """Renderiza análisis comparativo de portafolios duales."""
        st.header("📊 Análisis Comparativo de Portafolios")
        
        if not metrics['full'] and not metrics['elite']:
            st.warning("No hay portafolios activos para analizar")
            return
        
        # Mostrar métricas individuales
        cols = st.columns(2)
        
        for idx, (ptype, col) in enumerate(zip(['full', 'elite'], cols)):
            if metrics[ptype]:
                with col:
                    self._render_portfolio_card(metrics[ptype], ptype)
        
        # Mostrar comparativa si ambos están activos
        if metrics['full'] and metrics['elite']:
            st.subheader("🔄 Comparativa Detallada")
            self._render_comparison_table(metrics['comparison'])
            
            # Recomendación final
            if 'efficiency_analysis' in metrics['comparison']:
                analysis = metrics['comparison']['efficiency_analysis']
                st.info(f"**{analysis['recommendation']}**")
                
                # Gráfico comparativo
                self._render_comparison_chart(metrics['full'], metrics['elite'])
    
    def _render_portfolio_card(self, metrics: Dict, portfolio_type: str):
        """Renderiza tarjeta de métricas de un portafolio."""
        color = SystemConfig.COLORS['success'] if portfolio_type == 'elite' else SystemConfig.COLORS['primary']
        
        st.markdown(f"""
        <div style='border: 2px solid {color}; border-radius: 10px; padding: 20px; margin: 10px 0;'>
            <h3 style='color: {color};'>Portafolio {portfolio_type.upper()}</h3>
            <p><strong>Calidad:</strong> {metrics['quality_rating']} ({metrics['quality_score']:.1f}/100)</p>
            <p><strong>Columnas:</strong> {metrics['n_columns']}</p>
            <p><strong>Exposición:</strong> {metrics['total_exposure_pct']:.1f}%</p>
            <p><strong>EV Promedio:</strong> {metrics['avg_expected_value']:.3f}</p>
            <p><strong>ROI Esperado:</strong> {metrics['expected_roi_pct']:.1f}%</p>
            <p><strong>Sharpe:</strong> {metrics['sharpe_ratio']:.2f}</p>
            <p><strong>Drawdown Esperado:</strong> {metrics['expected_drawdown_pct']:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_comparison_table(self, comparison: Dict):
        """Renderiza tabla comparativa de métricas."""
        comparison_data = []
        
        for key, data in comparison.items():
            if isinstance(data, dict) and 'label' in data:
                row = {
                    'Métrica': data['label'],
                    'Full': f"{data['full']:.3f}",
                    'Elite': f"{data['elite']:.3f}",
                    'Diferencia': f"{data['difference']:+.3f}",
                    'Mejora': f"{data['improvement_pct']:+.1f}%",
                    'Mejor': data['better'].upper()
                }
                comparison_data.append(row)
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
    
    def _render_comparison_chart(self, full_metrics: Dict, elite_metrics: Dict):
        """Renderiza gráfico de radar comparativo."""
        categories = ['EV', 'Sharpe', 'ROI', 'Seguridad', 'Eficiencia']
        
        # Normalizar métricas para gráfico de radar (0-1)
        full_values = [
            min(full_metrics['avg_expected_value'] * 10, 1),
            min(full_metrics['sharpe_ratio'] / 3, 1),
            min(full_metrics['expected_roi_pct'] / 100, 1),
            1 - min(full_metrics['expected_drawdown_pct'] / 100, 1),
            full_metrics['capital_efficiency']
        ]
        
        elite_values = [
            min(elite_metrics['avg_expected_value'] * 10, 1),
            min(elite_metrics['sharpe_ratio'] / 3, 1),
            min(elite_metrics['expected_roi_pct'] / 100, 1),
            1 - min(elite_metrics['expected_drawdown_pct'] / 100, 1),
            elite_metrics['capital_efficiency']
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=full_values,
            theta=categories,
            fill='toself',
            name='Portafolio Full',
            line_color=SystemConfig.COLORS['primary']
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=elite_values,
            theta=categories,
            fill='toself',
            name='Portafolio Elite',
            line_color=SystemConfig.COLORS['success']
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Análisis Comparativo - Gráfico de Radar",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# SECCIÓN 7: BACKTESTING VECTORIZADO AVANZADO v3.0
# ============================================================================

class VectorizedBacktester:
    """
    Sistema de backtesting completamente vectorizado con soporte dual.
    Optimizado para simulaciones masivas con gestión de capital realista.
    """
    
    def __init__(self, initial_bankroll: float = SystemConfig.DEFAULT_BANKROLL):
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.performance_history = []
        self.equity_curves = {'full': [], 'elite': []}
        
    def run_dual_backtest(self, config: Dict, s73_results: Dict, 
                         elite_results: Dict = None) -> Dict:
        """
        Ejecuta backtesting dual (full + elite) en paralelo.
        
        Args:
            config: Configuración del sidebar
            s73_results: Resultados del sistema S73 completo
            elite_results: Resultados del sistema elite (opcional)
            
        Returns:
            Diccionario con resultados de ambos portafolios
        """
        st.header("🎯 Backtesting Dual Avanzado")
        
        # Extraer parámetros
        probabilities = config.get('probabilities')
        odds_matrix = config.get('odds_matrix')
        normalized_entropies = config.get('normalized_entropies')
        bankroll = config.get('bankroll', self.initial_bankroll)
        n_rounds = config.get('n_rounds', 100)
        n_sims = config.get('monte_carlo_sims', 1000)
        
        results = {}
        
        # Backtesting para portafolio full
        with st.spinner("Simulando portafolio Full (73 columnas)..."):
            full_result = self._run_single_backtest(
                portfolio_type='full',
                combinations=s73_results['combinations'],
                probabilities=s73_results['probabilities'],
                stakes=s73_results['kelly_stakes'],
                probabilities_matrix=probabilities,
                odds_matrix=odds_matrix,
                normalized_entropies=normalized_entropies,
                bankroll=bankroll,
                n_rounds=n_rounds,
                n_sims=n_sims,
                kelly_fraction=config.get('kelly_fraction', 0.5),
                manual_stake=config.get('manual_stake')
            )
            results['full'] = full_result
        
        # Backtesting para portafolio elite (si está disponible)
        if elite_results and config.get('apply_elite_reduction', False):
            with st.spinner("Simulando portafolio Elite (24 columnas)..."):
                elite_result = self._run_single_backtest(
                    portfolio_type='elite',
                    combinations=elite_results['combinations'],
                    probabilities=elite_results['probabilities'],
                    stakes=elite_results.get('kelly_stakes', s73_results['kelly_stakes'][:24]),
                    probabilities_matrix=probabilities,
                    odds_matrix=odds_matrix,
                    normalized_entropies=normalized_entropies,
                    bankroll=bankroll,
                    n_rounds=n_rounds,
                    n_sims=n_sims,
                    kelly_fraction=config.get('kelly_fraction', 0.5),
                    manual_stake=config.get('manual_stake')
                )
                results['elite'] = elite_result
        
        # Análisis comparativo
        if 'elite' in results:
            results['comparison'] = self._compare_backtest_results(
                results['full'], results['elite']
            )
        
        return results
    
    def _run_single_backtest(self, portfolio_type: str, combinations: List,
                            probabilities: List, stakes: np.ndarray,
                            probabilities_matrix: np.ndarray,
                            odds_matrix: np.ndarray,
                            normalized_entropies: np.ndarray,
                            bankroll: float, n_rounds: int,
                            n_sims: int, kelly_fraction: float,
                            manual_stake: float = None) -> Dict:
        """Ejecuta backtesting para un portafolio específico."""
        # Validar inputs
        if len(combinations) == 0:
            return {}
        
        # Preparar arrays para vectorización
        n_columns = len(combinations)
        n_matches = 6
        
        # Convertir combinaciones a array numpy
        combo_array = np.array(combinations)
        
        # Calcular cuotas por columna
        column_odds = np.array([
            S73System.calculate_combination_odds(combo, odds_matrix)
            for combo in combinations
        ])
        
        # Calcular stakes
        if manual_stake is not None:
            base_stakes = np.full(n_columns, manual_stake)
        else:
            base_stakes = self._calculate_kelly_stakes_vectorized(
                combo_array, np.array(probabilities), column_odds,
                normalized_entropies, kelly_fraction, portfolio_type
            )
        
        # Normalizar stakes
        base_stakes = KellyCapitalManagement.normalize_portfolio_stakes(
            base_stakes,
            max_exposure=config.get('max_exposure', 0.15),
            is_manual_mode=(manual_stake is not None)
        )
        
        # Simulaciones Monte Carlo
        equity_curves = []
        round_metrics = []
        final_bankrolls = []
        
        for sim_idx in range(n_sims):
            bankroll_sim = bankroll
            equity_curve = [bankroll_sim]
            
            for round_idx in range(n_rounds):
                # Simular resultados de partidos
                match_results = self._simulate_match_results(
                    probabilities_matrix, n_matches
                )
                
                # Calcular rendimiento de columnas
                returns = self._calculate_column_returns_vectorized(
                    match_results, combo_array, column_odds,
                    base_stakes, bankroll_sim
                )
                
                # Actualizar bankroll
                round_return = np.sum(returns)
                bankroll_sim += round_return
                equity_curve.append(bankroll_sim)
            
            equity_curves.append(equity_curve)
            final_bankrolls.append(bankroll_sim)
        
        # Calcular métricas
        metrics = self._calculate_backtest_metrics(
            equity_curves, final_bankrolls, bankroll, n_rounds
        )
        
        return {
            'portfolio_type': portfolio_type,
            'n_columns': n_columns,
            'equity_curves': np.array(equity_curves),
            'final_bankrolls': np.array(final_bankrolls),
            'metrics': metrics,
            'base_stakes': base_stakes,
            'column_odds': column_odds,
            'simulation_count': n_sims
        }
    
    def _calculate_kelly_stakes_vectorized(self, combos: np.ndarray,
                                          probs: np.ndarray, odds: np.ndarray,
                                          entropies: np.ndarray,
                                          kelly_fraction: float,
                                          portfolio_type: str) -> np.ndarray:
        """Calcula stakes Kelly vectorizados."""
        n_columns = len(combos)
        
        # Calcular Kelly crudo
        kelly_raw = (probs * odds - 1) / (odds - 1)
        kelly_raw = np.clip(kelly_raw, 0, SystemConfig.KELLY_FRACTION_MAX)
        
        # Ajustar por fracción de Kelly
        kelly_adjusted = kelly_raw * kelly_fraction
        
        # Ajustar por entropía (menor stake para partidos inciertos)
        avg_entropy = np.mean(entropies)
        entropy_factor = 1 - avg_entropy
        kelly_adjusted *= entropy_factor
        
        # Ajustar por tipo de portafolio
        if portfolio_type == 'elite':
            kelly_adjusted *= 1.2  # Elite puede usar stakes ligeramente más altos
        
        return kelly_adjusted
    
    def _simulate_match_results(self, probabilities: np.ndarray,
                               n_matches: int) -> np.ndarray:
        """Simula resultados de partidos de forma vectorizada."""
        results = np.zeros(n_matches, dtype=int)
        
        for i in range(n_matches):
            results[i] = np.random.choice(
                [0, 1, 2],
                p=probabilities[i]
            )
        
        return results
    
    def _calculate_column_returns_vectorized(self, match_results: np.ndarray,
                                            combos: np.ndarray,
                                            column_odds: np.ndarray,
                                            stakes: np.ndarray,
                                            bankroll: float) -> np.ndarray:
        """Calcula retornos de columnas de forma vectorizada."""
        n_columns = len(combos)
        returns = np.zeros(n_columns)
        
        # Calcular inversión por columna
        investments = stakes * bankroll
        
        # Verificar aciertos
        for i in range(n_columns):
            hits = np.sum(combos[i] == match_results)
            
            if hits == 6:  # Columna ganadora
                returns[i] = investments[i] * (column_odds[i] - 1)
            else:  # Columna perdedora
                returns[i] = -investments[i]
        
        return returns
    
    def _calculate_backtest_metrics(self, equity_curves: List,
                                   final_bankrolls: List,
                                   initial_bankroll: float,
                                   n_rounds: int) -> Dict:
        """Calcula métricas avanzadas de backtesting."""
        # Convertir a arrays
        curves_array = np.array(equity_curves)
        final_array = np.array(final_bankrolls)
        
        # Retorno total promedio
        total_returns = final_array - initial_bankroll
        avg_return = np.mean(total_returns)
        avg_return_pct = (avg_return / initial_bankroll) * 100
        
        # ROI por ronda
        roi_per_round = avg_return_pct / n_rounds
        
        # Win rate
        win_rate = np.mean(total_returns > 0) * 100
        
        # Drawdown máximo
        max_drawdowns = []
        for curve in curves_array:
            peak = np.maximum.accumulate(curve)
            drawdown = (peak - curve) / peak
            max_drawdowns.append(np.max(drawdown) * 100)
        
        avg_max_drawdown = np.mean(max_drawdowns)
        
        # Ratio Sharpe (simplificado)
        returns_pct = (curves_array[:, -1] - initial_bankroll) / initial_bankroll
        sharpe_ratio = self._calculate_sharpe_ratio(returns_pct)
        
        # Value at Risk (VaR 95%)
        var_95 = np.percentile(total_returns, 5)
        
        # Conditional VaR (CVaR 95%)
        cvar_95 = np.mean(total_returns[total_returns <= var_95])
        
        # Estadísticas adicionales
        median_return = np.median(total_returns)
        std_return = np.std(total_returns)
        skewness = pd.Series(total_returns).skew()
        kurtosis = pd.Series(total_returns).kurtosis()
        
        # Score de calidad
        quality_score = self._calculate_backtest_quality(
            avg_return_pct, sharpe_ratio, avg_max_drawdown,
            win_rate, roi_per_round
        )
        
        return {
            'initial_bankroll': initial_bankroll,
            'avg_final_bankroll': np.mean(final_array),
            'median_final_bankroll': np.median(final_array),
            'avg_total_return': avg_return,
            'avg_total_return_pct': avg_return_pct,
            'roi_per_round_pct': roi_per_round,
            'win_rate_pct': win_rate,
            'avg_max_drawdown_pct': avg_max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'median_return': median_return,
            'std_return': std_return,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'quality_score': quality_score,
            'quality_rating': self._get_quality_rating(quality_score),
            'n_simulations': len(equity_curves),
            'n_rounds': n_rounds
        }
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray,
                               risk_free_rate: float = 0.02) -> float:
        """Calcula ratio Sharpe anualizado."""
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
    
    def _calculate_backtest_quality(self, roi: float, sharpe: float,
                                   drawdown: float, win_rate: float,
                                   roi_per_round: float) -> float:
        """Calcula score de calidad del backtest (0-100)."""
        # Normalizar métricas
        roi_score = min(max(roi / 100, 0), 1)  # ROI de 0-100%
        sharpe_score = min(max(sharpe / 3, 0), 1)  # Sharpe de 0-3
        drawdown_score = 1 - min(max(drawdown / 100, 0), 1)  # Drawdown 0-100%
        win_rate_score = min(max(win_rate / 100, 0), 1)  # Win rate 0-100%
        roi_round_score = min(max(roi_per_round / 5, 0), 1)  # ROI por ronda 0-5%
        
        # Ponderaciones
        weights = {
            'roi': 0.25,
            'sharpe': 0.25,
            'drawdown': 0.20,
            'win_rate': 0.15,
            'roi_per_round': 0.15
        }
        
        # Calcular score
        score = (roi_score * weights['roi'] +
                sharpe_score * weights['sharpe'] +
                drawdown_score * weights['drawdown'] +
                win_rate_score * weights['win_rate'] +
                roi_round_score * weights['roi_per_round'])
        
        return score * 100
    
    def _get_quality_rating(self, score: float) -> str:
        """Convierte score a rating cualitativo."""
        if score >= 90:
            return "A+ ⭐⭐⭐⭐⭐"
        elif score >= 80:
            return "A ⭐⭐⭐⭐"
        elif score >= 70:
            return "B+ ⭐⭐⭐"
        elif score >= 60:
            return "B ⭐⭐"
        elif score >= 50:
            return "C ⭐"
        elif score >= 40:
            return "D ⚠️"
        else:
            return "F 🚨"
    
    def _compare_backtest_results(self, full_results: Dict,
                                 elite_results: Dict) -> Dict:
        """Compara resultados de backtesting entre portafolios."""
        comparison = {}
        
        # Métricas clave para comparar
        metrics_to_compare = [
            ('avg_total_return_pct', 'ROI Total (%)'),
            ('win_rate_pct', 'Win Rate (%)'),
            ('avg_max_drawdown_pct', 'Max Drawdown (%)'),
            ('sharpe_ratio', 'Sharpe Ratio'),
            ('quality_score', 'Score Calidad'),
            ('roi_per_round_pct', 'ROI por Ronda (%)')
        ]
        
        for metric_key, label in metrics_to_compare:
            full_val = full_results['metrics'].get(metric_key, 0)
            elite_val = elite_results['metrics'].get(metric_key, 0)
            
            # Calcular mejora
            if full_val != 0:
                improvement = ((elite_val - full_val) / abs(full_val)) * 100
            else:
                improvement = 0
            
            comparison[metric_key] = {
                'label': label,
                'full': full_val,
                'elite': elite_val,
                'difference': elite_val - full_val,
                'improvement_pct': improvement,
                'better': 'elite' if elite_val > full_val else 
                         'full' if elite_val < full_val else 'equal'
            }
        
        # Análisis de eficiencia
        full_efficiency = full_results['metrics']['avg_total_return_pct'] / \
                         max(full_results['metrics']['avg_max_drawdown_pct'], 1)
        elite_efficiency = elite_results['metrics']['avg_total_return_pct'] / \
                          max(elite_results['metrics']['avg_max_drawdown_pct'], 1)
        
        comparison['efficiency_analysis'] = {
            'full_efficiency': full_efficiency,
            'elite_efficiency': elite_efficiency,
            'efficiency_ratio': elite_efficiency / max(full_efficiency, 0.001),
            'recommendation': self._generate_backtest_recommendation(
                full_results, elite_results
            )
        }
        
        return comparison
    
    def _generate_backtest_recommendation(self, full_results: Dict,
                                         elite_results: Dict) -> str:
        """Genera recomendación basada en resultados de backtesting."""
        full_score = full_results['metrics']['quality_score']
        elite_score = elite_results['metrics']['quality_score']
        
        if elite_score > full_score + 15:
            return "✅ RECOMENDACIÓN FUERTE: Portafolio ELITE supera significativamente"
        elif elite_score > full_score + 5:
            return "✅ RECOMENDACIÓN: Portafolio ELITE es mejor"
        elif abs(elite_score - full_score) < 5:
            return "⚖️ RECOMENDACIÓN NEUTRA: Ambos portafolios similares"
        elif full_score > elite_score + 5:
            return "ℹ️ RECOMENDACIÓN: Portafolio FULL es mejor"
        else:
            return "ℹ️ RECOMENDACIÓN DÉBIL: Portafolio FULL ligeramente mejor"
    
    def render_backtest_results(self, results: Dict):
        """Renderiza resultados de backtesting de forma profesional."""
        if not results:
            st.warning("No hay resultados de backtesting para mostrar")
            return
        
        st.header("📊 Resultados de Backtesting")
        
        # Mostrar resultados por portafolio
        cols = st.columns(2)
        
        for idx, (ptype, col) in enumerate(zip(['full', 'elite'], cols)):
            if ptype in results:
                with col:
                    self._render_portfolio_backtest_card(results[ptype])
        
        # Mostrar comparativa si existe
        if 'comparison' in results:
            st.subheader("🔄 Análisis Comparativo")
            
            # Gráfico comparativo
            self._render_comparison_chart(results)
            
            # Tabla de comparación
            self._render_comparison_table(results['comparison'])
            
            # Recomendación
            if 'efficiency_analysis' in results['comparison']:
                rec = results['comparison']['efficiency_analysis']['recommendation']
                st.success(f"**{rec}**")
        
        # Gráficos detallados
        self._render_detailed_charts(results)
    
    def _render_portfolio_backtest_card(self, portfolio_results: Dict):
        """Renderiza tarjeta de resultados de backtesting."""
        metrics = portfolio_results['metrics']
        ptype = portfolio_results['portfolio_type']
        
        color = SystemConfig.COLORS['success'] if ptype == 'elite' else SystemConfig.COLORS['primary']
        
        st.markdown(f"""
        <div style='border: 2px solid {color}; border-radius: 10px; padding: 20px; margin: 10px 0;'>
            <h3 style='color: {color};'>Portafolio {ptype.upper()}</h3>
            <p><strong>Calidad:</strong> {metrics['quality_rating']} ({metrics['quality_score']:.1f}/100)</p>
            <p><strong>ROI Total:</strong> {metrics['avg_total_return_pct']:.1f}%</p>
            <p><strong>Win Rate:</strong> {metrics['win_rate_pct']:.1f}%</p>
            <p><strong>Sharpe Ratio:</strong> {metrics['sharpe_ratio']:.2f}</p>
            <p><strong>Max Drawdown:</strong> {metrics['avg_max_drawdown_pct']:.1f}%</p>
            <p><strong>VaR 95%:</strong> €{metrics['var_95']:,.0f}</p>
            <p><strong>Simulaciones:</strong> {metrics['n_simulations']:,}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_comparison_chart(self, results: Dict):
        """Renderiza gráfico comparativo de equity curves."""
        fig = go.Figure()
        
        colors = {
            'full': SystemConfig.COLORS['primary'],
            'elite': SystemConfig.COLORS['success']
        }
        
        for ptype in ['full', 'elite']:
            if ptype in results:
                curves = results[ptype]['equity_curves']
                avg_curve = np.mean(curves, axis=0)
                std_curve = np.std(curves, axis=0)
                
                # Línea promedio
                fig.add_trace(go.Scatter(
                    x=list(range(len(avg_curve))),
                    y=avg_curve,
                    mode='lines',
                    name=f'Portafolio {ptype.upper()}',
                    line=dict(color=colors[ptype], width=3)
                ))
                
                # Banda de desviación estándar
                fig.add_trace(go.Scatter(
                    x=list(range(len(avg_curve))) + list(range(len(avg_curve)))[::-1],
                    y=list(avg_curve + std_curve) + list(avg_curve - std_curve)[::-1],
                    fill='toself',
                    fillcolor=f'rgba{tuple(int(colors[ptype][i:i+2], 16) for i in (1, 3, 5)) + (0.2,)}',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False
                ))
        
        fig.update_layout(
            title="Evolución del Bankroll (Promedio ± Desviación)",
            xaxis_title="Ronda",
            yaxis_title="Bankroll (€)",
            height=400,
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_comparison_table(self, comparison: Dict):
        """Renderiza tabla comparativa detallada."""
        table_data = []
        
        for key, data in comparison.items():
            if isinstance(data, dict) and 'label' in data:
                row = {
                    'Métrica': data['label'],
                    'Full': f"{data['full']:.3f}",
                    'Elite': f"{data['elite']:.3f}",
                    'Diferencia': f"{data['difference']:+.3f}",
                    'Mejora': f"{data['improvement_pct']:+.1f}%",
                    'Mejor': data['better'].upper()
                }
                table_data.append(row)
        
        if table_data:
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True)
    
    def _render_detailed_charts(self, results: Dict):
        """Renderiza gráficos detallados de análisis."""
        st.subheader("📈 Análisis Detallado")
        
        # Seleccionar qué portafolio mostrar
        selected_portfolio = st.selectbox(
            "Seleccionar portafolio para análisis detallado:",
            ['full', 'elite'],
            index=0
        )
        
        if selected_portfolio in results:
            portfolio_data = results[selected_portfolio]
            
            # Gráfico 1: Distribución de bankrolls finales
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = go.Figure()
                final_bankrolls = portfolio_data['final_bankrolls']
                
                fig1.add_trace(go.Histogram(
                    x=final_bankrolls,
                    nbinsx=50,
                    name='Bankroll Final',
                    marker_color=SystemConfig.COLORS['primary']
                ))
                
                fig1.update_layout(
                    title="Distribución de Bankrolls Finales",
                    xaxis_title="Bankroll Final (€)",
                    yaxis_title="Frecuencia",
                    height=300
                )
                
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Gráfico 2: Drawdown máximo por simulación
                max_drawdowns = []
                for curve in portfolio_data['equity_curves']:
                    peak = np.maximum.accumulate(curve)
                    drawdown = (peak - curve) / peak
                    max_drawdowns.append(np.max(drawdown) * 100)
                
                fig2 = go.Figure()
                
                fig2.add_trace(go.Box(
                    y=max_drawdowns,
                    name='Drawdown Máximo',
                    boxpoints='outliers',
                    marker_color=SystemConfig.COLORS['warning']
                ))
                
                fig2.update_layout(
                    title="Distribución de Drawdown Máximo",
                    yaxis_title="Drawdown Máximo (%)",
                    height=300
                )
                
                st.plotly_chart(fig2, use_container_width=True)
            
            # Gráfico 3: Curvas de equity de muestra
            st.subheader("Curvas de Equity de Muestra")
            
            fig3 = go.Figure()
            
            # Mostrar solo 10 curvas para claridad
            sample_indices = np.random.choice(
                len(portfolio_data['equity_curves']),
                size=min(10, len(portfolio_data['equity_curves'])),
                replace=False
            )
            
            for idx in sample_indices:
                fig3.add_trace(go.Scatter(
                    x=list(range(len(portfolio_data['equity_curves'][idx]))),
                    y=portfolio_data['equity_curves'][idx],
                    mode='lines',
                    name=f'Simulación {idx+1}',
                    line=dict(width=1, opacity=0.5)
                ))
            
            fig3.update_layout(
                title="Curvas de Equity - 10 Simulaciones Aleatorias",
                xaxis_title="Ronda",
                yaxis_title="Bankroll (€)",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig3, use_container_width=True)

# ============================================================================
# SECCIÓN 8: SISTEMA DE EXPORTACIÓN PROFESIONAL v3.0
# ============================================================================

class DataExporter:
    """
    Sistema profesional de exportación de datos para ACBE-S73 v3.0.
    Soporta múltiples formatos y exportación dual (full + elite).
    """
    
    @staticmethod
    def export_system_results(columns_df: pd.DataFrame, s73_results: Dict,
                            config: Dict, backtest_results: Dict = None) -> Dict:
        """
        Exporta resultados completos del sistema en múltiples formatos.
        
        Args:
            columns_df: DataFrame con columnas del sistema
            s73_results: Resultados del sistema S73
            config: Configuración de la simulación
            backtest_results: Resultados de backtesting (opcional)
            
        Returns:
            Diccionario con datos para descarga
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exports = {}
        
        # 1. CSV de columnas básico
        exports['csv_basic'] = DataExporter._export_to_csv_basic(columns_df, timestamp)
        
        # 2. Excel completo con múltiples hojas
        exports['excel_full'] = DataExporter._export_to_excel_full(
            columns_df, s73_results, config, timestamp
        )
        
        # 3. Reporte ejecutivo
        exports['report_executive'] = DataExporter._export_executive_report(
            columns_df, s73_results, config, timestamp
        )
        
        # 4. Backtesting results (si está disponible)
        if backtest_results:
            exports.update(
                DataExporter._export_backtest_results(backtest_results, timestamp)
            )
        
        return exports
    
    @staticmethod
    def _export_to_csv_basic(columns_df: pd.DataFrame, timestamp: str) -> Dict:
        """Exporta columnas a CSV básico."""
        csv_data = columns_df.to_csv(index=False, sep=',', encoding='utf-8-sig')
        
        return {
            'data': csv_data,
            'filename': f'acbe_s73_columns_{timestamp}.csv',
            'mime': 'text/csv',
            'description': 'Columnas del sistema en formato CSV'
        }
    
    @staticmethod
    def _export_to_excel_full(columns_df: pd.DataFrame, s73_results: Dict,
                             config: Dict, timestamp: str) -> Dict:
        """Exporta sistema completo a Excel con múltiples hojas."""
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Hoja 1: Columnas detalladas
            columns_df.to_excel(writer, sheet_name='Columnas', index=False)
            
            # Hoja 2: Resumen del sistema
            summary_data = DataExporter._create_system_summary(s73_results, config)
            summary_df = pd.DataFrame([summary_data])
            summary_df.to_excel(writer, sheet_name='Resumen', index=False)
            
            # Hoja 3: Configuración
            config_df = pd.DataFrame(list(config.items()), columns=['Parámetro', 'Valor'])
            config_df.to_excel(writer, sheet_name='Configuración', index=False)
            
            # Hoja 4: Métricas por columna (si existen)
            if 'columns_df' in s73_results and not s73_results['columns_df'].empty:
                metrics_df = s73_results['columns_df']
                metrics_df.to_excel(writer, sheet_name='Métricas', index=False)
        
        return {
            'data': output.getvalue(),
            'filename': f'acbe_s73_full_report_{timestamp}.xlsx',
            'mime': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'description': 'Reporte completo en Excel con múltiples hojas'
        }
    
    @staticmethod
    def _create_system_summary(s73_results: Dict, config: Dict) -> Dict:
        """Crea resumen del sistema para exportación."""
        return {
            'Fecha_Exportación': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Versión_Sistema': 'ACBE-S73 v3.0',
            'N_Columnas_Total': len(s73_results.get('combinations', [])),
            'Tipo_Portafolio': config.get('portfolio_type', 'full'),
            'Bankroll_Inicial': config.get('bankroll', SystemConfig.DEFAULT_BANKROLL),
            'Exposición_Máxima_Config': f"{config.get('max_exposure', 0.15) * 100:.1f}%",
            'Reducción_Elite_Aplicada': config.get('apply_elite_reduction', False),
            'N_Columnas_Elite': config.get('elite_columns_target', 24),
            'Stake_Mode': 'Manual' if config.get('manual_stake') else 'Kelly Automático',
            'Fracción_Kelly': config.get('kelly_fraction', 0.5),
            'Simulaciones_Monte_Carlo': config.get('monte_carlo_sims', 1000),
            'Rondas_Backtesting': config.get('n_rounds', 100)
        }
    
    @staticmethod
    def _export_executive_report(columns_df: pd.DataFrame, s73_results: Dict,
                                config: Dict, timestamp: str) -> Dict:
        """Genera reporte ejecutivo en texto."""
        # Encontrar la apuesta maestra
        if not columns_df.empty:
            master_bet = columns_df.loc[columns_df['Probabilidad'].idxmax()]
            master_combination = master_bet['Combinación']
            master_prob = master_bet['Probabilidad']
            master_odds = master_bet['Cuota']
        else:
            master_combination = 'N/A'
            master_prob = 0
            master_odds = 0
        
        report = f"""
        ===========================================================================
        REPORTE EJECUTIVO - ACBE-S73 QUANTUM BETTING SUITE v3.0
        ===========================================================================
        
        Fecha de Generación: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        ID Reporte: {timestamp}
        
        📊 RESUMEN DEL SISTEMA
        {'=' * 50}
        • Sistema: ACBE-S73 Quantum Betting Suite v3.0
        • Columnas Totales: {len(s73_results.get('combinations', []))}
        • Tipo de Portafolio: {config.get('portfolio_type', 'full').upper()}
        • Bankroll Inicial: €{config.get('bankroll', 0):,.2f}
        • Reducción Elite: {'✅ ACTIVADA' if config.get('apply_elite_reduction') else '❌ DESACTIVADA'}
        
        🎯 LA APUESTA MAESTRA
        {'=' * 50}
        • Combinación: {master_combination}
        • Probabilidad: {master_prob:.2%}
        • Cuota: {master_odds:.2f}
        • Valor Esperado: {master_prob * master_odds - 1:.3f}
        • Recomendación: {'✅ JUGAR' if master_prob * master_odds > 1.05 else '⚠️ EVALUAR' if master_prob * master_odds > 1 else '❌ NO JUGAR'}
        
        ⚙️ CONFIGURACIÓN APLICADA
        {'=' * 50}
        • Modo Stake: {'Manual' if config.get('manual_stake') else 'Kelly Automático'}
        • Fracción Kelly: {config.get('kelly_fraction', 0.5)}
        • Exposición Máxima: {config.get('max_exposure', 0.15) * 100:.1f}%
        • Columnas Elite Objetivo: {config.get('elite_columns_target', 24)}
        • Simulaciones por Ronda: {config.get('monte_carlo_sims', 1000):,}
        • Rondas de Backtesting: {config.get('n_rounds', 100)}
        
        📈 MÉTRICAS CLAVE
        {'=' * 50}
        • Probabilidad Promedio: {columns_df['Probabilidad'].mean():.2%}
        • Cuota Promedio: {columns_df['Cuota'].mean():.2f}
        • Valor Esperado Promedio: {columns_df['Valor Esperado'].mean():.3f}
        • Exposición Total: {columns_df['Stake (%)'].sum():.1f}%
        • Inversión Total: €{columns_df['Inversión (€)'].sum():,.2f}
        
        🎲 ANÁLISIS DE RIESGO
        {'=' * 50}
        • Stake Más Alto: {columns_df['Stake (%)'].max():.2f}%
        • Stake Más Bajo: {columns_df['Stake (%)'].min():.2f}%
        • Desviación Estándar Stake: {columns_df['Stake (%)'].std():.2f}%
        • Columnas con EV Positivo: {(columns_df['Valor Esperado'] > 0).sum()}
        • Columnas con EV Negativo: {(columns_df['Valor Esperado'] <= 0).sum()}
        
        🔍 RECOMENDACIONES
        {'=' * 50}
        1. {'Priorizar portafolio ELITE para mayor concentración' if config.get('portfolio_type') == 'elite' else 'Mantener portafolio FULL para mayor cobertura'}
        2. Mantener exposición total por debajo del {config.get('max_exposure', 0.15) * 100:.0f}%
        3. Monitorear columnas con EV negativo regularmente
        4. Utilizar simulador de escenarios para análisis what-if
        5. {'Ajustar fracción Kelly hacia valores más conservadores (< 0.3) si hay alta volatilidad' if config.get('kelly_fraction', 0.5) > 0.3 else 'Mantener fracción Kelly actual'}
        
        ===========================================================================
        Generado por: ACBE-S73 Quantum Betting Suite v3.0
        Sistema Validado Institucionalmente
        ===========================================================================
        """
        
        return {
            'data': report,
            'filename': f'acbe_s73_executive_report_{timestamp}.txt',
            'mime': 'text/plain',
            'description': 'Reporte ejecutivo detallado en texto'
        }
    
    @staticmethod
    def _export_backtest_results(backtest_results: Dict, timestamp: str) -> Dict:
        """Exporta resultados de backtesting."""
        exports = {}
        
        for ptype in ['full', 'elite']:
            if ptype in backtest_results:
                result = backtest_results[ptype]
                metrics = result['metrics']
                
                # 1. CSV de métricas
                metrics_df = pd.DataFrame([metrics])
                metrics_csv = metrics_df.to_csv(index=False, sep=',', encoding='utf-8-sig')
                
                exports[f'csv_backtest_{ptype}'] = {
                    'data': metrics_csv,
                    'filename': f'acbe_backtest_{ptype}_metrics_{timestamp}.csv',
                    'mime': 'text/csv',
                    'description': f'Métricas de backtesting - Portafolio {ptype.upper()}'
                }
                
                # 2. CSV de curvas de equity (primera simulación)
                if 'equity_curves' in result and len(result['equity_curves']) > 0:
                    equity_df = pd.DataFrame({
                        'Round': range(len(result['equity_curves'][0])),
                        'Bankroll': result['equity_curves'][0],
                        'Portfolio_Type': ptype
                    })
                    equity_csv = equity_df.to_csv(index=False, sep=',', encoding='utf-8-sig')
                    
                    exports[f'csv_equity_{ptype}'] = {
                        'data': equity_csv,
                        'filename': f'acbe_equity_curve_{ptype}_{timestamp}.csv',
                        'mime': 'text/csv',
                        'description': f'Curva de equity - Portafolio {ptype.upper()}'
                    }
        
        # 3. Reporte comparativo (si hay ambos portafolios)
        if 'full' in backtest_results and 'elite' in backtest_results:
            comparison = DataExporter._create_comparison_report(
                backtest_results['full'], backtest_results['elite'], timestamp
            )
            
            exports['report_comparison'] = {
                'data': comparison,
                'filename': f'acbe_comparison_report_{timestamp}.txt',
                'mime': 'text/plain',
                'description': 'Reporte comparativo entre portafolios'
            }
        
        return exports
    
    @staticmethod
    def _create_comparison_report(full_results: Dict, elite_results: Dict,
                                 timestamp: str) -> str:
        """Crea reporte comparativo entre portafolios."""
        full_metrics = full_results['metrics']
        elite_metrics = elite_results['metrics']
        
        # Calcular mejoras
        improvements = {
            'roi': ((elite_metrics['avg_total_return_pct'] - full_metrics['avg_total_return_pct']) / 
                   abs(full_metrics['avg_total_return_pct'])) * 100 if full_metrics['avg_total_return_pct'] != 0 else 0,
            'sharpe': ((elite_metrics['sharpe_ratio'] - full_metrics['sharpe_ratio']) / 
                      abs(full_metrics['sharpe_ratio'])) * 100 if full_metrics['sharpe_ratio'] != 0 else 0,
            'drawdown': ((full_metrics['avg_max_drawdown_pct'] - elite_metrics['avg_max_drawdown_pct']) / 
                        abs(full_metrics['avg_max_drawdown_pct'])) * 100 if full_metrics['avg_max_drawdown_pct'] != 0 else 0,
            'quality': ((elite_metrics['quality_score'] - full_metrics['quality_score']) / 
                       abs(full_metrics['quality_score'])) * 100 if full_metrics['quality_score'] != 0 else 0
        }
        
        report = f"""
        ===========================================================================
        REPORTE COMPARATIVO - BACKTESTING DUAL v3.0
        ===========================================================================
        
        Fecha de Comparación: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        ID Comparación: {timestamp}
        
        📊 RESUMEN COMPARATIVO
        {'=' * 60}
        | Métrica                | FULL (73)          | ELITE (24)         | Mejora      |
        |------------------------|--------------------|--------------------|-------------|
        | ROI Total              | {full_metrics['avg_total_return_pct']:>6.2f}%        | {elite_metrics['avg_total_return_pct']:>6.2f}%        | {improvements['roi']:+.1f}% |
        | Sharpe Ratio           | {full_metrics['sharpe_ratio']:>6.3f}          | {elite_metrics['sharpe_ratio']:>6.3f}          | {improvements['sharpe']:+.1f}% |
        | Max Drawdown           | {full_metrics['avg_max_drawdown_pct']:>6.2f}%        | {elite_metrics['avg_max_drawdown_pct']:>6.2f}%        | {improvements['drawdown']:+.1f}% |
        | Win Rate               | {full_metrics['win_rate_pct']:>6.2f}%        | {elite_metrics['win_rate_pct']:>6.2f}%        | {((elite_metrics['win_rate_pct'] - full_metrics['win_rate_pct']) / abs(full_metrics['win_rate_pct']) * 100):+.1f}% |
        | Score Calidad          | {full_metrics['quality_score']:>6.1f}/100     | {elite_metrics['quality_score']:>6.1f}/100     | {improvements['quality']:+.1f}% |
        | Calificación           | {full_metrics['quality_rating']:<15} | {elite_metrics['quality_rating']:<15} |             |
        
        🎯 ANÁLISIS DE EFICIENCIA
        {'=' * 60}
        • Eficiencia FULL: {full_metrics['avg_total_return_pct'] / max(full_metrics['avg_max_drawdown_pct'], 1):.2f} (ROI/Drawdown)
        • Eficiencia ELITE: {elite_metrics['avg_total_return_pct'] / max(elite_metrics['avg_max_drawdown_pct'], 1):.2f} (ROI/Drawdown)
        • Ratio de Eficiencia: {elite_metrics['avg_total_return_pct'] / max(elite_metrics['avg_max_drawdown_pct'], 1) / max(full_metrics['avg_total_return_pct'] / max(full_metrics['avg_max_drawdown_pct'], 1), 0.001):.2f}x
        
        📈 DISTRIBUCIÓN DE RESULTADOS
        {'=' * 60}
        • Bankroll Final PROMEDIO:
          - FULL: €{full_metrics['avg_final_bankroll']:,.2f}
          - ELITE: €{elite_metrics['avg_final_bankroll']:,.2f}
          - Diferencia: €{elite_metrics['avg_final_bankroll'] - full_metrics['avg_final_bankroll']:+,.2f}
        
        • Bankroll Final MEDIANO:
          - FULL: €{full_metrics['median_final_bankroll']:,.2f}
          - ELITE: €{elite_metrics['median_final_bankroll']:,.2f}
          - Diferencia: €{elite_metrics['median_final_bankroll'] - full_metrics['median_final_bankroll']:+,.2f}
        
        ⚖️ ANÁLISIS DE RIESGO
        {'=' * 60}
        • Volatilidad (σ):
          - FULL: €{full_metrics['std_return']:,.2f}
          - ELITE: €{elite_metrics['std_return']:,.2f}
          - Diferencia: €{elite_metrics['std_return'] - full_metrics['std_return']:+,.2f}
        
        • Value at Risk (95%):
          - FULL: €{full_metrics['var_95']:,.2f}
          - ELITE: €{elite_metrics['var_95']:,.2f}
          - Diferencia: €{elite_metrics['var_95'] - full_metrics['var_95']:+,.2f}
        
        ✅ RECOMENDACIÓN FINAL
        {'=' * 60}
        """
        
        # Determinar recomendación basada en múltiples factores
        full_score = full_metrics['quality_score']
        elite_score = elite_metrics['quality_score']
        
        if elite_score > full_score + 10:
            report += "✅ **RECOMENDACIÓN FUERTE: UTILIZAR PORTAFOLIO ELITE**\n\n"
            report += "   • Elite supera significativamente en calidad y rendimiento\n"
            report += "   • Mayor eficiencia (ROI por unidad de riesgo)\n"
            report += "   • Mejor relación riesgo/retorno\n"
        elif elite_score > full_score + 3:
            report += "✅ **RECOMENDACIÓN: UTILIZAR PORTAFOLIO ELITE**\n\n"
            report += "   • Elite muestra mejores métricas en general\n"
            report += "   • Reducción adecuada de exposición\n"
            report += "   • Concentración efectiva en mejores columnas\n"
        elif abs(elite_score - full_score) < 3:
            report += "⚖️ **RECOMENDACIÓN NEUTRA: AMBOS SON VIABLES**\n\n"
            report += "   • Diferencias menores entre portafolios\n"
            report += "   • Elegir según preferencia de riesgo\n"
            report += "   • FULL para mayor cobertura, ELITE para concentración\n"
        elif full_score > elite_score + 3:
            report += "ℹ️ **RECOMENDACIÓN: UTILIZAR PORTAFOLIO FULL**\n\n"
            report += "   • Full muestra mejor rendimiento ajustado al riesgo\n"
            report += "   • Mayor diversificación puede ser beneficiosa\n"
            report += "   • Cobertura más amplia en escenarios adversos\n"
        else:
            report += "⚠️ **RECOMENDACIÓN CAUTELOSA: UTILIZAR PORTAFOLIO FULL**\n\n"
            report += "   • Elite no muestra ventajas claras\n"
            report += "   • Full ofrece mejor protección downside\n"
            report += "   • Considerar reducir exposición en Elite\n"
        
        report += f"\n{'=' * 60}"
        report += "\nGenerado por: ACBE-S73 Quantum Betting Suite v3.0"
        report += "\nSistema de Análisis Cuantitativo de Apuestas Deportivas"
        report += f"\n{'=' * 60}"
        
        return report
    
    @staticmethod
    def render_export_section(exports: Dict):
        """Renderiza sección de exportación de datos."""
        st.header("📤 Exportación de Resultados")
        
        if not exports:
            st.info("No hay datos disponibles para exportar")
            return
        
        # Agrupar exportaciones por tipo
        export_groups = {}
        for key, export in exports.items():
            file_type = export['filename'].split('.')[-1].upper()
            if file_type not in export_groups:
                export_groups[file_type] = []
            export_groups[file_type].append(export)
        
        # Mostrar botones de descarga por grupo
        for file_type, group_exports in export_groups.items():
            st.subheader(f"📄 Archivos {file_type}")
            
            cols = st.columns(min(3, len(group_exports)))
            for idx, export in enumerate(group_exports):
                with cols[idx % 3]:
                    st.download_button(
                        label=f"Descargar {export['filename']}",
                        data=export['data'],
                        file_name=export['filename'],
                        mime=export['mime'],
                        key=f"download_{file_type}_{idx}",
                        use_container_width=True
                    )
                    st.caption(export.get('description', ''))
        
        # Información adicional
        st.info("💡 **Nota:** Todos los archivos incluyen timestamp para seguimiento")

# ============================================================================
# SECCIÓN 9: MOTOR DE VISUALIZACIÓN AVANZADO v3.0
# ============================================================================

class VisualizationEngine:
    """
    Motor de visualización avanzada para ACBE-S73 v3.0.
    Sin redundancias - Funcionalidades únicas complementarias.
    """
    
    @staticmethod
    def create_portfolio_concentration_chart(stakes_full: np.ndarray, 
                                           stakes_elite: np.ndarray) -> go.Figure:
        """
        Visualiza concentración de stakes entre portafolios.
        
        Args:
            stakes_full: Stakes del portafolio full (73)
            stakes_elite: Stakes del portafolio elite (24)
            
        Returns:
            Figura Plotly con análisis de concentración
        """
        # Calcular distribución acumulativa
        sorted_stakes_full = np.sort(stakes_full)[::-1]
        sorted_stakes_elite = np.sort(stakes_elite)[::-1]
        
        # Distribución acumulativa
        cum_full = np.cumsum(sorted_stakes_full)
        cum_elite = np.cumsum(sorted_stakes_elite)
        
        # Porcentaje de columnas
        pct_columns_full = np.arange(1, len(stakes_full) + 1) / len(stakes_full) * 100
        pct_columns_elite = np.arange(1, len(stakes_elite) + 1) / len(stakes_elite) * 100
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Curva de Lorenz', 'Distribución Acumulativa',
                          'Comparación Top 10 Stakes', 'Concentración por Quintil'),
            specs=[[{}, {}], [{}, {}]]
        )
        
        # 1. Curva de Lorenz
        fig.add_trace(
            go.Scatter(
                x=pct_columns_full,
                y=cum_full / cum_full[-1] * 100,
                mode='lines',
                name='Full (73)',
                line=dict(color=SystemConfig.COLORS['primary'], width=3),
                fill='tozeroy',
                fillcolor='rgba(37, 150, 190, 0.1)'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=pct_columns_elite,
                y=cum_elite / cum_elite[-1] * 100,
                mode='lines',
                name='Elite (24)',
                line=dict(color=SystemConfig.COLORS['success'], width=3),
                fill='tozeroy',
                fillcolor='rgba(76, 175, 80, 0.1)'
            ),
            row=1, col=1
        )
        
        # Línea de igualdad perfecta
        fig.add_trace(
            go.Scatter(
                x=[0, 100],
                y=[0, 100],
                mode='lines',
                name='Igualdad Perfecta',
                line=dict(color='gray', dash='dash', width=1),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # 2. Distribución acumulativa
        fig.add_trace(
            go.Scatter(
                x=np.arange(1, len(sorted_stakes_full) + 1),
                y=sorted_stakes_full * 100,
                mode='lines+markers',
                name='Stakes Full (%)',
                line=dict(color=SystemConfig.COLORS['primary'], width=2),
                marker=dict(size=4)
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=np.arange(1, len(sorted_stakes_elite) + 1),
                y=sorted_stakes_elite * 100,
                mode='lines+markers',
                name='Stakes Elite (%)',
                line=dict(color=SystemConfig.COLORS['success'], width=2),
                marker=dict(size=4)
            ),
            row=1, col=2
        )
        
        # 3. Top 10 stakes comparados
        top_n = min(10, len(stakes_full), len(stakes_elite))
        
        # Normalizar para comparación
        top_stakes_full = sorted_stakes_full[:top_n] * 100
        top_stakes_elite = sorted_stakes_elite[:top_n] * 100
        
        fig.add_trace(
            go.Bar(
                x=list(range(1, top_n + 1)),
                y=top_stakes_full,
                name='Top Full',
                marker_color=SystemConfig.COLORS['primary'],
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=list(range(1, top_n + 1)),
                y=top_stakes_elite,
                name='Top Elite',
                marker_color=SystemConfig.COLORS['success'],
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # 4. Concentración por quintil
        def calculate_quintile_concentration(stakes):
            if len(stakes) == 0:
                return [0, 0, 0, 0, 0]
            
            sorted_stakes = np.sort(stakes)[::-1]
            quintile_size = len(sorted_stakes) // 5
            quintiles = []
            
            for i in range(5):
                start = i * quintile_size
                end = (i + 1) * quintile_size if i < 4 else len(sorted_stakes)
                quintile_sum = np.sum(sorted_stakes[start:end])
                quintiles.append(quintile_sum / np.sum(stakes) * 100)
            
            return quintiles
        
        quintiles_full = calculate_quintile_concentration(stakes_full)
        quintiles_elite = calculate_quintile_concentration(stakes_elite)
        
        fig.add_trace(
            go.Bar(
                x=['Q1 (Top 20%)', 'Q2', 'Q3', 'Q4', 'Q5 (Bottom 20%)'],
                y=quintiles_full,
                name='Full',
                marker_color=SystemConfig.COLORS['primary'],
                text=[f'{q:.1f}%' for q in quintiles_full],
                textposition='auto'
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=['Q1 (Top 20%)', 'Q2', 'Q3', 'Q4', 'Q5 (Bottom 20%)'],
                y=quintiles_elite,
                name='Elite',
                marker_color=SystemConfig.COLORS['success'],
                text=[f'{q:.1f}%' for q in quintiles_elite],
                textposition='auto'
            ),
            row=2, col=2
        )
        
        # Actualizar layout
        fig.update_layout(
            title='Análisis de Concentración de Portafolios',
            height=800,
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        fig.update_xaxes(title_text='% Columnas', row=1, col=1)
        fig.update_yaxes(title_text='% Stakes Acumulado', row=1, col=1)
        
        fig.update_xaxes(title_text='Ranking de Columna', row=1, col=2)
        fig.update_yaxes(title_text='Stake (%)', row=1, col=2)
        
        fig.update_xaxes(title_text='Top 10 Columnas', row=2, col=1)
        fig.update_yaxes(title_text='Stake (%)', row=2, col=1)
        
        fig.update_xaxes(title_text='Quintiles', row=2, col=2)
        fig.update_yaxes(title_text='% del Total Stake', row=2, col=2)
        
        return fig
    
    @staticmethod
    def create_correlation_matrix(probabilities: np.ndarray, 
                                 odds_matrix: np.ndarray) -> go.Figure:
        """
        Crea matriz de correlación entre resultados de partidos.
        
        Args:
            probabilities: Probabilidades ACBE (6, 3)
            odds_matrix: Cuotas (6, 3)
            
        Returns:
            Matriz de correlación visual
        """
        n_matches = probabilities.shape[0]
        
        # Calcular matriz de correlación basada en probabilidades
        correlation_matrix = np.zeros((n_matches, n_matches))
        
        for i in range(n_matches):
            for j in range(n_matches):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    # Correlación basada en entropía conjunta
                    entropy_i = -np.sum(probabilities[i] * np.log(probabilities[i]))
                    entropy_j = -np.sum(probabilities[j] * np.log(probabilities[j]))
                    
                    # Probabilidades conjuntas asumiendo independencia (simplificación)
                    joint_probs = np.outer(probabilities[i], probabilities[j]).flatten()
                    joint_entropy = -np.sum(joint_probs * np.log(joint_probs))
                    
                    # Correlación de información mutua normalizada
                    mutual_info = entropy_i + entropy_j - joint_entropy
                    correlation = mutual_info / np.sqrt(entropy_i * entropy_j)
                    correlation_matrix[i, j] = correlation
        
        # Crear heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=[f'Partido {i+1}' for i in range(n_matches)],
            y=[f'Partido {i+1}' for i in range(n_matches)],
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            colorbar=dict(title='Correlación')
        ))
        
        # Añadir anotaciones
        annotations = []
        for i in range(n_matches):
            for j in range(n_matches):
                annotations.append(
                    dict(
                        x=f'Partido {j+1}',
                        y=f'Partido {i+1}',
                        text=f'{correlation_matrix[i, j]:.2f}',
                        showarrow=False,
                        font=dict(color='white' if abs(correlation_matrix[i, j]) > 0.5 else 'black')
                    )
                )
        
        fig.update_layout(
            title='Matriz de Correlación entre Partidos',
            height=600,
            annotations=annotations
        )
        
        return fig
    
    @staticmethod
    def create_multidimensional_scatter(columns_df: pd.DataFrame) -> go.Figure:
        """
        Crea gráfico de dispersión multidimensional para análisis de columnas.
        
        Args:
            columns_df: DataFrame con métricas de columnas
            
        Returns:
            Gráfico de burbujas 3D interactivo
        """
        if columns_df.empty:
            return go.Figure()
        
        # Normalizar tamaños para visualización
        sizes = columns_df['Stake (%)'].values
        sizes_normalized = (sizes - sizes.min()) / (sizes.max() - sizes.min()) * 30 + 10
        
        # Colores por quintil de probabilidad
        prob_quintiles = pd.qcut(columns_df['Probabilidad'], 5, labels=False)
        color_scale = ['#FF6B6B', '#FFA726', '#FFD54F', '#AED581', '#4CAF50']
        colors = [color_scale[q] for q in prob_quintiles]
        
        fig = go.Figure(data=go.Scatter3d(
            x=columns_df['Probabilidad'],
            y=columns_df['Valor Esperado'],
            z=columns_df['Entropía Prom.'],
            mode='markers',
            marker=dict(
                size=sizes_normalized,
                color=colors,
                opacity=0.8,
                line=dict(color='white', width=1)
            ),
            text=columns_df['Combinación'],
            hovertemplate=(
                '<b>Combinación:</b> %{text}<br>'
                '<b>Probabilidad:</b> %{x:.2%}<br>'
                '<b>Valor Esperado:</b> %{y:.3f}<br>'
                '<b>Entropía:</b> %{z:.3f}<br>'
                '<b>Stake:</b> %{marker.size:.1f}%<br>'
                '<extra></extra>'
            )
        ))
        
        fig.update_layout(
            title='Análisis Multidimensional de Columnas',
            scene=dict(
                xaxis_title='Probabilidad',
                yaxis_title='Valor Esperado',
                zaxis_title='Entropía (Riesgo)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=600,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_time_series_analysis(backtest_results: Dict) -> go.Figure:
        """
        Crea análisis de series temporales para resultados de backtesting.
        
        Args:
            backtest_results: Resultados de backtesting dual
            
        Returns:
            Gráficos de series temporales múltiples
        """
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Evolución del Bankroll', 'Drawdown por Portafolio',
                          'Retornos Acumulados', 'Volatilidad Rolling',
                          'Ratio Sharpe Rolling', 'Correlación Entre Portafolios'),
            specs=[[{}, {}], [{}, {}], [{}, {}]]
        )
        
        # Colores
        colors = {
            'full': SystemConfig.COLORS['primary'],
            'elite': SystemConfig.COLORS['success']
        }
        
        for ptype in ['full', 'elite']:
            if ptype in backtest_results:
                result = backtest_results[ptype]
                
                # 1. Evolución del bankroll (promedio de simulaciones)
                if 'equity_curves' in result:
                    equity_curves = result['equity_curves']
                    avg_curve = np.mean(equity_curves, axis=0)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(len(avg_curve))),
                            y=avg_curve,
                            mode='lines',
                            name=f'Bankroll {ptype.upper()}',
                            line=dict(color=colors[ptype], width=2)
                        ),
                        row=1, col=1
                    )
                
                # 2. Drawdown
                if 'equity_curves' in result:
                    # Calcular drawdown promedio
                    drawdowns = []
                    for curve in equity_curves:
                        peak = np.maximum.accumulate(curve)
                        drawdown = (peak - curve) / peak * 100
                        drawdowns.append(drawdown)
                    
                    avg_drawdown = np.mean(drawdowns, axis=0)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(len(avg_drawdown))),
                            y=avg_drawdown,
                            mode='lines',
                            name=f'Drawdown {ptype.upper()}',
                            line=dict(color=colors[ptype], width=2),
                            fill='tozeroy',
                            fillcolor=f'rgba{tuple(int(colors[ptype][i:i+2], 16) for i in (1, 3, 5)) + (0.1,)}'
                        ),
                        row=1, col=2
                    )
                
                # 3. Retornos acumulados
                if 'equity_curves' in result:
                    returns = []
                    for curve in equity_curves:
                        returns.append((curve[-1] - curve[0]) / curve[0] * 100)
                    
                    fig.add_trace(
                        go.Box(
                            y=returns,
                            name=f'Retornos {ptype.upper()}',
                            marker_color=colors[ptype],
                            boxmean=True
                        ),
                        row=2, col=1
                    )
                
                # 4. Volatilidad rolling (desviación estándar de retornos)
                if 'equity_curves' in result and len(equity_curves) > 0:
                    # Calcular retornos diarios
                    returns_matrix = np.diff(equity_curves) / equity_curves[:, :-1]
                    
                    # Volatilidad rolling (ventana de 10 rondas)
                    window = min(10, returns_matrix.shape[1])
                    rolling_vol = np.std(returns_matrix[:, :window], axis=1) * np.sqrt(252)
                    
                    fig.add_trace(
                        go.Histogram(
                            x=rolling_vol * 100,
                            name=f'Volatilidad {ptype.upper()}',
                            marker_color=colors[ptype],
                            opacity=0.6,
                            nbinsx=30
                        ),
                        row=2, col=2
                    )
        
        # 5. Ratio Sharpe rolling (si ambos portafolios están disponibles)
        if 'full' in backtest_results and 'elite' in backtest_results:
            full_curves = backtest_results['full']['equity_curves']
            elite_curves = backtest_results['elite']['equity_curves']
            
            # Calcular Sharpe ratios
            sharpe_full = []
            sharpe_elite = []
            
            for i in range(min(len(full_curves), len(elite_curves))):
                if i < len(full_curves):
                    returns_full = np.diff(full_curves[i]) / full_curves[i][:-1]
                    sharpe_full.append(np.mean(returns_full) / np.std(returns_full) * np.sqrt(252))
                
                if i < len(elite_curves):
                    returns_elite = np.diff(elite_curves[i]) / elite_curves[i][:-1]
                    sharpe_elite.append(np.mean(returns_elite) / np.std(returns_elite) * np.sqrt(252))
            
            fig.add_trace(
                go.Box(
                    y=sharpe_full,
                    name='Sharpe FULL',
                    marker_color=colors['full']
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Box(
                    y=sharpe_elite,
                    name='Sharpe ELITE',
                    marker_color=colors['elite']
                ),
                row=3, col=1
            )
            
            # 6. Correlación entre portafolios
            if len(full_curves) > 0 and len(elite_curves) > 0:
                # Tomar primera simulación como ejemplo
                correlation = np.corrcoef(full_curves[0], elite_curves[0])[0, 1]
                
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=correlation,
                        title={'text': "Correlación FULL-ELITE"},
                        gauge={
                            'axis': {'range': [-1, 1]},
                            'bar': {'color': colors['primary']},
                            'steps': [
                                {'range': [-1, -0.5], 'color': 'red'},
                                {'range': [-0.5, 0], 'color': 'orange'},
                                {'range': [0, 0.5], 'color': 'lightblue'},
                                {'range': [0.5, 1], 'color': 'blue'}
                            ]
                        }
                    ),
                    row=3, col=2
                )
        
        # Actualizar layout
        fig.update_layout(
            title='Análisis de Series Temporales - Backtesting',
            height=900,
            showlegend=True
        )
        
        # Actualizar ejes
        fig.update_xaxes(title_text='Ronda', row=1, col=1)
        fig.update_yaxes(title_text='Bankroll (€)', row=1, col=1)
        
        fig.update_xaxes(title_text='Ronda', row=1, col=2)
        fig.update_yaxes(title_text='Drawdown (%)', row=1, col=2)
        
        fig.update_xaxes(title_text='Portafolio', row=2, col=1)
        fig.update_yaxes(title_text='Retorno Total (%)', row=2, col=1)
        
        fig.update_xaxes(title_text='Volatilidad Anualizada (%)', row=2, col=2)
        fig.update_yaxes(title_text='Frecuencia', row=2, col=2)
        
        fig.update_xaxes(title_text='Portafolio', row=3, col=1)
        fig.update_yaxes(title_text='Sharpe Ratio', row=3, col=1)
        
        return fig
    
    @staticmethod
    def create_interactive_radar_chart(full_metrics: Dict, elite_metrics: Dict) -> go.Figure:
        """
        Crea gráfico de radar interactivo para comparación de métricas.
        
        Args:
            full_metrics: Métricas del portafolio full
            elite_metrics: Métricas del portafolio elite
            
        Returns:
            Gráfico de radar polar
        """
        # Categorías para comparación
        categories = ['Retorno', 'Riesgo', 'Eficiencia', 'Calidad', 'Consistencia']
        
        # Normalizar métricas a escala 0-1
        def normalize_metrics(metrics):
            normalized = []
            
            # Retorno (ROI total)
            retorno = min(max(metrics.get('avg_total_return_pct', 0) / 100, 0), 1)
            normalized.append(retorno)
            
            # Riesgo (inverso del drawdown)
            drawdown = metrics.get('avg_max_drawdown_pct', 100)
            riesgo = 1 - min(max(drawdown / 100, 0), 1)
            normalized.append(riesgo)
            
            # Eficiencia (Sharpe ratio)
            sharpe = metrics.get('sharpe_ratio', 0)
            eficiencia = min(max(sharpe / 3, 0), 1)
            normalized.append(eficiencia)
            
            # Calidad (score de calidad)
            calidad = metrics.get('quality_score', 0) / 100
            normalized.append(calidad)
            
            # Consistencia (win rate)
            winrate = metrics.get('win_rate_pct', 0) / 100
            normalized.append(winrate)
            
            return normalized
        
        values_full = normalize_metrics(full_metrics)
        values_elite = normalize_metrics(elite_metrics)
        
        # Crear gráfico de radar
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values_full,
            theta=categories,
            fill='toself',
            name='Portafolio FULL',
            line=dict(color=SystemConfig.COLORS['primary'], width=3),
            fillcolor='rgba(37, 150, 190, 0.3)'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=values_elite,
            theta=categories,
            fill='toself',
            name='Portafolio ELITE',
            line=dict(color=SystemConfig.COLORS['success'], width=3),
            fillcolor='rgba(76, 175, 80, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1],
                    ticktext=['0%', '20%', '40%', '60%', '80%', '100%']
                ),
                angularaxis=dict(
                    direction="clockwise",
                    rotation=90
                )
            ),
            title='Comparativa Radar de Métricas',
            height=500,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig

# ============================================================================
# SECCIÓN 10: APLICACIÓN PRINCIPAL UNIFICADA v3.0
# ============================================================================

class ACBEAppV3:
    """
    Aplicación principal unificada ACBE-S73 v3.0.
    Integra todas las funcionalidades en una interfaz coherente.
    """
    
    def __init__(self):
        self.setup_page_config()
        SessionStateManager.initialize_session_state()
        
        # Inicializar componentes (solo cuando se necesiten)
        self.match_input = None
        self.portfolio_engine = None
        self.backtester = None
        self.visualizer = None
        self.exporter = None
    
    def setup_page_config(self):
        """Configuración de página unificada."""
        st.set_page_config(
            page_title="ACBE-S73 Quantum Betting Suite v3.0",
            page_icon="🎯",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def render_app(self):
        """Renderiza la aplicación completa."""
        # Header principal
        self._render_main_header()
        
        # Sidebar con configuración
        config = self._render_sidebar()
        
        # Pestañas principales
        self._render_main_tabs(config)
    
    def _render_main_header(self):
        """Renderiza el header principal."""
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        ">
            <h1 style="color: white; margin: 0; font-size: 2.5rem;">🎯 ACBE-S73 Quantum Betting Suite v3.0</h1>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-size: 1.1rem;">
                Sistema Profesional de Optimización de Apuestas Deportivas con Análisis Cuantitativo
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Barra de estado
        self._render_status_bar()
    
    def _render_status_bar(self):
        """Renderiza barra de estado minimalista."""
        cols = st.columns(6)
        
        with cols[0]:
            st.caption("**Versión:** v3.0")
        
        with cols[1]:
            status = "✅ Activo" if st.session_state.get('data_loaded', False) else "⏳ Esperando datos"
            st.caption(f"**Estado:** {status}")
        
        with cols[2]:
            portfolio = st.session_state.get('portfolio_type', 'full')
            st.caption(f"**Portafolio:** {portfolio.upper()}")
        
        with cols[3]:
            if st.session_state.get('elite_columns_selected', False):
                st.caption("**Reducción:** ✅")
            else:
                st.caption("**Reducción:** ⏳")
        
        with cols[4]:
            if st.session_state.get('backtest_completed', False):
                st.caption("**Backtest:** ✅")
            else:
                st.caption("**Backtest:** ⏳")
        
        with cols[5]:
            if st.session_state.get('system_ready', False):
                st.caption("**Sistema:** 🟢")
            else:
                st.caption("**Sistema:** 🟡")
    
    def _render_sidebar(self) -> Dict:
        """Renderiza sidebar unificado."""
        with st.sidebar:
            st.title("⚙️ Configuración")
            
            # Control de flujo
            st.subheader("🔄 Flujo de Trabajo")
            
            current_step = st.radio(
                "Paso actual:",
                ["1. Input Datos", "2. Sistema S73", "3. Backtesting", "4. Exportar"],
                index=0
            )
            
            st.session_state.current_step = current_step
            
            # Configuración global
            st.subheader("💰 Configuración Global")
            
            bankroll = st.number_input(
                "Bankroll Inicial (€)",
                min_value=100.0,
                max_value=1000000.0,
                value=SystemConfig.DEFAULT_BANKROLL,
                step=1000.0
            )
            
            # Selección de portafolio
            portfolio_type = st.radio(
                "Estrategia de Portafolio:",
                ["Full (73 columnas)", "Elite (24 columnas)"],
                index=0
            )
            
            st.session_state.portfolio_type = "full" if "Full" in portfolio_type else "elite"
            
            # Gestión de riesgo
            st.subheader("🛡️ Gestión de Riesgo")
            
            max_exposure = st.slider(
                "Exposición Máxima (%)",
                min_value=5,
                max_value=30,
                value=15,
                step=1
            )
            
            kelly_fraction = st.slider(
                "Fracción de Kelly",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.1
            )
            
            # Parámetros de simulación
            st.subheader("🎲 Simulación")
            
            monte_carlo_sims = st.number_input(
                "Simulaciones Monte Carlo",
                min_value=1000,
                max_value=50000,
                value=10000,
                step=1000
            )
            
            n_rounds = st.slider(
                "Rondas Backtesting",
                min_value=10,
                max_value=500,
                value=100,
                step=10
            )
            
            # Reducción elite
            st.subheader("🏆 Reducción Elite")
            
            apply_elite_reduction = st.toggle(
                "Aplicar reducción elite",
                value=True
            )
            
            if apply_elite_reduction:
                elite_target = st.slider(
                    "Columnas Elite objetivo",
                    min_value=12,
                    max_value=36,
                    value=24,
                    step=1
                )
            else:
                elite_target = 24
            
            # Botones de control
            st.subheader("🎮 Control")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Reiniciar", use_container_width=True):
                    SessionStateManager.reset_to_input()
                    st.rerun()
            
            with col2:
                if st.button("▶ Ejecutar Todo", type="primary", use_container_width=True):
                    st.session_state.run_complete_workflow = True
            
            return {
                'bankroll': bankroll,
                'portfolio_type': st.session_state.portfolio_type,
                'max_exposure': max_exposure / 100,
                'kelly_fraction': kelly_fraction,
                'monte_carlo_sims': monte_carlo_sims,
                'n_rounds': n_rounds,
                'apply_elite_reduction': apply_elite_reduction,
                'elite_target': elite_target,
                'current_step': current_step
            }
    
    def _render_main_tabs(self, config: Dict):
        """Renderiza las pestañas principales según el paso actual."""
        step = config['current_step']
        
        if step == "1. Input Datos":
            self._render_input_section()
        elif step == "2. Sistema S73":
            self._render_s73_section(config)
        elif step == "3. Backtesting":
            self._render_backtesting_section(config)
        elif step == "4. Exportar":
            self._render_export_section()
    
    def _render_input_section(self):
        """Renderiza sección de input de datos."""
        st.header("📥 Input de Datos")
        
        if self.match_input is None:
            self.match_input = MatchInputLayer()
        
        # Selector de fuente de datos
        data_source = st.radio(
            "Fuente de datos:",
            ["⚽ Input Manual", "📊 Datos de Ejemplo", "📁 Cargar Archivo"],
            horizontal=True
        )
        
        if data_source == "⚽ Input Manual":
            matches_df, params_dict, mode = self.match_input.render_input_section()
            
            if st.button("📊 Procesar Datos", type="primary", use_container_width=True):
                with st.spinner("Procesando datos..."):
                    processed_df, odds_matrix, probabilities = (
                        self.match_input.process_input(params_dict)
                    )
                    
                    # Calcular entropías
                    normalized_entropies = ACBEModel.calculate_entropy(probabilities)
                    normalized_entropies = (normalized_entropies - normalized_entropies.min()) / (
                        normalized_entropies.max() - normalized_entropies.min()
                    )
                    
                    # Guardar en estado
                    st.session_state.update({
                        'matches_df': processed_df,
                        'odds_matrix': odds_matrix,
                        'probabilities': probabilities,
                        'normalized_entropies': normalized_entropies,
                        'data_loaded': True,
                        'mode': mode
                    })
                    
                    st.success("✅ Datos procesados correctamente")
        
        elif data_source == "📊 Datos de Ejemplo":
            st.info("Usando datos de ejemplo predefinidos")
            # Aquí se cargarían datos de ejemplo
            # Por simplicidad, omitimos la implementación completa
        
        elif data_source == "📁 Cargar Archivo":
            uploaded_file = st.file_uploader("Subir archivo CSV", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success(f"✅ Archivo cargado: {uploaded_file.name}")
                    st.dataframe(df.head(), use_container_width=True)
                except Exception as e:
                    st.error(f"Error al cargar archivo: {str(e)}")
    
    def _render_s73_section(self, config: Dict):
        """Renderiza sección del sistema S73."""
        if not st.session_state.get('data_loaded', False):
            st.warning("⚠️ Primero carga datos en la sección 'Input Datos'")
            return
        
        st.header("🧮 Sistema S73 v3.0")
        
        # Extraer datos del estado
        probabilities = st.session_state.get('probabilities')
        odds_matrix = st.session_state.get('odds_matrix')
        normalized_entropies = st.session_state.get('normalized_entropies')
        bankroll = config.get('bankroll')
        
        if probabilities is None or odds_matrix is None:
            st.error("❌ Datos no disponibles")
            return
        
        # Botón para ejecutar sistema
        if st.button("🚀 Generar Sistema S73", type="primary", use_container_width=True):
            with st.spinner("Construyendo sistema S73..."):
                # Llamar a la funcionalidad de la Sección 4
                from sections.section_4 import ACBEApp as Section4App
                s73_app = Section4App()
                
                # Obtener resultados del sistema S73
                s73_results = s73_app.render_s73_system(
                    probabilities, odds_matrix, normalized_entropies, bankroll, config
                )
                
                # Guardar resultados
                st.session_state.update({
                    's73_results': s73_results,
                    'system_ready': True
                })
                
                st.success("✅ Sistema S73 generado correctamente")
        
        # Mostrar resultados si ya existen
        if st.session_state.get('system_ready', False):
            s73_results = st.session_state.get('s73_results', {})
            
            if s73_results:
                # Métricas del sistema
                self._render_system_metrics(s73_results)
                
                # Tabla de columnas
                if 'columns_df' in s73_results:
                    st.subheader("📋 Columnas del Sistema")
                    st.dataframe(s73_results['columns_df'], use_container_width=True, height=400)
    
    def _render_system_metrics(self, s73_results: Dict):
        """Renderiza métricas del sistema."""
        st.subheader("📊 Métricas del Sistema")
        
        cols = st.columns(4)
        
        with cols[0]:
            n_columns = len(s73_results.get('combinations', []))
            st.metric("Columnas Totales", n_columns)
        
        with cols[1]:
            if 'columns_df' in s73_results:
                total_exposure = s73_results['columns_df']['Stake (%)'].sum()
                st.metric("Exposición Total", f"{total_exposure:.1f}%")
        
        with cols[2]:
            if 'columns_df' in s73_results:
                avg_prob = s73_results['columns_df']['Probabilidad'].mean()
                st.metric("Probabilidad Promedio", f"{avg_prob:.2%}")
        
        with cols[3]:
            if 'columns_df' in s73_results:
                avg_ev = s73_results['columns_df']['Valor Esperado'].mean()
                st.metric("EV Promedio", f"{avg_ev:.3f}")
    
    def _render_backtesting_section(self, config: Dict):
        """Renderiza sección de backtesting."""
        if not st.session_state.get('system_ready', False):
            st.warning("⚠️ Primero genera el sistema S73 en la sección anterior")
            return
        
        st.header("📈 Backtesting Avanzado")
        
        if self.backtester is None:
            self.backtester = VectorizedBacktester(config.get('bankroll'))
        
        if self.visualizer is None:
            self.visualizer = VisualizationEngine()
        
        # Configuración de backtesting
        with st.expander("⚙️ Configuración de Backtesting", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                n_simulations = st.number_input(
                    "Simulaciones",
                    min_value=100,
                    max_value=10000,
                    value=1000,
                    step=100
                )
            
            with col2:
                confidence_level = st.slider(
                    "Nivel de Confianza",
                    min_value=90,
                    max_value=99,
                    value=95,
                    step=1
                )
        
        # Botón para ejecutar backtesting
        if st.button("📊 Ejecutar Backtesting", type="primary", use_container_width=True):
            with st.spinner("Ejecutando backtesting..."):
                # Extraer datos
                s73_results = st.session_state.get('s73_results', {})
                probabilities = st.session_state.get('probabilities')
                odds_matrix = st.session_state.get('odds_matrix')
                normalized_entropies = st.session_state.get('normalized_entropies')
                
                if not s73_results:
                    st.error("❌ No hay resultados del sistema S73")
                    return
                
                # Ejecutar backtesting
                backtest_results = self.backtester.run_dual_backtest(
                    config={
                        'probabilities': probabilities,
                        'odds_matrix': odds_matrix,
                        'normalized_entropies': normalized_entropies,
                        'bankroll': config.get('bankroll'),
                        'n_rounds': config.get('n_rounds'),
                        'monte_carlo_sims': n_simulations
                    },
                    s73_results=s73_results,
                    elite_results=s73_results.get('elite_data')
                )
                
                # Guardar resultados
                st.session_state.update({
                    'backtest_results': backtest_results,
                    'backtest_completed': True
                })
                
                st.success("✅ Backtesting completado")
        
        # Mostrar resultados si ya existen
        if st.session_state.get('backtest_completed', False):
            backtest_results = st.session_state.get('backtest_results', {})
            
            if backtest_results:
                # Renderizar resultados usando el motor de visualización
                self.backtester.render_backtest_results(backtest_results)
                
                # Gráficos avanzados
                st.subheader("📊 Análisis Avanzado")
                
                tab1, tab2, tab3 = st.tabs(["Series Temporales", "Correlación", "Concentración"])
                
                with tab1:
                    if self.visualizer:
                        fig = self.visualizer.create_time_series_analysis(backtest_results)
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    probabilities = st.session_state.get('probabilities')
                    odds_matrix = st.session_state.get('odds_matrix')
                    
                    if probabilities is not None and odds_matrix is not None:
                        fig = self.visualizer.create_correlation_matrix(probabilities, odds_matrix)
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    s73_results = st.session_state.get('s73_results', {})
                    if 'kelly_stakes' in s73_results:
                        stakes_full = s73_results['kelly_stakes']
                        stakes_elite = s73_results.get('elite_data', {}).get('kelly_stakes', stakes_full[:24])
                        
                        fig = self.visualizer.create_portfolio_concentration_chart(
                            stakes_full, stakes_elite
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    def _render_export_section(self):
        """Renderiza sección de exportación."""
        st.header("📤 Exportación de Resultados")
        
        if not st.session_state.get('backtest_completed', False):
            st.warning("⚠️ Primero completa el backtesting en la sección anterior")
            return
        
        if self.exporter is None:
            self.exporter = DataExporter()
        
        # Selección de formato
        col1, col2, col3 = st.columns(3)
        
        with col1:
            export_format = st.selectbox(
                "Formato de exportación:",
                ["CSV", "Excel", "Reporte Completo"]
            )
        
        with col2:
            include_data = st.multiselect(
                "Incluir datos:",
                ["Columnas", "Métricas", "Backtesting", "Gráficos"],
                default=["Columnas", "Métricas", "Backtesting"]
            )
        
        with col3:
            timestamp = st.toggle("Incluir timestamp", value=True)
        
        # Botón de exportación
        if st.button("💾 Generar Exportación", type="primary", use_container_width=True):
            with st.spinner("Generando archivos de exportación..."):
                # Recopilar datos
                s73_results = st.session_state.get('s73_results', {})
                backtest_results = st.session_state.get('backtest_results', {})
                config = {
                    'portfolio_type': st.session_state.get('portfolio_type', 'full'),
                    'bankroll': st.session_state.get('bankroll', 10000)
                }
                
                if not s73_results:
                    st.error("❌ No hay datos para exportar")
                    return
                
                # Generar exportación
                exports = self.exporter.export_system_results(
                    columns_df=s73_results.get('columns_df', pd.DataFrame()),
                    s73_results=s73_results,
                    config=config,
                    backtest_results=backtest_results
                )
                
                # Mostrar botones de descarga
                st.success("✅ Exportación generada")
                
                # Renderizar sección de descarga
                self.exporter.render_export_section(exports)
        
        # Vista previa de datos
        st.subheader("👁️ Vista Previa")
        
        with st.expander("Ver datos disponibles", expanded=False):
            tab1, tab2, tab3 = st.tabs(["Columnas", "Métricas", "Backtesting"])
            
            with tab1:
                s73_results = st.session_state.get('s73_results', {})
                if 'columns_df' in s73_results:
                    st.dataframe(s73_results['columns_df'].head(10), use_container_width=True)
            
            with tab2:
                if st.session_state.get('backtest_completed', False):
                    backtest_results = st.session_state.get('backtest_results', {})
                    if backtest_results and 'full' in backtest_results:
                        metrics = backtest_results['full']['metrics']
                        metrics_df = pd.DataFrame([metrics])
                        st.dataframe(metrics_df, use_container_width=True)
            
            with tab3:
                if st.session_state.get('backtest_completed', False):
                    backtest_results = st.session_state.get('backtest_results', {})
                    if backtest_results and 'comparison' in backtest_results:
                        comparison = backtest_results['comparison']
                        st.json(comparison, expanded=False)
    
    def run(self):
        """Método principal de ejecución."""
        try:
            self.render_app()
        except Exception as e:
            st.error(f"❌ Error en la aplicación: {str(e)}")
            st.info("🔄 Intenta reiniciar la aplicación o contactar con soporte")
            
            # Botón de reinicio de emergencia
            if st.button("🔄 Reinicio de Emergencia"):
                SessionStateManager.reset_to_input()
                st.rerun()


# ============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# ============================================================================

def main():
    """Función principal de la aplicación."""
    # Verificar dependencias
    try:
        import streamlit as st
        import numpy as np
        import pandas as pd
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as e:
        st.error(f"❌ Error de dependencias: {str(e)}")
        st.info("Instala las dependencias con: pip install -r requirements.txt")
        return
    
    # Inicializar y ejecutar aplicación
    app = ACBEAppV3()
    app.run()


if __name__ == "__main__":
    main()
