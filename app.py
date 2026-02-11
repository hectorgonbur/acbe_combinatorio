def render_analysis_phase(self, config: Dict):
        """Fase 2: Análisis completo con pestañas - VERIFICADO."""
        
        # 1. Validaciones de carga
        if not st.session_state.get('data_loaded', False):
            st.error("❌ No hay datos cargados. Vuelve a la fase de input.")
            return

        matches_data = st.session_state.get('matches_data', {})
        if not matches_data:
            st.error("❌ Los datos no se cargaron correctamente")
            return
            
        probabilities = matches_data.get('probabilities')
        odds_matrix = matches_data.get('odds_matrix')
        normalized_entropies = matches_data.get('normalized_entropies')

        if probabilities is None or odds_matrix is None:
            st.error("❌ Datos incompletos. Faltan probabilidades o cuotas")
            return

        # 2. Estructura de Pestañas
        tabs = st.tabs([
            "📊 Análisis ACBE",
            "🧮 Sistema S73", 
            "🏆 Portafolio Elite",
            "📈 Backtesting",
            "💾 Exportar"
        ])

        # 3. Recuperar Estado Actual
        s73_state = st.session_state.get('s73_results')
        elite_state = st.session_state.get('elite_results')
        backtest_state = st.session_state.get('backtest_results')

        # ===== PESTAÑA 1: ANÁLISIS ACBE =====
        with tabs[0]:
            self.render_acbe_analysis(
                probabilities, odds_matrix, normalized_entropies, config
            )

        # ===== PESTAÑA 2: SISTEMA S73 =====
        with tabs[1]:
            if s73_state is None:
                st.info("El sistema S73 no ha sido generado aún.")
                
                # --- BOTÓN DE GENERACIÓN ---
                if st.button("⚙️ Generar Sistema S73 (Cobertura 2 Errores)", type="primary"):
                    with st.spinner("Calculando combinaciones óptimas..."):
                        # A. Generar datos
                        generated_data = self.generate_s73_system(
                            probabilities, odds_matrix, normalized_entropies, config
                        )
                        
                        # B. Validar y Guardar
                        if generated_data is not None:
                            # Guardamos en session_state usando el mismo nombre consistente
                            st.session_state.s73_results = generated_data
                            st.session_state.s73_executed = True
                            
                            # C. Feedback visual y recarga
                            st.success("✅ ¡Sistema generado correctamente! Cargando tabla...")
                            time.sleep(0.5) # Pequeña pausa para ver el mensaje
                            st.rerun()      # FORZAR RECARGA (Soluciona el "no pasa nada")
            else:
                # Si ya existen datos, mostramos la interfaz
                col_btn, _ = st.columns([1, 3])
                with col_btn:
                    if st.button("🔄 Regenerar Sistema"):
                        st.session_state.s73_results = None
                        st.rerun()
                
                self.render_s73_system_detailed(s73_state, config)

        # ===== PESTAÑA 3: PORTAFOLIO ELITE =====
        with tabs[2]:
            if s73_state:
                # Si no existe Elite, botón para crearlo
                if elite_state is None:
                    st.info("El Portafolio Elite no ha sido calculado.")
                    if st.button("🏆 Calcular Portafolio Elite"):
                         elite_data = self.render_elite_portfolio(
                            s73_state, probabilities, odds_matrix, normalized_entropies, config
                        )
                         if elite_data:
                             st.session_state.elite_results = elite_data
                             st.rerun()
                else:
                    # Si ya existe, mostrarlo (pasando los datos correctamente)
                    # Nota: render_elite_portfolio en tu código anterior devolvía datos, 
                    # aquí llamamos a la visualización directa:
                    self.render_elite_results(elite_state, elite_state['metrics']['elite'])
                    
                    if st.button("🔄 Recalcular Elite"):
                        st.session_state.elite_results = None
                        st.rerun()
            else:
                st.warning("⚠️ Primero debes generar el Sistema S73 en la pestaña anterior.")

        # ===== PESTAÑA 4: BACKTESTING =====
        with tabs[3]:
            if s73_state:
                backtest_results = self.render_backtesting(
                    s73_state, elite_state, probabilities, odds_matrix, normalized_entropies, config
                )
            else:
                st.warning("⚠️ Genera el Sistema S73 para habilitar el backtesting.")

        # ===== PESTAÑA 5: EXPORTAR =====
        with tabs[4]:
            self.render_export_section(s73_state, elite_state, backtest_state, config)
