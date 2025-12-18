import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def show_observability_dashboard():
    """Display professional Splunk observability dashboard."""

    st.markdown('<div class="observability-page-active"></div>', unsafe_allow_html=True)

    # Use container to isolate dashboard content and prevent leaking
    with st.container():
        st.title("Splunk Observability Dashboard")
        st.markdown("*USF Concierge - Production Monitoring*")

        st.info("""
        **Demo Dashboard:** This dashboard demonstrates production-grade observability infrastructure.
        In a live environment, these metrics would be queried from Splunk HEC in real-time.
        """)

        # Metrics row
        st.subheader("System Health Metrics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Requests/min",
                value="47",
                delta="+5%",
                help="Total request throughput"
            )

        with col2:
            st.metric(
                label="P95 Latency",
                value="1.2s",
                delta="-0.3s",
                delta_color="inverse",
                help="95th percentile response time"
            )

        with col3:
            st.metric(
                label="Error Rate",
                value="0.8%",
                delta="-0.2%",
                delta_color="inverse",
                help="Percentage of failed requests"
            )

        with col4:
            st.metric(
                label="RAG Quality Score",
                value="0.78",
                delta="+0.05",
                help="Average cross-encoder relevance score"
            )

        st.divider()

        # Charts section
        col1, col2 = st.columns(2)

        # Generate mock data for last 24 hours
        dates = pd.date_range(end=datetime.now(), periods=24, freq='h')

        with col1:
            st.subheader("Request Volume Over Time")
            st.caption("Request + regenerate events emitted from app.py")

            df_requests = pd.DataFrame({
                'Time': dates,
                'Requests': np.random.randint(30, 80, 24)
            })
            st.line_chart(df_requests.set_index('Time'))

        with col2:
            st.subheader("RAG Retrieval Quality")
            st.caption("Average rerank score from utils.rag pipeline")

            df_quality = pd.DataFrame({
                'Time': dates,
                'Avg Score': np.random.uniform(0.6, 0.9, 24)
            })
            st.line_chart(df_quality.set_index('Time'))

        # Second row of charts
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Response Latency Percentiles")
            st.caption("P50 / P95 request latency derived from Splunk request events")

            df_latency = pd.DataFrame({
                'Time': dates,
                'P50': np.random.uniform(800, 1200, 24),
                'P95': np.random.uniform(1500, 2500, 24)
            })
            st.line_chart(df_latency.set_index('Time'))

        with col2:
            st.subheader("Database Query Performance")
            st.caption("Supabase operations instrumented in utils.database")

            df_db = pd.DataFrame({
                'Time': dates,
                'Duration (ms)': np.random.uniform(30, 120, 24)
            })
            st.area_chart(df_db.set_index('Time'))

        st.divider()

        # Event category breakdown
        st.subheader("Event Distribution by Category")
        st.caption("Request, RAG, LLM, API, MCP, Database, Security, and Agent streams")

        col1, col2 = st.columns([2, 1])

        with col1:
            event_data = pd.DataFrame({
                'Category': ['Request', 'RAG', 'LLM', 'API', 'Database', 'MCP', 'Security', 'Agent'],
                'Count': [640, 480, 220, 160, 130, 110, 40, 25]
            })
            st.bar_chart(event_data.set_index('Category'), height=300)

        with col2:
            st.markdown("**Event Categories**")
            st.markdown("""
            **Request** – app.py request_start / request_complete / request_error events  
            **RAG** – Retrieval, rerank, neighbor fetch, pipeline complete events from utils.rag  
            **LLM** – Azure Phi-4 streaming + completion metrics (tokens, latency)  
            **API** – Gmail, Calendar, Hugging Face calls logged via tools/google_tools.py and utils/rag.py  
            **Database** – Supabase session/message CRUD operations from utils.database  
            **MCP** – client + server tool call telemetry in agents/mcp.py  
            **Security** – sanitize/injection logs from utils.security and content filter trips in utils.azure_llm.py  
            **Agent** – UI interactions from components/assistants (email/meeting assistants)
            """)

        st.divider()

        st.subheader("Production Infrastructure")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Observability Stack**")
            impl_data = {
                "Component": [
                    "Event Collector",
                    "Transport",
                    "Instrumented Modules",
                    "Event Categories",
                    "Fallback / Resilience"
                ],
                "Status": [
                    "utils.splunk_logger.SplunkLogger (async queue, timed context helper)",
                    "Splunk HEC (HTTPS + retry + TLS disable opt for Cloud trial)",
                    "app.py, utils/rag.py, utils/azure_llm.py, utils/database.py, utils/security.py, tools/google_tools.py, agents/mcp.py, components/assistants.py",
                    "Request, RAG, LLM, API, Database, MCP, Security, Agent UI",
                    "Batch flush (10 events / 5s), fallback to logs/splunk_fallback.log on failure"
                ]
            }
            df_impl = pd.DataFrame(impl_data)
            st.dataframe(
                df_impl,
                hide_index=True,
                width="stretch",
                column_config={
                    "Component": st.column_config.TextColumn("Component", width="small"),
                    "Status": st.column_config.TextColumn("Status", width="large")
                }
            )

        with col2:
            st.markdown("**Performance Optimizations**")
            perf_data = {
                "Optimization": [
                    "Session/Message Cache",
                    "Rerank Cache",
                    "Neighbor Batching",
                    "MCP Metrics",
                    "API Metadata"
                ],
                "Impact": [
                    "2s TTL caches cut duplicate Supabase reads",
                    "LRU cache (100 entries) avoids duplicate cross-encoder work",
                    "Single Supabase query fetches neighbor chunks instead of N round trips",
                    "Client + server instrumentation exposes per-tool latency + success",
                    "Each Gmail/Calendar call logs recipient, status, duration for audit"
                ]
            }
            df_perf = pd.DataFrame(perf_data)
            st.dataframe(
                df_perf,
                hide_index=True,
                width="stretch",
                column_config={
                    "Optimization": st.column_config.TextColumn("Optimization", width="small"),
                    "Impact": st.column_config.TextColumn("Impact", width="large")
                }
            )
