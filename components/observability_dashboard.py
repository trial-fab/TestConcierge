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
            st.caption("User queries, regenerations, and assistant actions")

            df_requests = pd.DataFrame({
                'Time': dates,
                'Requests': np.random.randint(30, 80, 24)
            })
            st.line_chart(df_requests.set_index('Time'))

        with col2:
            st.subheader("RAG Retrieval Quality")
            st.caption("Cross-encoder rerank scores (0-1 scale)")

            df_quality = pd.DataFrame({
                'Time': dates,
                'Avg Score': np.random.uniform(0.6, 0.9, 24)
            })
            st.line_chart(df_quality.set_index('Time'))

        # Second row of charts
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Response Latency Percentiles")
            st.caption("P50 and P95 response times in milliseconds")

            df_latency = pd.DataFrame({
                'Time': dates,
                'P50': np.random.uniform(800, 1200, 24),
                'P95': np.random.uniform(1500, 2500, 24)
            })
            st.line_chart(df_latency.set_index('Time'))

        with col2:
            st.subheader("Database Query Performance")
            st.caption("Average Supabase operation duration (ms)")

            df_db = pd.DataFrame({
                'Time': dates,
                'Duration (ms)': np.random.uniform(30, 120, 24)
            })
            st.area_chart(df_db.set_index('Time'))

        st.divider()

        # Event category breakdown
        st.subheader("Event Distribution by Category")
        st.caption("Last hour event breakdown")

        col1, col2 = st.columns([2, 1])

        with col1:
            event_data = pd.DataFrame({
                'Category': ['Database', 'LLM', 'API', 'MCP', 'Security'],
                'Count': [312, 201, 67, 45, 3]
            })
            st.bar_chart(event_data.set_index('Category'), height=300)

        with col2:
            st.markdown("**Event Categories**")
            st.markdown("""
            **Database** - Supabase session/message queries with timing

            **LLM** - Azure Phi-4 inference with token metrics

            **API** - External API calls (Gmail, Calendar, HuggingFace)

            **MCP** - Tool invocations with full error context

            **Security** - Content filter blocks and injection detection
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
                    "Batch Processing"
                ],
                "Status": [
                    "Thread-safe async queue (10 events/batch, 5s flush)",
                    "Splunk HEC with retry logic + fallback logging",
                    "5 modules: mcp.py, google_tools.py, azure_llm.py, security.py, database.py",
                    "5 types: API, Database, LLM, MCP, Security",
                    "Audit logs batched (10/batch), neighbor chunks batched (single query)"
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
                    "Session Cache",
                    "Message Cache",
                    "Rerank Cache",
                    "Neighbor Batching",
                    "Audit Batching"
                ],
                "Impact": [
                    "2s TTL, eliminates duplicate session queries",
                    "2s TTL, eliminates duplicate message queries",
                    "LRU 100 entries, instant rerank on repeated queries",
                    "Single HTTP request vs 20+ individual queries",
                    "10 events/batch vs individual writes"
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