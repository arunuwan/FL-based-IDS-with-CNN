import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import time
import glob
from datetime import datetime
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Federated Learning Monitor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .status-running {
        color: #28a745;
        font-weight: bold;
    }
    .status-stopped {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🚀 Federated Learning Intrusion Detection Monitor</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📊 Control Panel")
st.sidebar.markdown("---")

# Auto-refresh settings
auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh", value=True)
refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 1, 30, 5)

# Display settings
show_raw_data = st.sidebar.checkbox("📋 Show Raw Data", value=False)
chart_theme = st.sidebar.selectbox("🎨 Chart Theme", ["plotly", "plotly_white", "plotly_dark"])

# Data loading functions
@st.cache_data(ttl=5)  # Cache for 5 seconds
def load_training_logs():
    """Load training logs from JSON files"""
    logs = []
    log_files = glob.glob("logs/*.json")
    
    for file in log_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                logs.extend(data)
        except:
            continue
    
    return pd.DataFrame(logs) if logs else pd.DataFrame()

@st.cache_data(ttl=5)
def load_server_weights():
    """Load server weight files info"""
    server_files = glob.glob("Server/*.pkl")
    server_info = []
    
    for file in server_files:
        try:
            stat = os.stat(file)
            server_info.append({
                'Server': os.path.basename(file).replace('.pkl', ''),
                'Last_Updated': datetime.fromtimestamp(stat.st_mtime).strftime('%H:%M:%S'),
                'Size_KB': round(stat.st_size / 1024, 2),
                'Status': '✅ Ready'
            })
        except:
            continue
    
    return pd.DataFrame(server_info)

@st.cache_data(ttl=5)
def load_model_info():
    """Load federated model information"""
    model_path = "CentralServer/fl_model.h5"
    if os.path.exists(model_path):
        stat = os.stat(model_path)
        return {
            'exists': True,
            'last_updated': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            'size_mb': round(stat.st_size / (1024 * 1024), 2)
        }
    return {'exists': False}

# Main dashboard
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

# Load data
training_logs = load_training_logs()
server_weights = load_server_weights()
model_info = load_model_info()

# Top metrics row
st.markdown("### 📈 System Overview")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="🔄 Current Epoch",
        value=f"{len(training_logs)}" if not training_logs.empty else "0",
        delta="+1" if not training_logs.empty else None
    )

with col2:
    avg_acc = training_logs['accuracy'].mean() if not training_logs.empty and 'accuracy' in training_logs.columns else 0
    st.metric(
        label="🎯 Avg Accuracy",
        value=f"{avg_acc:.2f}%" if avg_acc > 0 else "N/A"
    )

with col3:
    avg_loss = training_logs['loss'].mean() if not training_logs.empty and 'loss' in training_logs.columns else 0
    st.metric(
        label="📉 Avg Loss",
        value=f"{avg_loss:.4f}" if avg_loss > 0 else "N/A"
    )

with col4:
    st.metric(
        label="🖥️ Active Servers",
        value=f"{len(server_weights)}/5"
    )

with col5:
    status = "🟢 Running" if model_info['exists'] else "🔴 Stopped"
    st.metric(
        label="⚡ System Status",
        value=status
    )

# Server performance section
st.markdown("---")
st.markdown("### 🖥️ Server Performance")

if not server_weights.empty:
    # Server status table
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(server_weights, use_container_width=True)
    
    with col2:
        # Server status pie chart
        fig_pie = px.pie(
            values=[1] * len(server_weights),
            names=server_weights['Server'],
            title="Server Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_pie, use_container_width=True)
else:
    st.warning("⚠️ No server data available. Make sure the FL system is running.")

# Training progress section
st.markdown("---")
st.markdown("### 📊 Training Progress")

if not training_logs.empty:
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Accuracy & Loss", "🔄 Epoch Progress", "📋 Server Comparison", "📊 Raw Data"])
    
    with tab1:
        # Accuracy and Loss over time
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Training Accuracy', 'Training Loss'),
            vertical_spacing=0.1
        )
        
        if 'accuracy' in training_logs.columns:
            fig.add_trace(
                go.Scatter(
                    x=training_logs.index,
                    y=training_logs['accuracy'],
                    mode='lines+markers',
                    name='Accuracy',
                    line=dict(color='#1f77b4', width=3)
                ),
                row=1, col=1
            )
        
        if 'loss' in training_logs.columns:
            fig.add_trace(
                go.Scatter(
                    x=training_logs.index,
                    y=training_logs['loss'],
                    mode='lines+markers',
                    name='Loss',
                    line=dict(color='#ff7f0e', width=3)
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            template=chart_theme
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Epoch progress bar
        total_epochs = 300
        current_epoch = len(training_logs)
        progress = current_epoch / total_epochs
        
        st.progress(progress)
        st.markdown(f"**Progress: {current_epoch}/{total_epochs} epochs ({progress:.1%})**")
        
        # ETA calculation
        if current_epoch > 0:
            avg_time_per_epoch = 30  # Approximate seconds per epoch
            remaining_epochs = total_epochs - current_epoch
            eta_seconds = remaining_epochs * avg_time_per_epoch
            eta_hours = eta_seconds / 3600
            
            st.info(f"⏱️ Estimated time remaining: {eta_hours:.1f} hours")
    
    with tab3:
        # Server comparison
        if 'server' in training_logs.columns:
            server_comparison = training_logs.groupby('server').agg({
                'accuracy': ['mean', 'std'],
                'loss': ['mean', 'std']
            }).round(4)
            
            st.dataframe(server_comparison)
            
            # Server accuracy comparison chart
            fig_server = px.box(
                training_logs,
                x='server',
                y='accuracy',
                title='Server Accuracy Distribution',
                color='server'
            )
            st.plotly_chart(fig_server, use_container_width=True)
        else:
            st.info("Server comparison data not available yet.")
    
    with tab4:
        # Raw data table
        if show_raw_data:
            st.dataframe(training_logs, use_container_width=True)
        else:
            st.info("Enable 'Show Raw Data' in sidebar to view detailed logs.")

else:
    st.info("📝 No training data available yet. The FL system will start logging data once training begins.")
    
    # Show example of what the dashboard will look like
    st.markdown("#### 🎯 What to expect:")
    st.markdown("""
    - **Real-time metrics** as training progresses
    - **Server performance** comparison
    - **Accuracy and loss** curves over epochs
    - **Progress tracking** with ETA estimates
    - **Model status** and file information
    """)

# Model information section
st.markdown("---")
st.markdown("### 🤖 Federated Model Status")

if model_info['exists']:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success(f"✅ Model exists")
    
    with col2:
        st.info(f"📅 Last updated: {model_info['last_updated']}")
    
    with col3:
        st.info(f"💾 Size: {model_info['size_mb']} MB")
    
    # Model performance metrics
    if not training_logs.empty and 'test_accuracy' in training_logs.columns:
        latest_test_acc = training_logs['test_accuracy'].iloc[-1]
        st.metric("🎯 Latest Test Accuracy", f"{latest_test_acc:.2f}%")
else:
    st.warning("⚠️ No federated model found. Training in progress...")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🚀 Federated Learning Intrusion Detection System Monitor</p>
    <p>Built with Streamlit | Real-time monitoring dashboard</p>
</div>
""", unsafe_allow_html=True)
