"""VIRALYTIC Dashboard — Enhanced with Smart Collector"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import sys
import os
import re
import time

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viralyctic.core.ingestor import ViralycticIngestor
from viralyctic.core.pattern_engine import PatternEngine
from viralyctic.core.predictor import ViralPredictor
from viralyctic.core.optimizer import ViralOptimizer
from core.config import VideoFeatures

# Page config
st.set_page_config(
    page_title="VIRALYTIC | Viral Content Optimization",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 3rem; font-weight: 800; background: linear-gradient(45deg, #FF6B6B, #4ECDC4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .metric-card { background: #1E1E1E; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #FF6B6B; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .viral-high { color: #00C853; font-weight: bold; font-size: 1.2rem; }
    .viral-med { color: #FFD600; font-weight: bold; font-size: 1.2rem; }
    .viral-low { color: #FF1744; font-weight: bold; font-size: 1.2rem; }
    .recommendation-box { background: linear-gradient(135deg, #2D2D2D 0%, #1E1E1E 100%); padding: 1.2rem; border-radius: 10px; margin: 0.8rem 0; border: 1px solid #333; }
    .stButton>button { background: linear-gradient(45deg, #FF6B6B, #FF8E53); color: white; border: none; border-radius: 8px; padding: 0.75rem 2rem; font-weight: 600; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(255,107,107,0.4); }
    .collection-item { background: #1E1E1E; padding: 1rem; margin: 0.5rem 0; border-radius: 8px; border-left: 3px solid #4ECDC4; }
</style>
""", unsafe_allow_html=True)

# Session state
if 'analyzed_videos' not in st.session_state:
    st.session_state.analyzed_videos = []
if 'pattern_engine' not in st.session_state:
    st.session_state.pattern_engine = PatternEngine()
if 'predictor' not in st.session_state:
    st.session_state.predictor = ViralPredictor()
if 'optimizer' not in st.session_state:
    st.session_state.optimizer = ViralOptimizer(st.session_state.pattern_engine)
if 'collected_urls' not in st.session_state:
    st.session_state.collected_urls = []
if 'collection_progress' not in st.session_state:
    st.session_state.collection_progress = 0

def main():
    # Header
    st.markdown('<div class="main-header">🔥 VIRALYTIC</div>', unsafe_allow_html=True)
    st.caption("Engineering virality through pattern recognition | No API keys required")
    
    # Navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "📥 Collect Videos", 
        "🔬 Analyze & Train", 
        "🎯 Predict & Optimize",
        "📊 Database & Export"
    ])
    
    with tab1:
        collect_tab()
    with tab2:
        analyze_tab()
    with tab3:
        predict_tab()
    with tab4:
        database_tab()

def collect_tab():
    """Smart URL collector with batch processing"""
    st.header("📥 Viral Video Collection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Paste URLs (Bulk Input)")
        url_input = st.text_area(
            "Paste TikTok or YouTube Shorts URLs (one per line)",
            placeholder="https://www.tiktok.com/@user/video/123456...\nhttps://youtube.com/shorts/abcd...",
            height=200
        )
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("➕ Add to Collection", use_container_width=True):
                urls = [u.strip() for u in url_input.split('\n') if u.strip() and ('tiktok' in u or 'youtube' in u)]
                if urls:
                    st.session_state.collected_urls.extend(urls)
                    st.success(f"Added {len(urls)} URLs. Total: {len(st.session_state.collected_urls)}")
                else:
                    st.error("No valid URLs found")
        
        with col_b:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.collected_urls = []
                st.rerun()
    
    with col2:
        st.subheader("Quick Add Templates")
        st.caption("Click to add sample viral video URLs")
        
        if st.button("🎵 Add 10 Trending Music Videos", use_container_width=True):
            samples = [f"https://tiktok.com/music/trending-{i}" for i in range(10)]
            st.session_state.collected_urls.extend(samples)
            st.rerun()
            
        if st.button("🎤 Add 10 Singing Covers", use_container_width=True):
            samples = [f"https://youtube.com/shorts/cover-{i}" for i in range(10)]
            st.session_state.collected_urls.extend(samples)
            st.rerun()
    
    # Display collection
    if st.session_state.collected_urls:
        st.divider()
        st.subheader(f"Collection Status: {len(st.session_state.collected_urls)} URLs")
        
        # Progress to 500
        progress = min(len(st.session_state.collected_urls) / 500, 1.0)
        st.progress(progress, text=f"Progress to 500: {len(st.session_state.collected_urls)}/500")
        
        # Show recent additions
        with st.expander("View Collected URLs"):
            for i, url in enumerate(st.session_state.collected_urls[-10:]):
                st.markdown(f'<div class="collection-item">{i+1}. {url[:60]}...</div>', 
                           unsafe_allow_html=True)
        
        # Export option
        if st.button("💾 Export URL List to CSV"):
            df = pd.DataFrame({"url": st.session_state.collected_urls})
            csv = df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                "viral_urls_collection.csv",
                "text/csv"
            )

def analyze_tab():
    """Analysis and training interface"""
    st.header("🔬 Pattern Analysis & Model Training")
    
    # Source selection
    source = st.radio(
        "Data Source",
        ["Use Collected URLs", "Upload CSV File", "Simulation Mode (Fast Test)"],
        horizontal=True
    )
    
    urls = []
    
    if source == "Use Collected URLs":
        if len(st.session_state.collected_urls) < 10:
            st.warning(f"Only {len(st.session_state.collected_urls)} URLs collected. Need at least 10.")
            return
        urls = st.session_state.collected_urls.copy()
        st.success(f"Using {len(urls)} collected URLs")
        
    elif source == "Upload CSV File":
        uploaded = st.file_uploader("Upload CSV with 'url' column", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            if 'url' not in df.columns:
                st.error("CSV must have a 'url' column")
                return
            urls = df['url'].tolist()
            st.success(f"Loaded {len(urls)} URLs from CSV")
        else:
            return
            
    else:  # Simulation
        count = st.slider("Simulation Sample Size", 50, 500, 200, 50)
        urls = [f"https://sim.video/{i}" for i in range(count)]
        st.info(f"Will generate {count} realistic viral video simulations")
    
    # Analysis settings
    col1, col2, col3 = st.columns(3)
    with col1:
        viral_threshold = st.slider("Viral Threshold", 60, 90, 75, 5)
    with col2:
        auto_train = st.toggle("Auto-train ML Model", value=True)
    with col3:
        save_patterns = st.toggle("Save to Database", value=True)
    
    if st.button("🚀 Start Deep Analysis", type="primary", use_container_width=True):
        with st.spinner(f"Analyzing {len(urls)} videos... This may take 2-5 minutes"):
            # Initialize ingestor in simulation mode (no APIs needed)
            ingestor = ViralyticIngestor(simulation_mode=True)
            
            # Process in batches for progress
            progress_bar = st.progress(0)
            videos = []
            
            batch_size = 10
            for i in range(0, len(urls), batch_size):
                batch = urls[i:i+batch_size]
                batch_videos = ingestor.ingest_batch_urls(batch)
                videos.extend(batch_videos)
                progress_bar.progress(min((i + batch_size) / len(urls), 1.0))
                time.sleep(0.1)  # UI update
            
            # Update threshold
            st.session_state.pattern_engine.viral_threshold = viral_threshold
            
            # Analyze patterns
            analysis = st.session_state.pattern_engine.analyze_batch(videos)
            
            # Train model if enabled
            train_result = {"status": "skipped"}
            if auto_train and len(videos) >= 10:
                train_result = st.session_state.predictor.train(videos)
            
            # Store
            if save_patterns:
                st.session_state.analyzed_videos.extend(videos)
            
            # Display results
            display_analysis_results(analysis, videos, train_result)

def display_analysis_results(analysis, videos, train_result):
    """Show comprehensive analysis output"""
    st.success(f"✅ Analysis Complete | {analysis['viral_count']} viral patterns found in {analysis['total_analyzed']} videos")
    
    # Key metrics
    cols = st.columns(4)
    metrics = [
        ("🎬 Videos", analysis['total_analyzed'], "Total analyzed"),
        ("🔥 Viral", analysis['viral_count'], f"Score ≥{st.session_state.pattern_engine.viral_threshold}"),
        ("📈 Rate", f"{analysis['viral_rate']:.1%}", "Viral hit rate"),
        ("🎯 Patterns", analysis['patterns_discovered'], "Traces identified")
    ]
    for col, (label, value, help_text) in zip(cols, metrics):
        with col:
            st.metric(label, value, help=help_text)
    
    # ML Model status
    if train_result.get('status') == 'trained':
        st.info(f"🤖 ML Model Trained | Accuracy: {train_result['accuracy']:.1%} | Samples: {train_result['samples_used']}")
        
        # Feature importance chart
        if train_result.get('feature_importance'):
            feat_imp = train_result['feature_importance']
            fig = px.bar(
                x=list(feat_imp.keys()), 
                y=list(feat_imp.values()),
                title="What Makes Videos Viral (Feature Importance)",
                labels={'x': 'Feature', 'y': 'Impact on Virality'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Tabs for detailed results
    tab1, tab2, tab3 = st.tabs(["🔍 Discovered Patterns", "📊 Distributions", "📋 Raw Data"])
    
    with tab1:
        st.subheader("Viral Traces Discovered")
        
        # Sort by correlation strength
        patterns = sorted(analysis['patterns'], key=lambda x: x['correlation'], reverse=True)
        
        for i, pattern in enumerate(patterns[:10]):
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.markdown(f"**{i+1}. {pattern['feature']}** = `{pattern['value']}`")
                with col2:
                    st.progress(pattern['correlation'], 
                              text=f"Strength: {pattern['correlation']:.0%}")
                with col3:
                    st.caption(f"Found in {pattern['occurrence']:.0%} of viral videos")
                
                # Confidence badge
                conf = pattern.get('confidence', 0.5)
                color = '#00C853' if conf > 0.8 else '#FFD600' if conf > 0.6 else '#FF1744'
                st.markdown(f"<span style='color:{color};'>● Confidence: {conf:.0%}</span>", 
                           unsafe_allow_html=True)
                st.divider()
        
        st.subheader("💡 Strategic Insights")
        for insight in analysis['key_insights']:
            st.info(insight)
    
    with tab2:
        # Platform distribution
        platform_data = analysis['platform_breakdown']
        if platform_data:
            fig1 = px.pie(
                values=list(platform_data.values()), 
                names=list(platform_data.keys()),
                title="Platform Distribution",
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        # Viral score distribution
        scores = [v.viral_score for v in videos]
        fig2 = px.histogram(
            x=scores, 
            nbins=20, 
            title="Viral Score Distribution",
            labels={"x": "Viral Score", "count": "Videos"},
            color_discrete_sequence=['#FF6B6B']
        )
        fig2.add_vline(
            x=st.session_state.pattern_engine.viral_threshold, 
            line_dash="dash", 
            line_color="white",
            annotation_text="Viral Threshold"
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Duration vs Virality scatter
        df_viz = pd.DataFrame([
            {"duration": v.duration_sec, "viral_score": v.viral_score, 
             "platform": v.platform, "hook": v.hook_type}
            for v in videos
        ])
        fig3 = px.scatter(
            df_viz, x="duration", y="viral_score", color="platform", symbol="hook",
            title="Duration vs Viral Score (by Platform & Hook Type)",
            hover_data=["hook"]
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with tab3:
        df_data = [{
            "ID": v.video_id[:8],
            "Platform": v.platform,
            "Viral Score": f"{v.viral_score:.1f}",
            "Duration": f"{v.duration_sec:.0f}s",
            "Hook": v.hook_type,
            "Audio": v.audio_type,
            "Views": f"{v.view_count:,}",
            "Shares": f"{v.share_count:,}"
        } for v in videos]
        st.dataframe(pd.DataFrame(df_data), use_container_width=True, height=400)

def predict_tab():
    """Prediction and optimization interface"""
    st.header("🎯 Predict Viral Potential & Optimize")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input Video")
        
        # Quick input mode
        input_mode = st.radio("Input Mode", ["Quick (Features Only)", "Full URL Analysis"], horizontal=True)
        
        if input_mode == "Full URL Analysis":
            url = st.text_input("Video URL", placeholder="https://tiktok.com/... or https://youtube.com/shorts/...")
            video = None
            
            if url and st.button("Fetch & Analyze"):
                with st.spinner("Analyzing..."):
                    ingestor = ViralyticIngestor(simulation_mode=True)
                    video = ingestor.ingest_single_url(url)
        else:
            # Quick feature input
            with st.form("quick_features"):
                duration = st.slider("Duration (seconds)", 5, 120, 30, 5)
                hook = st.selectbox("Hook Type", 
                    ["visual_jump", "audio_drop", "text_reveal", "face_closeup", "action_start", "question_hook"])
                audio = st.select_slider("Audio Type", 
                    options=["original", "remix", "trending", "viral_sound"])
                platform = st.selectbox("Platform", ["tiktok", "youtube_shorts"])
                
                submitted = st.form_submit_button("Analyze Features")
                if submitted:
                    # Create synthetic video with these features
                    video = VideoFeatures(
                        video_id="manual_input",
                        platform=platform,
                        url="manual",
                        title="Manual Input",
                        duration_sec=duration,
                        hook_type=hook,
                        hook_duration=1.5,
                        audio_type=audio,
                        bpm=128,
                        caption="Sample caption with 🔥 emojis",
                        hashtags=["viral", "trending", "music"],
                        post_time=datetime.now().isoformat(),
                        features_vector=[duration/60, 0.8 if hook in ["visual_jump", "audio_drop"] else 0.6, 
                                       0.5, 0.9 if audio == "trending" else 0.5, 0.85, 0.6, 0.5, 0.3, 0.1, 0.8]
                    )
                    # Estimate base viral score from features
                    base_score = 50
                    if 28 <= duration <= 40: base_score += 15
                    if hook in ["visual_jump", "audio_drop"]: base_score += 15
                    if audio == "trending": base_score += 10
                    video.viral_score = min(100, base_score)
        
        if 'video' in locals() and video:
            st.session_state.current_video = video
            st.success("Video loaded. Click 'Predict & Optimize' →")
    
    with col2:
        st.subheader("Prediction Results")
        
        if st.button("🔮 Predict & Optimize", type="primary", use_container_width=True):
            if 'current_video' not in st.session_state:
                st.error("Please input a video first")
                return
            
            video = st.session_state.current_video
            
            # Predict
            prediction = st.session_state.predictor.predict(video)
            
            # Optimize
            optimization = st.session_state.optimizer.optimize(video)
            
            display_prediction_results(video, prediction, optimization)

def display_prediction_results(video, prediction, optimization):
    """Show prediction and optimization"""
    score = prediction['viral_probability']
    
    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Viral Probability", 'font': {'size': 24, 'color': 'white'}},
        delta={'reference': 50, 'valueformat': '.0f'},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#FF6B6B" if score > 75 else "#FFD600" if score > 50 else "#FF1744"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(255, 23, 68, 0.2)'},
                {'range': [50, 75], 'color': 'rgba(255, 214, 0, 0.2)'},
                {'range': [75, 100], 'color': 'rgba(0, 200, 83, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 75
            }
        }
    ))
    fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)
    
    # Verdict
    if score >= 75:
        st.success(f"🚀 HIGH VIRAL POTENTIAL: {score:.0f}%")
        st.caption("This configuration hits multiple viral traces. Strong candidate.")
    elif score >= 50:
        st.warning(f"⚡ MODERATE POTENTIAL: {score:.0f}%")
        st.caption("Optimization recommended. See suggestions below.")
    else:
        st.error(f"📉 LOW POTENTIAL: {score:.0f}%")
        st.caption("Major changes needed. Review all recommendations.")
    
    # Optimization plan
    st.divider()
    st.subheader(f"🎯 Optimization Plan (+{optimization['improvement_potential']:.0f} points potential)")
    
    # Priority recommendations
    for rec in optimization['recommendations'][:4]:
        priority_color = "#FF6B6B" if rec['priority'] == 1 else "#FFD600" if rec['priority'] == 2 else "#4ECDC4"
        
        st.markdown(f"""
        <div class="recommendation-box">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: {priority_color}; font-weight: bold;">[{rec['category'].upper()}]</span>
                <span style="background: {priority_color}33; color: {priority_color}; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem;">Priority {rec['priority']}</span>
            </div>
            <div style="margin: 8px 0;">
                <span style="text-decoration: line-through; opacity: 0.6;">{rec['current']}</span><br>
                <span style="color: #4ECDC4; font-weight: 600;">→ {rec['recommended']}</span>
            </div>
            <div style="font-size: 0.9rem; color: #aaa; margin-top: 8px;">
                📈 Impact: {rec['impact']:.0%} | {rec['how_to']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # A/B Variants
    st.subheader("🧪 A/B Test Variants")
    cols = st.columns(len(optimization['ab_variants']))
    
    for col, variant in zip(cols, optimization['ab_variants']):
        with col:
            st.markdown(f"**Variant {variant['variant_id']}**")
            st.caption(f"Focus: {variant['focus']}")
            for change in variant['changes']:
                st.write(f"• {change}")
            st.markdown(f"<span style='color: #4ECDC4;'>{variant['predicted_boost']}</span>", 
                       unsafe_allow_html=True)

def database_tab():
    """Database management and export"""
    st.header("📊 Database & System Export")
    
    db = st.session_state.pattern_engine.db
    
    # Stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Videos", len(st.session_state.analyzed_videos))
    col2.metric("Patterns Stored", len(db.patterns))
    col3.metric("Platforms", len(db.platform_stats))
    col4.metric("ML Model", "Trained" if st.session_state.predictor.is_trained else "Untrained")
    
    # Export options
    st.divider()
    st.subheader("💾 Export System State")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        if st.button("Export Database (JSON)", use_container_width=True):
            db.save_to_file("viralytic_database.json")
            with open("viralytic_database.json", "r") as f:
                st.download_button(
                    "Download Database JSON",
                    f.read(),
                    "viralytic_database.json",
                    "application/json"
                )
    
    with col_b:
        if st.session_state.analyzed_videos:
            df_export = pd.DataFrame([
                {
                    "url": v.url,
                    "platform": v.platform,
                    "viral_score": v.viral_score,
                    "duration": v.duration_sec,
                    "hook_type": v.hook_type,
                    "audio_type": v.audio_type,
                    "view_count": v.view_count
                }
                for v in st.session_state.analyzed_videos
            ])
            csv = df_export.to_csv(index=False)
            st.download_button(
                "Export Videos CSV",
                csv,
                "viral_videos_export.csv",
                "text/csv",
                use_container_width=True
            )
    
    # Platform insights
    if db.platform_stats:
        st.divider()
        st.subheader("Platform-Specific Intelligence")
        
        for platform, stats in db.platform_stats.items():
            with st.expander(f"{platform.replace('_', ' ').title()} Insights"):
                cols = st.columns(4)
                cols[0].metric("Videos Analyzed", stats['count'])
                cols[1].metric("Avg Viral Score", f"{stats['avg_viral_score']:.1f}")
                cols[2].metric("Dominant Hook", stats['top_hook'])
                cols[3].metric("Sweet Spot Duration", f"{stats['optimal_duration']:.0f}s")
                
                # Specific advice
                if platform == "tiktok":
                    st.info("💡 TikTok favors 21-34s videos with trending audio and visual jumps in first 1s")
                else:
                    st.info("💡 YouTube Shorts favors 30-45s with strong retention hooks and question-based titles")
    
    # Reset
    st.divider()
    if st.button("🗑️ Reset Entire Database", type="secondary"):
        st.session_state.analyzed_videos = []
        st.session_state.pattern_engine = PatternEngine()
        st.session_state.predictor = ViralPredictor()
        st.rerun()

if __name__ == "__main__":

    main()
