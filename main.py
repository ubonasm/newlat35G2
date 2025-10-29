import streamlit as st
import pandas as pd
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict, Counter
import io
import matplotlib.pyplot as plt
import networkx as nx
from io import BytesIO
import base64

# AI analysis functions
def analyze_topics_with_ai(utterances, num_topics, api_key, provider, model):
    """Analyze topics using AI"""
    text = "\n".join([f"{u['No']}. {u['Speaker']}: {u['Utterance']}" for u in utterances[:50]])
    
    prompt = f"""Below is a classroom discourse transcript. Extract {num_topics} main topics from this classroom session.

Transcript:
{text}

Respond in the following JSON format:
{{
  "topics": [
    {{
      "title": "Topic title",
      "description": "Topic description",
      "utterance_ids": [list of related utterance numbers]
    }}
  ]
}}
"""
    
    if provider == "Groq":
        from groq import Groq
        client = Groq(api_key=api_key)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert in educational discourse analysis. You analyze classroom transcripts and extract topics."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        result_text = response.choices[0].message.content
        
    else:  # Gemini
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model_instance = genai.GenerativeModel(model)
        
        response = model_instance.generate_content(prompt)
        result_text = response.text
    
    import re
    json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    else:
        return json.loads(result_text)


def analyze_speaker_with_ai(speaker_name, utterances, api_key, provider, model):
    """Analyze speaker using AI"""
    text = "\n".join([f"- {u}" for u in utterances[:30]])
    
    prompt = f"""Below are utterances from a speaker named "{speaker_name}" in a classroom. Analyze this speaker's role, characteristics, and communication style.

Utterances:
{text}

Respond in the following JSON format:
{{
  "role": "Speaker's role (e.g., Teacher, Student, Facilitator)",
  "characteristics": "Speaker characteristics (about 200 characters)",
  "communication_style": "Communication style description (about 200 characters)"
}}
"""
    
    if provider == "Groq":
        from groq import Groq
        client = Groq(api_key=api_key)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert in educational discourse analysis. You analyze speaker characteristics."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        result_text = response.choices[0].message.content
        
    else:  # Gemini
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model_instance = genai.GenerativeModel(model)
        
        response = model_instance.generate_content(prompt)
        result_text = response.text
    
    import re
    json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    else:
        return json.loads(result_text)

# ページ設定
st.set_page_config(
    page_title="Classroom Discourse Analysis System",
    page_icon="📚",
    layout="wide"
)

# セッション状態の初期化
if 'utterances' not in st.session_state:
    st.session_state.utterances = []
if 'codes' not in st.session_state:
    st.session_state.codes = []
if 'categories' not in st.session_state:
    st.session_state.categories = []
if 'codings' not in st.session_state:
    st.session_state.codings = []

# タイトル
st.title("📚 Classroom Discourse Analysis System")
st.markdown("---")

# サイドバー
with st.sidebar:
    st.header("Menu")
    menu = st.radio(
        "Select Function",
        ["Data Loading", "Coding", "Analysis & Visualization", "AI Analysis", "Data Management"]
    )
    
    st.markdown("---")
    st.subheader("AI Settings")
    
    ai_provider = st.selectbox(
        "AI Provider",
        ["Groq", "Gemini"],
        key="ai_provider"
    )
    
    if ai_provider == "Groq":
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            key="groq_api_key",
            help="Get from https://console.groq.com/keys"
        )
        model = st.selectbox(
            "Model",
            ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
            key="groq_model"
        )
    else:  # Gemini
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            key="gemini_api_key",
            help="Get from https://makersuite.google.com/app/apikey"
        )
        model = st.selectbox(
            "Model",
            ["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"],
            key="gemini_model"
        )
    
    if api_key:
        st.session_state.ai_api_key = api_key
        st.session_state.ai_model = model
        st.success(f"✅ {ai_provider} API configured")

# データ読み込みセクション
if menu == "Data Loading":
    st.header("📥 Load Classroom Records")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload CSV file (requires No, Speaker, Utterance columns)",
            type=['csv']
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                required_columns = ['No', 'Speaker', 'Utterance']
                if all(col in df.columns for col in required_columns):
                    st.session_state.utterances = df.to_dict('records')
                    st.success(f"✅ Loaded {len(df)} utterances")
                    
                    st.subheader("Data Preview")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    st.subheader("📊 Basic Statistics")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Total Utterances", len(df))
                    with col_b:
                        st.metric("Number of Speakers", df['Speaker'].nunique())
                    with col_c:
                        speaker_counts = df['Speaker'].value_counts()
                        st.metric("Most Active Speaker", f"{speaker_counts.index[0]} ({speaker_counts.values[0]} times)")
                else:
                    st.error("❌ CSV file must contain No, Speaker, Utterance columns")
            except Exception as e:
                st.error(f"❌ File loading error: {str(e)}")
    
    with col2:
        st.info("""
        **CSV Format Example:**

        No,Speaker,Utterance
        1,Teacher,Hello everyone. Today we'll discuss photosynthesis.
        2,Student1,What is photosynthesis?
        3,Teacher,Great question! Photosynthesis is how plants make food.
        4,Student2,Do all plants do photosynthesis?
        5,Teacher,Most plants do. They use sunlight, water, and carbon dioxide.
        6,Student1,That's interesting!
        7,Student3,Can we see photosynthesis happening?
        8,Teacher,Not directly, but we can observe its effects.
        9,Student2,Like plants growing?
        10,Teacher,Exactly! And they release oxygen too."""
        )
        st.markdown("---")
        st.subheader("Sample Data")
        st.write("You can try the system with sample data.")
        if st.button("Load Sample Data"):
            sample_data = """No,Speaker,Utterance
1,Teacher,Hello everyone. Today we'll discuss photosynthesis.
2,Student1,What is photosynthesis?
3,Teacher,Great question! Photosynthesis is how plants make food.
4,Student2,Do all plants do photosynthesis?
5,Teacher,Most plants do. They use sunlight, water, and carbon dioxide.
6,Student1,That's interesting!
7,Student3,Can we see photosynthesis happening?
8,Teacher,Not directly, but we can observe its effects.
9,Student2,Like plants growing?
10,Teacher,Exactly! And they release oxygen too."""
            sample_df = pd.read_csv(io.StringIO(sample_data))
            st.session_state.utterances = sample_df.to_dict('records')
            st.success(f"✅ Loaded {len(sample_df)} sample utterances")
            st.rerun()

# コーディングセクション
elif menu == "Coding":
    st.header("🏷️ Coding")
    
    if not st.session_state.utterances:
        st.warning("⚠️ Please load classroom records first")
    else:
        tab1, tab2, tab3 = st.tabs(["Code Management", "Apply Codes", "Code Categories"])
        
        with tab1:
            st.subheader("Code List")
            if st.session_state.codes:
                for i, code in enumerate(st.session_state.codes):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"**{i+1}.** {code}")
                    with col2:
                        if st.button("Delete", key=f"del_code_{i}"):
                            st.session_state.codes.pop(i)
                            st.rerun()
            else:
                st.info("No codes yet. Add your first code below.")
            
            st.markdown("---")
            st.subheader("Add New Code")
            new_code = st.text_input("Code name", "")
            code_description = st.text_area("Code description (optional)", "")
            
            if st.button("Add Code", type="primary"):
                if new_code:
                    st.session_state.codes.append({
                        'name': new_code,
                        'description': code_description,
                        'category': None
                    })
                    st.success(f"✅ Added code: {new_code}")
                    st.rerun()
                else:
                    st.error("❌ Please enter a code name")
        
        with tab2:
            st.subheader("Apply Codes to Utterances")
            
            if not st.session_state.codes:
                st.warning("⚠️ Please add codes first")
            else:
                df = pd.DataFrame(st.session_state.utterances)
                
                coding_type = st.radio(
                    "Coding Type",
                    ["Single Utterance", "Range of Utterances"]
                )
                
                if coding_type == "Single Utterance":
                    utterance_no = st.number_input(
                        "Utterance No.",
                        min_value=1,
                        max_value=len(df),
                        value=1
                    )
                    
                    selected_utterance = df[df['No'] == utterance_no].iloc[0]
                    st.info(f"**{selected_utterance['Speaker']}:** {selected_utterance['Utterance']}")
                    
                    selected_code = st.selectbox(
                        "Select Code",
                        [c['name'] if isinstance(c, dict) else c for c in st.session_state.codes]
                    )
                    
                    if st.button("Apply Code", type="primary"):
                        st.session_state.codings.append({
                            'utterance_range': [utterance_no],
                            'code': selected_code,
                            'timestamp': datetime.now().isoformat()
                        })
                        st.success(f"✅ Applied code '{selected_code}' to utterance {utterance_no}")
                
                else:  # Range
                    col1, col2 = st.columns(2)
                    with col1:
                        start_no = st.number_input("Start No.", min_value=1, max_value=len(df), value=1)
                    with col2:
                        end_no = st.number_input("End No.", min_value=1, max_value=len(df), value=min(5, len(df)))
                    
                    if start_no <= end_no:
                        range_df = df[(df['No'] >= start_no) & (df['No'] <= end_no)]
                        st.write(f"**Selected {len(range_df)} utterances:**")
                        st.dataframe(range_df, use_container_width=True)
                        
                        selected_code = st.selectbox(
                            "Select Code",
                            [c['name'] if isinstance(c, dict) else c for c in st.session_state.codes]
                        )
                        
                        if st.button("Apply Code to Range", type="primary"):
                            st.session_state.codings.append({
                                'utterance_range': list(range(start_no, end_no + 1)),
                                'code': selected_code,
                                'timestamp': datetime.now().isoformat()
                            })
                            st.success(f"✅ Applied code '{selected_code}' to utterances {start_no}-{end_no}")
                    else:
                        st.error("❌ Start No. must be less than or equal to End No.")
        
        with tab3:
            st.subheader("Code Categories")
            st.write("Organize codes into higher-level categories")
            
            if st.session_state.categories:
                for cat in st.session_state.categories:
                    with st.expander(f"📁 {cat['name']}", expanded=False):
                        st.write(f"**Description:** {cat.get('description', 'N/A')}")
                        st.write(f"**Codes:** {', '.join(cat.get('codes', []))}")
            else:
                st.info("No categories yet. Create your first category below.")
            
            st.markdown("---")
            st.subheader("Create New Category")
            cat_name = st.text_input("Category name", "")
            cat_description = st.text_area("Category description", "")
            
            if st.session_state.codes:
                selected_codes = st.multiselect(
                    "Select codes for this category",
                    [c['name'] if isinstance(c, dict) else c for c in st.session_state.codes]
                )
                
                if st.button("Create Category", type="primary"):
                    if cat_name:
                        st.session_state.categories.append({
                            'name': cat_name,
                            'description': cat_description,
                            'codes': selected_codes
                        })
                        st.success(f"✅ Created category: {cat_name}")
                        st.rerun()
                    else:
                        st.error("❌ Please enter a category name")

# 分析・可視化セクション
elif menu == "Analysis & Visualization":
    st.header("📊 Analysis & Visualization")
    
    if not st.session_state.utterances:
        st.warning("⚠️ Please load classroom records first")
    else:
        df = pd.DataFrame(st.session_state.utterances)
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "Speaker Distribution", 
            "Code Analysis", 
            "Code Relationships",
            "Timeline Analysis"
        ])
        
        with tab1:
            st.subheader("Speaker Distribution")
            speaker_counts = df['Speaker'].value_counts()
            
            fig = px.bar(
                x=speaker_counts.index,
                y=speaker_counts.values,
                labels={'x': 'Speaker', 'y': 'Number of Utterances'},
                title="Utterances by Speaker"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            if st.button("Export as PNG", key="export_speaker_dist"):
                img_bytes = fig.to_image(format="png", width=1200, height=600)
                st.download_button(
                    label="Download PNG",
                    data=img_bytes,
                    file_name="speaker_distribution.png",
                    mime="image/png"
                )
        
        with tab2:
            st.subheader("Code Distribution")
            
            if st.session_state.codings:
                code_counts = Counter()
                for coding in st.session_state.codings:
                    code_counts[coding['code']] += len(coding['utterance_range'])
                
                fig = px.pie(
                    names=list(code_counts.keys()),
                    values=list(code_counts.values()),
                    title="Code Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                if st.button("Export as PNG", key="export_code_dist"):
                    img_bytes = fig.to_image(format="png", width=1200, height=600)
                    st.download_button(
                        label="Download PNG",
                        data=img_bytes,
                        file_name="code_distribution.png",
                        mime="image/png"
                    )
                
                st.markdown("---")
                st.subheader("Code Co-occurrence Matrix")
                
                # Calculate co-occurrence
                cooccurrence = defaultdict(lambda: defaultdict(int))
                for i, coding1 in enumerate(st.session_state.codings):
                    for coding2 in st.session_state.codings[i+1:]:
                        range1 = set(coding1['utterance_range'])
                        range2 = set(coding2['utterance_range'])
                        if range1 & range2:  # If ranges overlap
                            cooccurrence[coding1['code']][coding2['code']] += 1
                            cooccurrence[coding2['code']][coding1['code']] += 1
                
                if cooccurrence:
                    codes = list(set([c['code'] for c in st.session_state.codings]))
                    matrix = [[cooccurrence[c1][c2] for c2 in codes] for c1 in codes]
                    
                    fig = px.imshow(
                        matrix,
                        x=codes,
                        y=codes,
                        labels=dict(x="Code", y="Code", color="Co-occurrence"),
                        title="Code Co-occurrence Matrix",
                        color_continuous_scale="Blues"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if st.button("Export as PNG", key="export_cooccurrence"):
                        img_bytes = fig.to_image(format="png", width=1200, height=800)
                        st.download_button(
                            label="Download PNG",
                            data=img_bytes,
                            file_name="code_cooccurrence.png",
                            mime="image/png"
                        )
            else:
                st.info("No codings yet. Please apply codes first.")
        
        with tab3:
            st.subheader("Code Relationship Network")
            
            if st.session_state.codings and len(st.session_state.codings) > 1:
                st.write("This network shows relationships between codes based on co-occurrence in utterances.")
                
                # Build network graph
                G = nx.Graph()
                
                # Add nodes
                codes = list(set([c['code'] for c in st.session_state.codings]))
                G.add_nodes_from(codes)
                
                # Add edges based on co-occurrence
                cooccurrence = defaultdict(lambda: defaultdict(int))
                for i, coding1 in enumerate(st.session_state.codings):
                    for coding2 in st.session_state.codings[i+1:]:
                        range1 = set(coding1['utterance_range'])
                        range2 = set(coding2['utterance_range'])
                        overlap = len(range1 & range2)
                        if overlap > 0:
                            cooccurrence[coding1['code']][coding2['code']] += overlap
                
                # Add edges with weights
                for code1 in cooccurrence:
                    for code2 in cooccurrence[code1]:
                        if cooccurrence[code1][code2] > 0:
                            G.add_edge(code1, code2, weight=cooccurrence[code1][code2])
                
                if G.number_of_edges() > 0:
                    # Create matplotlib figure
                    fig, ax = plt.subplots(figsize=(12, 8))
                    pos = nx.spring_layout(G, k=2, iterations=50)
                    
                    # Draw network
                    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                          node_size=3000, alpha=0.9, ax=ax)
                    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
                    
                    # Draw edges with varying thickness
                    edges = G.edges()
                    weights = [G[u][v]['weight'] for u, v in edges]
                    max_weight = max(weights) if weights else 1
                    
                    nx.draw_networkx_edges(G, pos, width=[w/max_weight*5 for w in weights],
                                          alpha=0.6, edge_color='gray', ax=ax)
                    
                    # Draw edge labels
                    edge_labels = nx.get_edge_attributes(G, 'weight')
                    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)
                    
                    ax.set_title("Code Relationship Network\n(Edge thickness = co-occurrence strength)", 
                               fontsize=14, fontweight='bold')
                    ax.axis('off')
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    
                    buf = BytesIO()
                    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                    buf.seek(0)
                    
                    st.download_button(
                        label="Download Network as PNG",
                        data=buf,
                        file_name="code_network.png",
                        mime="image/png"
                    )
                    
                    plt.close()
                else:
                    st.info("No code relationships found. Codes need to co-occur in the same utterances to show relationships.")
            else:
                st.info("Need at least 2 codings to show relationships.")
        
        with tab4:
            st.subheader("Timeline Analysis")
            
            if st.session_state.codings:
                st.write("This chart shows when different codes appear throughout the discourse.")
                
                # Prepare timeline data
                timeline_data = []
                for coding in st.session_state.codings:
                    for utt_no in coding['utterance_range']:
                        timeline_data.append({
                            'Utterance No': utt_no,
                            'Code': coding['code']
                        })
                
                timeline_df = pd.DataFrame(timeline_data)
                
                # Create timeline visualization
                fig = px.scatter(
                    timeline_df,
                    x='Utterance No',
                    y='Code',
                    color='Code',
                    title='Code Timeline',
                    labels={'Utterance No': 'Utterance Number', 'Code': 'Code'},
                    height=max(400, len(timeline_df['Code'].unique()) * 50)
                )
                
                fig.update_traces(marker=dict(size=12, symbol='square'))
                fig.update_layout(showlegend=True)
                
                st.plotly_chart(fig, use_container_width=True)
                
                if st.button("Export Timeline as PNG"):
                    img_bytes = fig.to_image(format="png", width=1400, height=max(600, len(timeline_df['Code'].unique()) * 50))
                    st.download_button(
                        label="Download PNG",
                        data=img_bytes,
                        file_name="code_timeline.png",
                        mime="image/png"
                    )
                
                st.markdown("---")
                st.subheader("Code Frequency Over Time")
                
                # Create bins for utterances
                num_bins = st.slider("Number of segments", 5, 20, 10)
                max_utt = df['No'].max()
                bin_size = max_utt // num_bins
                
                timeline_df['Segment'] = (timeline_df['Utterance No'] - 1) // bin_size + 1
                
                freq_data = timeline_df.groupby(['Segment', 'Code']).size().reset_index(name='Frequency')
                
                fig = px.line(
                    freq_data,
                    x='Segment',
                    y='Frequency',
                    color='Code',
                    markers=True,
                    title='Code Frequency Across Discourse Segments',
                    labels={'Segment': 'Discourse Segment', 'Frequency': 'Code Frequency'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                if st.button("Export Frequency Chart as PNG"):
                    img_bytes = fig.to_image(format="png", width=1400, height=600)
                    st.download_button(
                        label="Download PNG",
                        data=img_bytes,
                        file_name="code_frequency_timeline.png",
                        mime="image/png"
                    )
            else:
                st.info("No codings yet. Please apply codes first.")

# AI分析セクション
elif menu == "AI Analysis":
    st.header("🤖 AI Analysis")
    
    if not st.session_state.utterances:
        st.warning("⚠️ Please load classroom records first")
    elif not hasattr(st.session_state, 'ai_api_key') or not st.session_state.ai_api_key:
        st.warning("⚠️ Please configure AI API key in the sidebar")
    else:
        tab1, tab2 = st.tabs(["📊 Topic Analysis", "👥 Speaker Analysis"])
        
        with tab1:
            st.subheader("Classroom Topic Extraction")
            st.write("Use AI to automatically extract topics from the classroom discourse")
            
            num_topics = st.slider("Number of topics to extract", 3, 10, 5)
            
            if st.button("Run Topic Analysis", type="primary"):
                with st.spinner("Analyzing with AI..."):
                    try:
                        result = analyze_topics_with_ai(
                            st.session_state.utterances,
                            num_topics,
                            st.session_state.ai_api_key,
                            st.session_state.get('ai_provider', 'Groq'),
                            st.session_state.get('ai_model', 'llama-3.3-70b-versatile')
                        )
                        
                        st.success("✅ Analysis complete")
                        
                        for i, topic in enumerate(result['topics'], 1):
                            with st.expander(f"Topic {i}: {topic['title']}", expanded=True):
                                st.write(f"**Description:** {topic['description']}")
                                st.write(f"**Related utterance numbers:** {', '.join(map(str, topic['utterance_ids']))}")
                                
                    except Exception as e:
                        st.error(f"❌ AI analysis error: {str(e)}")
        
        with tab2:
            st.subheader("Speaker Role & Stance Analysis")
            st.write("Analyze each speaker's role and stance based on their utterances")
            
            df = pd.DataFrame(st.session_state.utterances)
            speakers = df['Speaker'].unique().tolist()
            
            selected_speaker = st.selectbox("Select speaker to analyze", speakers)
            
            if st.button("Run Speaker Analysis", type="primary"):
                with st.spinner("Analyzing with AI..."):
                    try:
                        speaker_utterances = df[df['Speaker'] == selected_speaker]['Utterance'].tolist()
                        
                        result = analyze_speaker_with_ai(
                            selected_speaker,
                            speaker_utterances,
                            st.session_state.ai_api_key,
                            st.session_state.get('ai_provider', 'Groq'),
                            st.session_state.get('ai_model', 'llama-3.3-70b-versatile')
                        )
                        
                        st.success("✅ Analysis complete")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Speaker", selected_speaker)
                            st.metric("Total Utterances", len(speaker_utterances))
                        with col2:
                            st.metric("Role", result['role'])
                        
                        st.subheader("Analysis Results")
                        st.write(f"**Role:** {result['role']}")
                        st.write(f"**Characteristics:** {result['characteristics']}")
                        st.write(f"**Communication Style:** {result['communication_style']}")
                        
                    except Exception as e:
                        st.error(f"❌ AI analysis error: {str(e)}")

# データ管理セクション
elif menu == "Data Management":
    st.header("💾 Data Management")
    
    tab1, tab2 = st.tabs(["Export Data", "Import/Reset"])
    
    with tab1:
        st.subheader("Export Analysis Data")
        
        if st.session_state.utterances:
            # Export utterances
            df = pd.DataFrame(st.session_state.utterances)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Utterances (CSV)",
                data=csv,
                file_name="utterances.csv",
                mime="text/csv"
            )
            
            # Export codings
            if st.session_state.codings:
                codings_df = pd.DataFrame(st.session_state.codings)
                codings_csv = codings_df.to_csv(index=False)
                st.download_button(
                    label="Download Codings (CSV)",
                    data=codings_csv,
                    file_name="codings.csv",
                    mime="text/csv"
                )
            
            # Export complete project
            project_data = {
                'utterances': st.session_state.utterances,
                'codes': st.session_state.codes,
                'categories': st.session_state.categories,
                'codings': st.session_state.codings
            }
            project_json = json.dumps(project_data, indent=2, ensure_ascii=False)
            st.download_button(
                label="Download Complete Project (JSON)",
                data=project_json,
                file_name="analysis_project.json",
                mime="application/json"
            )
        else:
            st.info("No data to export. Please load data first.")
    
    with tab2:
        st.subheader("Import Project")
        
        uploaded_project = st.file_uploader("Upload project JSON file", type=['json'])
        if uploaded_project is not None:
            try:
                project_data = json.load(uploaded_project)
                st.session_state.utterances = project_data.get('utterances', [])
                st.session_state.codes = project_data.get('codes', [])
                st.session_state.categories = project_data.get('categories', [])
                st.session_state.codings = project_data.get('codings', [])
                st.success("✅ Project imported successfully")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Import error: {str(e)}")
        
        st.markdown("---")
        st.subheader("Reset Data")
        st.warning("⚠️ This will delete all current data")
        
        if st.button("Reset All Data", type="secondary"):
            st.session_state.utterances = []
            st.session_state.codes = []
            st.session_state.categories = []
            st.session_state.codings = []
            st.success("✅ All data has been reset")
            st.rerun()
