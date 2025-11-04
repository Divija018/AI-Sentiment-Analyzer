import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline
from datetime import datetime
import re
from collections import Counter
import io

# ------------------------------
# PAGE SETUP
# ------------------------------
st.set_page_config(
    page_title="AI Sentiment Analyzer Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🤖 AI Sentiment Analyzer Pro</p>', unsafe_allow_html=True)
st.markdown("### 📊 Advanced Text Sentiment Analysis with AI-Powered Insights")

# ------------------------------
# SIDEBAR CONFIGURATION
# ------------------------------
with st.sidebar:
    st.header("⚙ Configuration")
    
    # Model selection
    model_choice = st.selectbox(
        "Select AI Model:",
        ["distilbert-base-uncased-finetuned-sst-2-english", 
         "cardiffnlp/twitter-roberta-base-sentiment-latest"],
        help="Choose the sentiment analysis model"
    )
    
    # Analysis options
    st.subheader("Analysis Options")
    show_confidence = st.checkbox("Show Confidence Scores", value=True)
    show_visualizations = st.checkbox("Show Visualizations", value=True)
    show_word_analysis = st.checkbox("Show Word Analysis", value=True)
    batch_size = st.slider("Batch Processing Size", 1, 50, 10)
    
    st.divider()
    st.info("💡 **Tip:** Upload CSV files with a 'Text' column for batch analysis")

# ------------------------------
# LOAD MODEL
# ------------------------------
@st.cache_resource
def load_model(model_name):
    try:
        return pipeline("sentiment-analysis", model=model_name)
    except:
        return pipeline("sentiment-analysis")

with st.spinner("🔄 Loading AI model..."):
    model = load_model(model_choice)
st.success("✅ Model loaded successfully!")

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def clean_text(text):
    """Clean and preprocess text"""
    text = re.sub(r'http\S+|www.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    return text.strip()

def extract_keywords(text, top_n=10):
    """Extract most common words"""
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    stop_words = {'that', 'this', 'with', 'from', 'have', 'been', 'were', 'will', 'would', 'could', 'should'}
    filtered_words = [w for w in words if w not in stop_words]
    return Counter(filtered_words).most_common(top_n)

def analyze_sentiment(text):
    """Analyze sentiment of single text"""
    cleaned_text = clean_text(text)
    if not cleaned_text:
        return None
    result = model(cleaned_text[:512])[0]  # Limit to 512 chars for efficiency
    return result

def create_sentiment_chart(df):
    """Create sentiment distribution chart"""
    sentiment_counts = df['Sentiment'].value_counts()
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Sentiment Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def create_confidence_chart(df):
    """Create confidence score distribution"""
    fig = px.histogram(
        df,
        x='Confidence',
        nbins=20,
        title="Confidence Score Distribution",
        labels={'Confidence': 'Confidence (%)', 'count': 'Frequency'},
        color_discrete_sequence=['#667eea']
    )
    return fig

def create_wordcloud_data(texts):
    """Prepare word frequency data"""
    all_text = ' '.join(texts)
    keywords = extract_keywords(all_text, top_n=15)
    return pd.DataFrame(keywords, columns=['Word', 'Frequency'])

# ------------------------------
# TAB INTERFACE
# ------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📝 Single Text Analysis", "📂 Batch File Analysis", "📊 Analytics Dashboard", "ℹ About"])

# ------------------------------
# TAB 1: SINGLE TEXT ANALYSIS
# ------------------------------
with tab1:
    st.subheader("Analyze Individual Text")
    
    # Initialize session state for text input
    if 'text_input' not in st.session_state:
        st.session_state.text_input = ""
    
    # Initialize keyboard update flag
    if 'keyboard_update' not in st.session_state:
        st.session_state.keyboard_update = 0
    
    # Text area with unique key that changes on keyboard input
    text_input = st.text_area(
        "💬 Enter your text here (Use physical keyboard OR virtual keyboard below):",
        value=st.session_state.text_input,
        height=150,
        placeholder="Type or paste your text here for sentiment analysis...",
        key=f"text_area_{st.session_state.keyboard_update}",
        help="You can type normally or use the virtual keyboard below"
    )
    
    # Update session state when user types directly
    st.session_state.text_input = text_input
    
    # Virtual Keyboard Section
    st.markdown("---")
    st.markdown("### ⌨ Virtual Keyboard with Emojis")
    
    # Keyboard mode selector
    keyboard_mode = st.radio(
        "Select Mode:",
        ["🔤 Letters & Numbers", "😀 Emojis - Emotions", "🎉 Emojis - Reactions", "🌟 Emojis - Objects"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    # Function to handle keyboard input
    def add_to_text(char):
        st.session_state.text_input += char
        st.session_state.keyboard_update += 1
    
    def delete_char():
        if st.session_state.text_input:
            st.session_state.text_input = st.session_state.text_input[:-1]
            st.session_state.keyboard_update += 1
    
    def clear_text():
        st.session_state.text_input = ""
        st.session_state.keyboard_update += 1
    
    if keyboard_mode == "🔤 Letters & Numbers":
        # Standard Keyboard Layout
        keyboard_rows = [
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'BACK'],
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
            ['Z', 'X', 'C', 'V', 'B', 'N', 'M', '!', '?', '.'],
            ['SPACE', ',', "'", '"', '-', '(', ')', 'CLEAR']
        ]
        
        for row_idx, row in enumerate(keyboard_rows):
            cols = st.columns(len(row))
            for col_idx, key in enumerate(row):
                with cols[col_idx]:
                    if key == 'SPACE':
                        st.button('␣ Space', key=f'key_{row_idx}_{col_idx}', 
                                 use_container_width=True, on_click=add_to_text, args=(' ',))
                    elif key == 'BACK':
                        st.button('⌫ Back', key=f'key_{row_idx}_{col_idx}', 
                                 use_container_width=True, on_click=delete_char)
                    elif key == 'CLEAR':
                        st.button('🗑 Clear', key=f'key_{row_idx}_{col_idx}', 
                                 use_container_width=True, on_click=clear_text)
                    else:
                        st.button(key, key=f'key_{row_idx}_{col_idx}', 
                                 use_container_width=True, on_click=add_to_text, args=(key.lower(),))
    
    elif keyboard_mode == "😀 Emojis - Emotions":
        # Emotion Emojis Keyboard
        st.markdown("**😀 Click any emoji to add it to your text**")
        emoji_rows = [
            ['😀', '😃', '😄', '😁', '😆', '😊', '😇', '🙂', '🙃', '😉'],
            ['😍', '🥰', '😘', '😗', '😚', '😙', '🥲', '😋', '😛', '😝'],
            ['🤗', '🤩', '🤔', '🤨', '😐', '😑', '😶', '🙄', '😏', '😣'],
            ['😥', '😮', '🤐', '😯', '😪', '😫', '🥱', '😴', '😌', '😔'],
            ['😞', '😓', '😩', '😢', '😭', '😤', '😠', '😡', '🤬', '🤯'],
            ['😳', '🥺', '😱', '😨', '😰', '😥', '😓', '🤗', '🤭', '😬']
        ]
        
        for row_idx, row in enumerate(emoji_rows):
            cols = st.columns(len(row))
            for col_idx, emoji in enumerate(row):
                with cols[col_idx]:
                    st.button(emoji, key=f'emoji_emo_{row_idx}_{col_idx}', 
                             use_container_width=True, on_click=add_to_text, args=(emoji,))
    
    elif keyboard_mode == "🎉 Emojis - Reactions":
        # Reaction Emojis Keyboard
        st.markdown("**👍 Click any emoji to add it to your text**")
        emoji_rows = [
            ['👍', '👎', '👌', '✌', '🤞', '🤟', '🤘', '🤙', '👈', '👉'],
            ['👆', '👇', '☝', '✋', '🤚', '🖐', '🖖', '👋', '🤝', '💪'],
            ['🙏', '✍', '💅', '🤳', '💃', '🕺', '👏', '🙌', '🤲', '🤝'],
            ['❤', '🧡', '💛', '💚', '💙', '💜', '🖤', '🤍', '🤎', '💔'],
            ['💕', '💞', '💓', '💗', '💖', '💘', '💝', '💟', '❣', '❤‍🔥'],
            ['👀', '👁', '🧠', '🫀', '🫁', '🦴', '🦷', '👄', '💋', '👅']
        ]
        
        for row_idx, row in enumerate(emoji_rows):
            cols = st.columns(len(row))
            for col_idx, emoji in enumerate(row):
                with cols[col_idx]:
                    st.button(emoji, key=f'emoji_react_{row_idx}_{col_idx}', 
                             use_container_width=True, on_click=add_to_text, args=(emoji,))
    
    elif keyboard_mode == "🌟 Emojis - Objects":
        # Object Emojis Keyboard
        st.markdown("**🎁 Click any emoji to add it to your text**")
        emoji_rows = [
            ['⭐', '🌟', '✨', '💫', '🔥', '💥', '💢', '💦', '💨', '🌈'],
            ['☀', '🌙', '⚡', '☁', '⛅', '🌤', '⛈', '🌧', '🌨', '❄'],
            ['🎉', '🎊', '🎈', '🎁', '🏆', '🥇', '🥈', '🥉', '🏅', '🎖'],
            ['⚽', '🏀', '🏈', '⚾', '🎾', '🏐', '🏉', '🎱', '🎮', '🎯'],
            ['🍕', '🍔', '🍟', '🌭', '🍿', '🧋', '☕', '🍺', '🍷', '🍰'],
            ['💻', '📱', '⌨', '🖱', '💾', '💿', '📀', '📷', '📺', '📻']
        ]
        
        for row_idx, row in enumerate(emoji_rows):
            cols = st.columns(len(row))
            for col_idx, emoji in enumerate(row):
                with cols[col_idx]:
                    st.button(emoji, key=f'emoji_obj_{row_idx}_{col_idx}', 
                             use_container_width=True, on_click=add_to_text, args=(emoji,))
    
    st.markdown("---")
    
    # Quick example and analyze section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
            if st.session_state.text_input.strip() == "":
                st.warning("⚠ Please enter some text to analyze!")
            else:
                with st.spinner("Analyzing..."):
                    result = analyze_sentiment(st.session_state.text_input)
                    
                    if result:
                        label = result['label']
                        confidence = result['score'] * 100
                        
                        # Results display
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Sentiment", label)
                        with col2:
                            st.metric("Confidence", f"{confidence:.2f}%")
                        with col3:
                            word_count = len(st.session_state.text_input.split())
                            st.metric("Word Count", word_count)
                        
                        # Visual feedback
                        if label.upper().startswith("POS"):
                            st.success("😊 Positive Sentiment Detected")
                            st.progress(confidence / 100)
                        elif label.upper().startswith("NEG"):
                            st.error("😡 Negative Sentiment Detected")
                            st.progress(confidence / 100)
                        else:
                            st.info("😐 Neutral Sentiment Detected")
                            st.progress(confidence / 100)
                        
                        # Word analysis
                        if show_word_analysis:
                            st.divider()
                            st.subheader("📝 Text Analysis")
                            keywords = extract_keywords(st.session_state.text_input, top_n=10)
                            if keywords:
                                keyword_df = pd.DataFrame(keywords, columns=['Word', 'Frequency'])
                                fig = px.bar(
                                    keyword_df,
                                    x='Frequency',
                                    y='Word',
                                    orientation='h',
                                    title="Most Common Words",
                                    color='Frequency',
                                    color_continuous_scale='Viridis'
                                )
                                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**📋 Quick Examples**")
        example_texts = {
            "😊 Positive": "I absolutely love this product! It exceeded all my expectations.",
            "😡 Negative": "This is the worst experience I've ever had. Very disappointed.",
            "😐 Neutral": "The package arrived on time. It was as described."
        }
        
        def set_example(text):
            st.session_state.text_input = text
            st.session_state.keyboard_update += 1
        
        for sentiment, text in example_texts.items():
            st.button(sentiment, key=f"example_{sentiment}", 
                     use_container_width=True, on_click=set_example, args=(text,))

# ------------------------------
# TAB 2: BATCH FILE ANALYSIS
# ------------------------------
with tab2:
    st.subheader("Batch Sentiment Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload a .txt or .csv file",
        type=["txt", "csv"],
        help="CSV files should contain a 'Text' column"
    )

    if uploaded_file is not None:
        try:
            # Load file
            if uploaded_file.name.endswith(".txt"):
                text_data = uploaded_file.read().decode("utf-8").splitlines()
                text_data = [line for line in text_data if line.strip()]  # Remove empty lines
                df = pd.DataFrame(text_data, columns=["Text"])
            elif uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                if "Text" not in df.columns:
                    st.error("❌ CSV file must contain a 'Text' column.")
                    st.stop()
            
            st.success(f"✅ File loaded successfully! ({len(df)} rows)")
            
            with st.expander("📄 Preview Data"):
                st.dataframe(df.head(10), use_container_width=True)

            if st.button("🔍 Analyze All Sentiments", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                total = len(df)
                
                for idx, text in enumerate(df["Text"]):
                    status_text.text(f"Processing: {idx + 1}/{total}")
                    progress_bar.progress((idx + 1) / total)
                    
                    result = analyze_sentiment(str(text))
                    if result:
                        results.append(result)
                    else:
                        results.append({"label": "UNKNOWN", "score": 0.0})
                
                df["Sentiment"] = [r["label"] for r in results]
                df["Confidence"] = [round(r["score"] * 100, 2) for r in results]
                df["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                status_text.empty()
                progress_bar.empty()
                st.success("✅ Analysis complete!")
                
                # Display results
                st.dataframe(df, use_container_width=True)
                
                # Summary statistics
                st.divider()
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Analyzed", len(df))
                with col2:
                    positive_count = sum(1 for s in df["Sentiment"] if s.upper().startswith("POS"))
                    st.metric("Positive", positive_count)
                with col3:
                    negative_count = sum(1 for s in df["Sentiment"] if s.upper().startswith("NEG"))
                    st.metric("Negative", negative_count)
                with col4:
                    avg_confidence = df["Confidence"].mean()
                    st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                
                # Visualizations
                if show_visualizations:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(create_sentiment_chart(df), use_container_width=True)
                    with col2:
                        st.plotly_chart(create_confidence_chart(df), use_container_width=True)
                
                # Download options
                st.divider()
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_download = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇ Download Results (CSV)",
                        data=csv_download,
                        file_name=f"sentiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    excel_buffer = io.BytesIO()
                    df.to_excel(excel_buffer, index=False, engine='openpyxl')
                    excel_download = excel_buffer.getvalue()
                    st.download_button(
                        label="⬇ Download Results (Excel)",
                        data=excel_download,
                        file_name=f"sentiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

# ------------------------------
# TAB 3: ANALYTICS DASHBOARD
# ------------------------------
with tab3:
    st.subheader("📊 Analytics & Insights")
    
    if 'df' in locals() and 'Sentiment' in df.columns:
        # Overall metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            pos_pct = (df['Sentiment'].str.upper().str.startswith('POS').sum() / len(df)) * 100
            st.metric("Positive %", f"{pos_pct:.1f}%")
        with col3:
            neg_pct = (df['Sentiment'].str.upper().str.startswith('NEG').sum() / len(df)) * 100
            st.metric("Negative %", f"{neg_pct:.1f}%")
        with col4:
            st.metric("Avg Confidence", f"{df['Confidence'].mean():.1f}%")
        
        st.divider()
        
        # Advanced visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Confidence by sentiment
            fig = px.box(
                df,
                x='Sentiment',
                y='Confidence',
                title="Confidence Distribution by Sentiment",
                color='Sentiment',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Word frequency analysis
            word_freq_df = create_wordcloud_data(df['Text'].astype(str).tolist())
            fig = px.treemap(
                word_freq_df,
                path=['Word'],
                values='Frequency',
                title="Most Frequent Words (Treemap)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed breakdown
        st.divider()
        st.subheader("🔍 Detailed Sentiment Breakdown")
        
        sentiment_summary = df.groupby('Sentiment').agg({
            'Confidence': ['mean', 'min', 'max', 'count']
        }).round(2)
        st.dataframe(sentiment_summary, use_container_width=True)
        
    else:
        st.info("📊 Upload and analyze a file in the 'Batch File Analysis' tab to see analytics dashboard.")

# ------------------------------
# TAB 4: ABOUT
# ------------------------------
with tab4:
    st.subheader("ℹ About This Application")
    
    st.markdown("""
    ### 🎯 Features
    
    - **Single Text Analysis**: Analyze individual pieces of text with instant results
    - **Batch Processing**: Upload CSV or TXT files for bulk sentiment analysis
    - **Advanced Visualizations**: Interactive charts and graphs powered by Plotly
    - **Word Analysis**: Identify most common words and phrases
    - **Multiple Export Formats**: Download results in CSV or Excel format
    - **Configurable Models**: Choose between different AI models for analysis
    - **Real-time Processing**: Fast and efficient sentiment analysis
    
    ### 🤖 AI Models
    
    This application uses state-of-the-art transformer models from Hugging Face:
    - **DistilBERT**: Efficient and accurate general-purpose sentiment analysis
    - **RoBERTa**: Advanced model fine-tuned on Twitter data for social media text
    
    ### 📊 Use Cases
    
    - **Customer Feedback Analysis**: Understand customer satisfaction from reviews
    - **Social Media Monitoring**: Track sentiment of posts and comments
    - **Product Reviews**: Analyze product feedback at scale
    - **Market Research**: Gauge public opinion on topics
    - **Content Moderation**: Identify negative or toxic content
    
    ### 🛠 Technical Stack
    
    - **Streamlit**: Web application framework
    - **Transformers**: AI models from Hugging Face
    - **Plotly**: Interactive visualizations
    - **Pandas**: Data manipulation and analysis
    
    ### 📝 Version
    **v2.0.1** - Enhanced with Working Virtual Keyboard
    
    ---
    
    💡 **Pro Tip**: For best results, provide clear and complete sentences. The AI model works best with natural language text.
    """)

# ------------------------------
# FOOTER
# ------------------------------
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🤖 AI Sentiment Analyzer Pro | Powered by Advanced Machine Learning</p>
        <p style='font-size: 0.8rem;'>Built with Streamlit • Transformers • Plotly</p>
    </div>
""", unsafe_allow_html=True)