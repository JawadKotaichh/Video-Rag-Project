# Video RAG (Retrieval-Augmented Generation) Project

A multimodal video retrieval system that enables semantic search across video content using both text and visual queries. This project implements multiple retrieval methods (FAISS, TF-IDF, BM25) to find relevant video segments based on natural language questions or uploaded images.

## Demo

### Application Interface

![Application Interface](data/main_page.png)

The Streamlit interface provides an intuitive way to query videos using natural language or visual queries. Users can enter text questions or upload images to find relevant video segments.

### Demo Video

Watch the demo video to see the system in action:

**📹 [Click here to view and play the demo video](https://github.com/JawadKotaichh/Video-Rag-Project/blob/main/data/demo.mp4?raw=true)**

---

## Features

- **Multimodal Retrieval**: Support for both text-based and image-based queries
- **Multiple Retrieval Methods**:
  - **FAISS (Text)**: Semantic search using sentence transformers
  - **FAISS (Image)**: Visual similarity search using CLIP embeddings
  - **TF-IDF**: Traditional lexical search
  - **BM25**: Probabilistic ranking function for text retrieval
- **Video Processing**: Automatic keyframe extraction and transcript generation
- **Interactive UI**: Streamlit-based web interface for easy querying
- **Evaluation Framework**: Built-in test suite for accuracy and latency benchmarking

## Architecture

The system follows a pipeline architecture:

1. **Extraction Phase**:

   - Keyframe extraction from video at regular intervals
   - Audio transcription using Whisper ASR

2. **Embedding Phase**:

   - Text embeddings using sentence transformers
   - Image embeddings using CLIP (multimodal model)

3. **Indexing Phase**:

   - FAISS indices for fast similarity search
   - TF-IDF vectorization for lexical search
   - BM25 index for probabilistic ranking

4. **Retrieval Phase**:
   - Query processing and embedding
   - Multi-method retrieval and ranking
   - Timestamp-based video segment return

## Installation

### Prerequisites

- Python 3.8+
- FFmpeg (for video processing)

### Setup

1. **Navigate to the project directory**:

   ```bash
   cd VideoRAG
   ```

2. **Create a virtual environment**:

   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**:

   - Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

5. **Download FFmpeg** (if not already installed):
   - The project includes FFmpeg binaries in `app/ffmpeg/`
   - Or download from [FFmpeg official website](https://ffmpeg.org/download.html)

## Usage

### 1. Prepare Your Video

Place your video file in `data/video.mp4` or update the path in `Paths.py`.

### 2. Extract Content

#### Extract Keyframes:

```bash
python app/Extract/extract_keyframes.py
```

#### Extract Transcript:

```bash
python app/Extract/extract_transcript.py
```

### 3. Generate Embeddings

#### Embed Images:

```bash
python "app/Embedding Images and Texts/embed_images.py"
```

#### Embed Text:

```bash
python "app/Embedding Images and Texts/embed_text.py"
```

### 4. Build Retrieval Indices

```bash
# Build FAISS text index
python "app/Building Retreival Techniques/build_faiss_text.py"

# Build FAISS image index
python "app/Building Retreival Techniques/build_faiss_image.py"

# Build TF-IDF index
python "app/Building Retreival Techniques/build_tfidf.py"

# Build BM25 index
python "app/Building Retreival Techniques/build_bm25.py"
```

### 5. Adjust the path in `Paths.py`

- Video file path
- Transcript paths
- Embedding output paths
- Index output paths
- Keyframe extraction interval

### 6. Run the Application

```bash
streamlit run app/main.py
```

The application will open in your browser at `http://localhost:8501`.
For more details, see the project documentation in `rag.pdf`.
