# Auto ML Analysis Pipeline

An AI-powered CSV analysis system that uses LangGraph and OpenAI to provide intelligent insights about your datasets.

## 🎯 What This Does

Upload a CSV file → LangGraph processes it → OpenAI analyzes it → Get comprehensive insights!

## 📁 Project Structure

```
studioProject/
├── auto_ml/                    # React frontend
│   ├── src/
│   │   ├── App.js             # Main upload component
│   │   └── App.css            # Styling
│   └── package.json
│
└── backend/                    # FastAPI backend
    ├── main.py                 # API endpoints
    ├── config.py               # OpenAI configuration
    ├── requirements.txt        # Python dependencies
    ├── agents/
    │   └── summarize_agent.py  # LLM analysis agent
    ├── graph/
    │   └── workflow.py         # LangGraph workflow
    └── uploads/                # Temporary file storage
```

## 🚀 Quick Start

### Backend Setup

1. **Navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up OpenAI API Key** (Required):
   
   Create a `.env` file in the backend directory:
   ```bash
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   ```
   
   Or create `.env` manually:
   ```env
   OPENAI_API_KEY=sk-proj-...your-key-here...
   OPENAI_MODEL=gpt-3.5-turbo
   OPENAI_MAX_TOKENS=2000
   OPENAI_TEMPERATURE=0.1
   ```
   
   > **Get your API key:** https://platform.openai.com/api-keys

5. **Run the server:**
   ```bash
   python main.py
   ```
   
   Server runs at: `http://localhost:8000`  
   API docs at: `http://localhost:8000/docs`

### Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd auto_ml
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the app:**
   ```bash
   npm start
   ```
   
   App opens at: `http://localhost:3000`

## 💡 How to Use

1. **Start both servers** (backend on port 8000, frontend on port 3000)
2. **Open** `http://localhost:3000` in your browser
3. **Select** a CSV file using the file picker
4. **Click** "Upload & Analyze"
5. **Wait** for the AI analysis (usually 5-10 seconds)
6. **View** the comprehensive insights from OpenAI!

## 🔄 System Architecture

### Data Flow

```
┌─────────────┐
│   User      │
│ Uploads CSV │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  React Frontend     │
│  (Port 3000)        │
└──────┬──────────────┘
       │ POST /api/upload
       ▼
┌─────────────────────┐
│  FastAPI Backend    │
│  (Port 8000)        │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  LangGraph          │
│  Workflow Engine    │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Summarize Agent    │
│  (LLM Analysis)     │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  OpenAI API         │
│  (GPT-3.5-turbo)    │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Analysis Results   │
│  Sent to Frontend   │
└─────────────────────┘
```

### LangGraph Workflow

```
START
  ↓
[Summarize Agent]
  ↓
  • Reads CSV data
  • Creates analysis prompt
  • Calls OpenAI API
  • Returns insights
  ↓
END
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/api/upload` | Upload CSV and get AI analysis |

### Upload Response Format

```json
{
  "message": "File uploaded and analyzed successfully.",
  "job_id": "uuid-here",
  "filename": "data.csv",
  "status": "summarized",
  "llm_analysis": "Comprehensive AI analysis text...",
  "data_summary": {
    "rows": 100,
    "columns": 5,
    "column_names": ["col1", "col2", ...]
  },
  "error": null
}
```

## ✨ Features

- ✅ **Simple CSV Upload** - Drag & drop or click to upload
- ✅ **LangGraph Integration** - Modular agent-based workflow
- ✅ **AI-Powered Analysis** - OpenAI GPT-3.5-turbo insights
- ✅ **Natural Language Output** - Easy-to-read analysis
- ✅ **Real-time Processing** - Synchronous analysis (no polling needed)
- ✅ **Clean Architecture** - Simplified, maintainable codebase
- ✅ **Modern UI** - Beautiful React interface
- ✅ **CORS Enabled** - Frontend-backend communication
- ✅ **Error Handling** - Graceful error messages

## 🧠 What the AI Analyzes

The LLM provides insights on:

1. **Dataset Description** - What the data represents
2. **Key Observations** - Important patterns and trends
3. **Data Insights** - Statistical and business insights
4. **ML Suggestions** - Potential machine learning tasks
5. **Data Quality** - Issues and recommendations

## 🔧 Technologies

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 19, Modern CSS |
| **Backend** | FastAPI, Python 3.13 |
| **Workflow** | LangGraph |
| **AI** | OpenAI GPT-3.5-turbo |
| **Data** | Pandas |
| **Async** | asyncio, async/await |

## 📦 Dependencies

### Backend (`requirements.txt`)
```
fastapi
uvicorn
python-multipart
pandas
langgraph
langchain
langchain-openai
openai
python-dotenv
```

### Frontend (`package.json`)
```
react: ^19.0.0
react-dom: ^19.0.0
```

## 🎓 Key Concepts

### LangGraph
- **StateGraph**: Defines the workflow structure
- **Nodes**: Individual processing units (agents)
- **Edges**: Connections between nodes
- **State**: Data that flows through the graph

### Workflow State
```python
class WorkflowState(TypedDict):
    job_id: str
    file_path: str
    filename: str
    raw_data: pd.DataFrame
    llm_analysis: Optional[str]
    data_summary: Optional[Dict]
    error: Optional[str]
    status: str
    started_at: datetime
    completed_at: Optional[datetime]
```

### Agent Pattern
Each agent:
1. Receives state
2. Processes data
3. Updates state
4. Returns state

## 🚧 Future Enhancements

- [ ] Add preprocessing agent (data cleaning)
- [ ] Add model selection agent
- [ ] Add training agent
- [ ] Add evaluation agent
- [ ] Add conditional branching in workflow
- [ ] Add streaming responses
- [ ] Add result visualization charts
- [ ] Add data export functionality
- [ ] Add user authentication
- [ ] Add job history and persistence

## 🐛 Troubleshooting

### "OPENAI_API_KEY environment variable is required"
- Make sure `.env` file exists in `backend/` directory
- Check that the API key is valid and has no extra spaces/newlines
- Verify the file is named exactly `.env` (not `.env.txt`)

### "Module not found" errors
- Activate the virtual environment: `source venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`

### Frontend can't connect to backend
- Ensure backend is running on port 8000
- Check CORS settings in `main.py`
- Verify frontend is using `http://localhost:8000`

### OpenAI API quota exceeded
- Check your OpenAI account billing
- Add credits at https://platform.openai.com/account/billing
- Consider using `gpt-3.5-turbo` instead of `gpt-4`

## 📝 License

This project is for educational purposes.

## 🤝 Contributing

This is a learning project. Feel free to fork and experiment!

---

**Built with ❤️ using LangGraph and OpenAI**