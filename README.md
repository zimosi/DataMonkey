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

