# 🐵 Data Monkey - Interactive ML Pipeline

> Making machine learning accessible through interactive, educational pipelines

An AI-powered EdTech platform that helps users learn data science by doing. Upload a CSV, watch the ML pipeline execute, and learn from insights at every step.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![React](https://img.shields.io/badge/react-19.2-61dafb)
![License](https://img.shields.io/badge/license-Educational-orange)

## ✨ Features

### 🔍 **5-Stage Automated ML Pipeline**

1. **Data Understanding** - AI analyzes your dataset and understands semantic meaning
2. **Preprocessing** - Automated data cleaning with configurable options
3. **Model Selection** - Trains 7+ models and compares performance
4. **Hyperparameter Tuning** - Optimizes best model automatically
5. **Prediction** - Generates predictions with tuned model

### 🎨 **Interactive Visualization**

- Real-time pipeline progress tracking
- Click any stage to view detailed results
- 10+ types of automated visualizations
- Before/after preprocessing comparisons
- Model performance charts

### 🧠 **AI-Powered Insights**

- Semantic analysis of dataset columns
- Auto-detection of target variable
- Problem type identification (classification/regression)
- Data quality assessment
- Preprocessing recommendations

### 🛠️ **Configurable & Interactive**

- Adjust preprocessing parameters
- Re-run stages with different configurations
- Compare model performances
- View feature importance
- Export results

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 14+
- OpenAI API key

### Installation

1. **Clone the repository:**
```bash
cd DataMonkey
```

2. **Set up backend:**
```bash
cd backend
pip install -r requirements.txt
```

3. **Create `.env` file:**
```bash
echo "OPENAI_API_KEY=your_key_here" > .env
```

4. **Set up frontend:**
```bash
cd auto_ml
npm install
```

### Running the Application

**Option 1: Automated (Recommended)**
```bash
chmod +x run.sh
./run.sh
```

**Option 2: Manual**

Terminal 1 (Backend):
```bash
cd backend
python main.py
```

Terminal 2 (Frontend):
```bash
cd auto_ml
npm start
```

### Access the Application

- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`

## 📖 How to Use

1. **Upload Dataset**
   - Drag & drop or click to upload CSV file
   - System validates and displays file info

2. **Run Pipeline**
   - Click "Run ML Pipeline" button
   - Watch real-time progress through 5 stages

3. **Explore Results**
   - Click any stage node to view details
   - Browse metrics, visualizations, and insights
   - Read AI-generated explanations

4. **Iterate (Optional)**
   - Adjust preprocessing settings
   - Re-run specific stages
   - Compare different approaches

## 📊 Example Use Cases

### Classification Problem
```csv
name,age,experience,salary,performance_rating
Alice,25,2,50000,high
Bob,30,5,60000,medium
...
```
**Pipeline detects:** Binary/multi-class classification
**Trains:** Logistic Regression, Random Forest, SVM, etc.
**Outputs:** Accuracy, confusion matrix, feature importance

### Regression Problem
```csv
date,temperature,humidity,wind_speed,energy_consumption
2024-01-01,72,45,12,850
2024-01-02,68,52,8,920
...
```
**Pipeline detects:** Regression problem
**Trains:** Linear Regression, Random Forest, Gradient Boosting
**Outputs:** R² score, RMSE, prediction plots

## 🏗️ Architecture

```
┌─────────────────┐
│  React Frontend │  Interactive UI with pipeline visualization
└────────┬────────┘
         │ HTTP/REST
┌────────▼────────┐
│  FastAPI Server │  6 RESTful endpoints
└────────┬────────┘
         │
┌────────▼─────────────────────────────────┐
│        Pipeline Orchestrator             │
└──┬────┬─────┬──────────┬──────────┬──────┘
   │    │     │          │          │
┌──▼──┐ │     │          │          │
│Agent│ │     │          │          │
│  1  │ │     │          │          │
│Data │ │     │          │          │
│Undr.│ │     │          │          │
└──┬──┘ │     │          │          │
   │    │     │          │          │
   │ ┌──▼────┐│          │          │
   │ │Agent 2││          │          │
   │ │Prepro.││          │          │
   │ └──┬────┘│          │          │
   │    │     │          │          │
   │    │  ┌──▼────────┐ │          │
   │    │  │  Agent 3  │ │          │
   │    │  │   Model   │ │          │
   │    │  │ Selection │ │          │
   │    │  └──┬────────┘ │          │
   │    │     │          │          │
   │    │     │   ┌──────▼────────┐ │
   │    │     │   │    Agent 4    │ │
   │    │     │   │ Hyperparameter│ │
   │    │     │   │    Tuning     │ │
   │    │     │   └──┬────────────┘ │
   ▼    ▼     ▼      ▼              ▼
┌────────────────────────────────────────┐
│     Scikit-learn ML Models             │
└────────────────────────────────────────┘
         │
         ▼
┌────────────────────────┐
│   OpenAI GPT (LLM)    │  Semantic Analysis
└────────────────────────┘
```

## 📁 Project Structure

```
DataMonkey/
├── backend/
│   ├── main.py                      # FastAPI application
│   ├── config.py                    # Configuration management
│   ├── requirements.txt             # Python dependencies
│   │
│   ├── agents/                      # 4 specialized agents
│   │   ├── data_understanding_agent.py
│   │   ├── preprocessing_agent.py
│   │   ├── model_selection_agent.py
│   │   └── hyperparameter_agent.py
│   │
│   └── pipeline/                    # Pipeline orchestration
│       ├── pipeline_state.py        # State management
│       └── pipeline_orchestrator.py # Workflow coordination
│
├── auto_ml/                         # React frontend
│   ├── src/
│   │   ├── App.js                   # Main component
│   │   ├── App.css                  # Main styles
│   │   └── components/              # React components
│   │       ├── FileUpload.js        # File upload UI
│   │       ├── PipelineGraph.js     # Pipeline visualization
│   │       └── StageDetails.js      # Stage details display
│   └── package.json
│
├── test_data.csv                    # Sample dataset
├── run.sh                           # Quick start script
├── QUICKSTART.md                    # Quick start guide
├── SETUP_AND_RUN.md                 # Detailed setup guide
└── MVP_SUMMARY.md                   # Complete implementation summary
```

## 🧪 Testing

Try the included sample dataset:
```bash
python test_pipeline.py
```

Or upload `test_data.csv` through the UI to see the complete pipeline in action.

## 🎓 Educational Benefits

### Learn by Doing
- Hands-on experience with real datasets
- See ML concepts applied in practice
- Understand the complete data science workflow

### Transparent Process
- Not a black box - see every transformation
- AI explains "why" behind decisions
- Learn best practices through recommendations

### Build Intuition
- Visual feedback at each stage
- Compare multiple approaches
- Understand impact of different choices

### Professional Workflow
- Industry-standard ML pipeline
- Best practices in data preprocessing
- Model selection and evaluation techniques

## 📚 Documentation

- [Quick Start Guide](QUICKSTART.md) - Get started in 3 steps
- [Setup & Run Guide](SETUP_AND_RUN.md) - Detailed installation and usage
- [MVP Summary](MVP_SUMMARY.md) - Complete implementation details
- [API Documentation](http://localhost:8000/docs) - Interactive API docs (when server running)

## 🛠️ Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **LangChain** - LLM application framework
- **OpenAI GPT** - Semantic analysis and insights
- **Scikit-learn** - Machine learning algorithms
- **Pandas** - Data manipulation
- **Matplotlib/Seaborn** - Visualizations

### Frontend
- **React 19** - UI framework
- **Modern CSS** - Responsive design
- **Fetch API** - HTTP requests

## 🎯 Alignment with Research

Based on the Milestone 1 EdTech domain analysis:

✅ **Bridges Theory and Practice** - Combines academic knowledge with hands-on application

✅ **Interactive Feedback Loop** - Not just answers, but guided exploration

✅ **Builds Intuition** - Visual and explanatory approach to ML concepts

✅ **Scalable Education** - Automated mentorship through AI agents

✅ **Addresses Pain Points** - Tackles theory-practice gap and lack of personalized feedback

## 🔮 Future Enhancements

- [ ] Additional ML models (XGBoost, LightGBM, Neural Networks)
- [ ] Feature engineering assistant
- [ ] Model deployment options
- [ ] Time-series data support
- [ ] Natural language queries
- [ ] Collaborative features
- [ ] Project save/load functionality
- [ ] Database persistence
- [ ] User authentication
- [ ] Export trained models

## 🐛 Troubleshooting

See [SETUP_AND_RUN.md](SETUP_AND_RUN.md) for common issues and solutions.

**Quick Fixes:**

```bash
# Backend issues
cd backend
pip install -r requirements.txt --force-reinstall

# Frontend issues
cd auto_ml
rm -rf node_modules package-lock.json
npm install

# Port conflicts
lsof -ti:8000 | xargs kill -9  # Backend
lsof -ti:3000 | xargs kill -9  # Frontend
```

## 📝 License

This project is for educational purposes.

## 🤝 Contributing

This is an MVP for educational research. Contributions and feedback welcome!

## 👥 Team

Team 67 - Data Monkey

## 📧 Support

For questions or issues:
1. Check the [troubleshooting guide](SETUP_AND_RUN.md#troubleshooting)
2. Review [documentation](QUICKSTART.md)
3. Check backend/frontend console logs

---

**Made with ❤️ for making data science education more accessible**

[Quick Start](QUICKSTART.md) • [Full Documentation](SETUP_AND_RUN.md) • [API Docs](http://localhost:8000/docs)
