# Data Monkey MVP - Setup and Run Instructions

## Overview
Data Monkey is an interactive machine learning pipeline that helps users learn data science by doing. It features 5 automated pipeline stages with AI-powered analysis and an interactive React frontend.

## Pipeline Stages

1. **Data Understanding** - Analyzes dataset and understands semantic meaning of columns
2. **Preprocessing** - Handles missing values, outliers, encoding, scaling
3. **Model Selection** - Trains and evaluates multiple ML models
4. **Hyperparameter Tuning** - Optimizes model parameters
5. **Prediction** - Generates predictions with the tuned model

## Prerequisites

- Python 3.8 or higher
- Node.js 14 or higher
- npm or yarn
- OpenAI API key

## Backend Setup

### 1. Navigate to backend directory
```bash
cd backend
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 4. Create .env file
Create a `.env` file in the `backend` directory with your OpenAI API key:

```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_MAX_TOKENS=2000
OPENAI_TEMPERATURE=0.3
```

### 5. Start the backend server
```bash
# From the backend directory
python main.py
```

The API will be available at: `http://localhost:8000`
API Documentation (Swagger): `http://localhost:8000/docs`

## Frontend Setup

### 1. Navigate to frontend directory
```bash
cd auto_ml
```

### 2. Install Node dependencies
```bash
npm install
```

### 3. Start the React development server
```bash
npm start
```

The frontend will open automatically at: `http://localhost:3000`

## Usage Guide

### Step 1: Upload Dataset
1. Open `http://localhost:3000` in your browser
2. Drag and drop or click to upload a CSV file
3. Wait for the upload confirmation

### Step 2: Run Pipeline
1. Click the "🚀 Run ML Pipeline" button
2. The system will automatically execute all 5 pipeline stages
3. Progress is shown in real-time

### Step 3: Explore Results
1. Click on any pipeline stage node to view details
2. View metrics, visualizations, and insights for each stage
3. Examine model comparisons and performance metrics

### Step 4: Iterate (Optional)
1. For the preprocessing stage, you can click "Re-run Preprocessing" to try different configurations
2. All subsequent stages will automatically update

## API Endpoints

### Main Endpoints
- `POST /api/upload` - Upload CSV file
- `POST /api/pipeline/run` - Execute complete pipeline
- `GET /api/pipeline/state/{job_id}` - Get pipeline state
- `GET /api/pipeline/graph/{job_id}` - Get pipeline graph structure
- `POST /api/pipeline/stage/rerun` - Re-run a specific stage

## Example Test Data

You can use the included `test_data.csv` file to test the system:

```csv
name,age,salary,department,performance_score
Alice,25,50000,Engineering,85
Bob,30,60000,Engineering,92
Charlie,35,55000,Marketing,78
Diana,28,52000,Sales,88
Eve,32,65000,Engineering,95
```

## Project Structure

```
DataMonkey/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Configuration
│   ├── requirements.txt        # Python dependencies
│   ├── agents/                 # AI agents
│   │   ├── data_understanding_agent.py
│   │   ├── preprocessing_agent.py
│   │   ├── model_selection_agent.py
│   │   └── hyperparameter_agent.py
│   └── pipeline/               # Pipeline orchestration
│       ├── pipeline_state.py
│       └── pipeline_orchestrator.py
├── auto_ml/
│   ├── src/
│   │   ├── App.js              # Main React component
│   │   ├── App.css             # Main styles
│   │   └── components/         # React components
│   │       ├── FileUpload.js
│   │       ├── PipelineGraph.js
│   │       └── StageDetails.js
│   └── package.json
└── test_data.csv               # Sample dataset
```

## Features

### Interactive Pipeline Visualization
- Real-time status updates for each stage
- Visual graph showing pipeline flow
- Click any stage to view detailed results

### AI-Powered Analysis
- Semantic understanding of dataset columns
- Automatic target variable detection
- Problem type classification (regression vs classification)
- Intelligent preprocessing recommendations

### Comprehensive Visualizations
- Data distribution plots
- Correlation heatmaps
- Missing value analysis
- Model performance comparisons
- Feature importance charts

### Configurable Preprocessing
- Multiple imputation strategies
- Outlier handling
- Feature scaling options
- Categorical encoding

### Multiple Model Evaluation
- 7+ models tried automatically
- Cross-validation scores
- Train/test metrics
- Best model selection

### Hyperparameter Optimization
- Grid search or random search
- Parameter importance analysis
- Performance improvements tracking

## Troubleshooting

### Backend Issues

**Port 8000 already in use:**
```bash
# Kill the process using port 8000
lsof -ti:8000 | xargs kill -9
```

**Import errors:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**OpenAI API errors:**
- Verify your API key in `.env` file
- Check your OpenAI account has available credits
- Ensure the model name is correct (gpt-3.5-turbo or gpt-4)

### Frontend Issues

**Port 3000 already in use:**
```bash
# Kill the process
lsof -ti:3000 | xargs kill -9
```

**Component not found errors:**
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

**CORS errors:**
- Ensure backend is running on `http://localhost:8000`
- Check CORS settings in `backend/main.py`

### Data Issues

**CSV upload fails:**
- Ensure file is valid CSV format
- Check file size (recommended < 10MB for MVP)
- Verify column names have no special characters

**Pipeline fails on specific stage:**
- Check backend console logs for detailed error messages
- Verify dataset has appropriate data types
- Ensure there's a suitable target column for ML

## Performance Notes

- Initial pipeline run may take 1-3 minutes depending on dataset size
- Larger datasets (>10,000 rows) may require longer processing time
- Model training time varies with number of features and rows
- Hyperparameter tuning can take additional time (30-60 seconds)

## Development Mode

### Run with auto-reload

**Backend:**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
npm start
```

## Next Steps / Future Enhancements

- [ ] Add more ML models (XGBoost, LightGBM, Neural Networks)
- [ ] Implement feature engineering suggestions
- [ ] Add model deployment options
- [ ] Support for time-series data
- [ ] Add data augmentation techniques
- [ ] Implement A/B testing for models
- [ ] Add export functionality for trained models
- [ ] Database persistence for pipeline states
- [ ] User authentication and project management
- [ ] Collaborative features

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review backend logs in terminal
3. Check browser console for frontend errors
4. Ensure all dependencies are properly installed

## License

This is an MVP for educational purposes.
