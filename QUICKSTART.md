# Data Monkey - Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### Step 2: Set Up Environment

Create `backend/.env` file:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### Step 3: Run the Application

**Option A: Use the automated script**
```bash
chmod +x run.sh
./run.sh
```

**Option B: Manual startup**

Terminal 1 (Backend):
```bash
cd backend
python main.py
```

Terminal 2 (Frontend):
```bash
cd auto_ml
npm install
npm start
```

## 📊 Try It Out

1. Open browser: `http://localhost:3000`
2. Upload `test_data.csv`
3. Click "Run ML Pipeline"
4. Watch the magic happen! ✨

## 🎯 What You Get

### 5 Automated Pipeline Stages:

1. **🔍 Data Understanding**
   - Semantic analysis of columns
   - Auto-detect target variable
   - Data quality assessment
   - Visualizations: distributions, correlations

2. **🧹 Preprocessing**
   - Handle missing values
   - Remove/cap outliers
   - Encode categorical variables
   - Scale numeric features
   - Before/after visualizations

3. **🤖 Model Selection**
   - Train 7+ models automatically
   - Compare performance metrics
   - Cross-validation scores
   - Feature importance analysis

4. **⚙️ Hyperparameter Tuning**
   - Grid/Random search
   - Parameter optimization
   - Performance improvement tracking

5. **📈 Prediction**
   - Best model predictions
   - Confidence scores
   - Exportable results

## 🎨 Interactive Features

- **Click any stage** to see detailed results
- **View visualizations** for each step
- **Re-run stages** with different configurations
- **Real-time progress** updates

## 📁 Example Datasets

Try these scenarios:

**Classification (test_data.csv):**
- Predict employee performance
- 5 features, binary/multi-class target

**Regression:**
- Create your own CSV with numerical target
- System auto-detects problem type

## 🛠️ Customization

### Modify Preprocessing

In StageDetails component, click "Re-run Preprocessing" with custom config:

```javascript
{
  handle_missing: true,
  missing_strategy: "mean",  // or "median", "mode"
  handle_outliers: true,
  outlier_method: "iqr",     // or "zscore"
  scale_features: true,
  scaling_method: "standard" // or "minmax", "robust"
}
```

### Add Custom Models

Edit `backend/agents/model_selection_agent.py`:

```python
def _get_models(self, problem_type: str):
    if problem_type == "classification":
        return {
            "Your Model": YourClassifier(),
            # ... existing models
        }
```

## 📊 Understanding Results

### Metrics Explained

**Classification:**
- **Accuracy**: % of correct predictions
- **Precision**: % of positive predictions that were correct
- **Recall**: % of actual positives found
- **F1 Score**: Balance of precision and recall

**Regression:**
- **R² Score**: How well model explains variance (higher is better)
- **RMSE**: Average prediction error (lower is better)
- **MAE**: Mean absolute error (lower is better)

### Visualizations

1. **Missing Values Heatmap** - Shows data completeness
2. **Distribution Plots** - Data spread and skewness
3. **Correlation Matrix** - Feature relationships
4. **Model Comparison** - Performance across algorithms
5. **Confusion Matrix** - Classification accuracy breakdown
6. **Actual vs Predicted** - Regression fit quality
7. **Feature Importance** - Which features matter most

## 🐛 Quick Troubleshooting

**"Module not found" errors:**
```bash
cd backend
pip install -r requirements.txt
```

**"Port already in use":**
```bash
lsof -ti:8000 | xargs kill -9  # Backend
lsof -ti:3000 | xargs kill -9  # Frontend
```

**OpenAI API errors:**
- Check your `.env` file has valid API key
- Verify OpenAI account has credits

**Pipeline fails:**
- Check backend terminal for error logs
- Ensure CSV has proper format (headers, no special chars)
- Verify dataset has enough rows (minimum 10)

## 🎓 Educational Value

Data Monkey teaches ML concepts through interaction:

1. **Data Understanding** → Learn about EDA and data profiling
2. **Preprocessing** → See impact of data cleaning
3. **Model Selection** → Compare algorithms objectively
4. **Tuning** → Understand hyperparameter optimization
5. **Evaluation** → Interpret model performance

## 📚 Next Steps

- Try different datasets (classification vs regression)
- Experiment with preprocessing configurations
- Compare model performances
- Read the semantic analysis insights
- Explore feature importance rankings

## 🎯 Pro Tips

1. **Start simple**: Use test_data.csv first
2. **Read insights**: LLM provides helpful context
3. **Compare metrics**: Look at train vs test scores
4. **Check visualizations**: They tell the story
5. **Iterate**: Re-run stages with different configs

## 📖 Full Documentation

See [SETUP_AND_RUN.md](SETUP_AND_RUN.md) for complete details.

---

**Happy Learning! 🐵**
