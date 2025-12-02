import React, { useState } from 'react';
import { useData } from '../context/DataContext';
import './AutoMLSection.css';

const AutoMLSection = () => {
  const { jobId, fileName, filePath } = useData();
  const [prompt, setPrompt] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [currentStep, setCurrentStep] = useState(null);
  const [mlResults, setMlResults] = useState(null);
  const [error, setError] = useState(null);

  const steps = [
    { id: 'PROFILE', name: 'Profile & Target Detection', icon: '📊' },
    { id: 'SPLIT', name: 'Data Splitting', icon: '✂️' },
    { id: 'PREPROCESS', name: 'Preprocessing', icon: '🔧' },
    { id: 'SEARCH', name: 'Model Search & Optimization', icon: '🔍' },
    { id: 'EVALUATE', name: 'Model Evaluation', icon: '✅' },
  ];

  const handlePrompt = async () => {
    if (!prompt) {
      alert('Please enter a prompt first.');
      return;
    }

    if (!jobId) {
      alert('Please upload a file first.');
      return;
    }

    setIsRunning(true);
    setCurrentStep('PROFILE');
    setMlResults(null);
    setError(null);

    // Simulate step progression (in real implementation, this could be done with streaming)
    const stepOrder = ['PROFILE', 'SPLIT', 'PREPROCESS', 'SEARCH', 'EVALUATE'];
    let stepIndex = 0;

    const updateStep = () => {
      if (stepIndex < stepOrder.length) {
        setCurrentStep(stepOrder[stepIndex]);
        stepIndex++;
        setTimeout(updateStep, 1000); // Update every second for visual feedback
      }
    };
    updateStep();

    try {
      const response = await fetch('http://localhost:8000/api/ml_pipeline', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt: prompt.trim(),
          jobId: jobId,
          filePath: filePath || null
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setMlResults(result.result);
      setCurrentStep('EVALUATE'); // Final step
    } catch (error) {
      console.error('Error:', error);
      setError(error.message);
      setCurrentStep(null);
    } finally {
      setIsRunning(false);
    }
  };

  const getBestModel = () => {
    if (!mlResults?.hpo_results || mlResults.hpo_results.length === 0) return null;
    
    const task = mlResults.task;
    const higherIsBetter = task === 'classification';
    
    return mlResults.hpo_results.reduce((best, current) => {
      if (!best) return current;
      if (higherIsBetter) {
        // For classification: higher score is better (accuracy)
        return current.score > best.score ? current : best;
      } else {
        // For regression: lower score is better (RMSE)
        return current.score < best.score ? current : best;
      }
    }, null);
  };

  const bestModel = getBestModel();

  return (
    <div className="card automl-section">
      <h2>AutoML Pipeline</h2>
      <p className="section-description">
        Automatically build and train machine learning models on your dataset. Our AI will handle feature engineering, model selection, and hyperparameter tuning.
      </p>

      {!jobId && (
        <div className="no-data">
          <p style={{ fontSize: '1.2rem', marginBottom: '1rem' }}>📁</p>
          <p>Please upload a CSV file first to use AutoML.</p>
        </div>
      )}

      {jobId && (
        <div className="automl-content">
          <div className="status-message" style={{ marginTop: '1.5rem' }}>
            <input
              type="text"
              placeholder="💬 Enter ML task description (e.g., 'predict the target column')"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              disabled={isRunning}
              onKeyPress={(e) => e.key === 'Enter' && !isRunning && jobId && prompt && handlePrompt()}
            />
            <button
              onClick={handlePrompt}
              disabled={!prompt || !jobId || isRunning}
              className="btn-primary"
            >
              {isRunning ? 'Processing...' : 'Start AutoML'}
            </button>
          </div>
        </div>
      )}

      {/* Progress Steps */}
      {isRunning && (
        <div className="pipeline-progress">
          <h3>Pipeline Progress</h3>
          <div className="steps-container">
            {steps.map((step, index) => {
              const isActive = currentStep === step.id;
              const isCompleted = steps.findIndex(s => s.id === currentStep) > index;
              
              return (
                <div key={step.id} className={`step-item ${isActive ? 'active' : ''} ${isCompleted ? 'completed' : ''}`}>
                  <div className="step-icon">
                    {isCompleted ? '✅' : isActive ? '⚙️' : step.icon}
                  </div>
                  <div className="step-info">
                    <div className="step-name">{step.name}</div>
                    {isActive && <div className="step-status">Running...</div>}
                    {isCompleted && <div className="step-status">Completed</div>}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="error-message">
          <h3>❌ Error</h3>
          <p>{error}</p>
        </div>
      )}

      {/* Results Display */}
      {mlResults && !isRunning && (
        <div className="ml-results">
          {/* Profile Results */}
          {mlResults.reason_profile && (
            <div className="result-section">
              <h3>Profile & Target Detection</h3>
              <div className="result-grid">
                <div className="result-item">
                  <strong>Target Column:</strong> {mlResults.target || 'N/A'}
                </div>
                <div className="result-item">
                  <strong>Task Type:</strong> {mlResults.task || 'N/A'}
                </div>
              </div>
              <div className="llm-response">
                <strong>Reasoning:</strong>
                <pre>{mlResults.reason_profile}</pre>
              </div>
            </div>
          )}

          {/* Preprocessing Results */}
          {mlResults.preprocess_plan && (
            <div className="result-section">
              <h3>Preprocessing Plan</h3>
              <div className="preprocess-details">
                <div className="result-item">
                  <strong>Numerical Imputation:</strong> {mlResults.preprocess_plan.impute?.num || 'N/A'}
                </div>
                <div className="result-item">
                  <strong>Categorical Imputation:</strong> {mlResults.preprocess_plan.impute?.cat || 'N/A'}
                </div>
                <div className="result-item">
                  <strong>Scaling:</strong> {mlResults.preprocess_plan.scale?.num || 'N/A'}
                </div>
                <div className="result-item">
                  <strong>Encoding:</strong> {mlResults.preprocess_plan.encode?.cat || 'N/A'}
                </div>
              </div>
            </div>
          )}

          {/* Model Search Results */}
          {mlResults.hpo_results && mlResults.hpo_results.length > 0 && (
            <div className="result-section">
              <h3>Model Search Results</h3>
              <div className="model-comparison">
                <table className="models-table">
                  <thead>
                    <tr>
                      <th>Algorithm</th>
                      <th>Score ({mlResults.task === 'classification' ? 'Accuracy' : 'RMSE'})</th>
                      <th>Trial</th>
                    </tr>
                  </thead>
                  <tbody>
                    {mlResults.hpo_results
                      .sort((a, b) => {
                        const higherIsBetter = mlResults.task === 'classification';
                        return higherIsBetter ? b.score - a.score : a.score - b.score;
                      })
                      .map((result, idx) => (
                        <tr key={idx} className={bestModel && result.algo === bestModel.algo ? 'best-model' : ''}>
                          <td>
                            <strong>{result.algo}</strong>
                            {bestModel && result.algo === bestModel.algo && ' 🏆'}
                          </td>
                          <td>{result.score.toFixed(4)}</td>
                          <td>{result.trial || 'N/A'}</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Best Model Summary */}
          {bestModel && (
            <div className="result-section best-model-section">
              <h3>Best Model</h3>
              <div className="best-model-info">
                <div className="result-item">
                  <strong>Algorithm:</strong> {bestModel.algo}
                </div>
                <div className="result-item">
                  <strong>Performance:</strong> {
                    mlResults.task === 'classification' 
                      ? `${(bestModel.score * 100).toFixed(2)}% (Accuracy)`
                      : `${bestModel.score.toFixed(4)} (RMSE)`
                  }
                </div>
                {mlResults.best_model_path && (
                  <div className="result-item">
                    <strong>Model Path:</strong> {mlResults.best_model_path}
                  </div>
                )}
                <div className="download-button-container">
                  <button
                    onClick={async () => {
                      try {
                        const downloadUrl = `http://localhost:8000/api/download_model/${jobId}`;
                        const response = await fetch(downloadUrl);
                        
                        if (!response.ok) {
                          let errorMessage = 'Failed to download model';
                          try {
                            const errorData = await response.json();
                            errorMessage = errorData.detail || errorMessage;
                          } catch {
                            errorMessage = `HTTP error! status: ${response.status}`;
                          }
                          throw new Error(errorMessage);
                        }
                        
                        const blob = await response.blob();
                        const url = window.URL.createObjectURL(blob);
                        const link = document.createElement('a');
                        link.href = url;
                        link.download = `model_${jobId}.pkl`;
                        document.body.appendChild(link);
                        link.click();
                        document.body.removeChild(link);
                        window.URL.revokeObjectURL(url);
                        
                        // Show success message
                        alert('Model downloaded successfully!');
                      } catch (error) {
                        console.error('Download error:', error);
                        alert(`Failed to download model: ${error.message}`);
                      }
                    }}
                    className="btn-download"
                    disabled={!mlResults.best_model_path}
                  >
                    Download Best Model
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Schema Information */}
          {mlResults.schema && (
            <div className="result-section">
              <h3>📋 Data Schema</h3>
              <div className="schema-display">
                <table className="schema-table">
                  <thead>
                    <tr>
                      <th>Column</th>
                      <th>Data Type</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(mlResults.schema).map(([col, dtype]) => (
                      <tr key={col}>
                        <td>{col}</td>
                        <td>{dtype}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default AutoMLSection;
