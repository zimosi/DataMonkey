import React, { useState } from 'react';
import './StageDetails.css';

function StageDetails({ stage, stageData, onRerun, jobId }) {
  const [showConfig, setShowConfig] = useState(false);

  if (!stageData) {
    return <div>No data available</div>;
  }

  const stageTitles = {
    data_understanding: 'Data Understanding & Analysis',
    preprocessing: 'Data Preprocessing',
    model_selection: 'Model Selection & Evaluation',
    hyperparameter_tuning: 'Hyperparameter Tuning',
    prediction: 'Prediction',
  };

  const renderMetrics = () => {
    if (!stageData.metrics || Object.keys(stageData.metrics).length === 0) {
      return <p>No metrics available</p>;
    }

    return (
      <div className="metrics-grid">
        {Object.entries(stageData.metrics).map(([key, value]) => (
          <div key={key} className="metric-card">
            <div className="metric-label">{key.replace(/_/g, ' ')}</div>
            <div className="metric-value">
              {typeof value === 'number' ? value.toFixed(4) : JSON.stringify(value)}
            </div>
          </div>
        ))}
      </div>
    );
  };

  const renderVisualizations = () => {
    if (!stageData.visualizations || stageData.visualizations.length === 0) {
      return <p>No visualizations available</p>;
    }

    return (
      <div className="visualizations-grid">
        {stageData.visualizations.map((vizPath, index) => (
          <div key={index} className="visualization-item">
            <img
              src={`http://localhost:8000/${vizPath}`}
              alt={`Visualization ${index + 1}`}
              className="visualization-image"
            />
          </div>
        ))}
      </div>
    );
  };

  const renderLogs = () => {
    if (!stageData.logs || stageData.logs.length === 0) {
      return <p>No logs available</p>;
    }

    return (
      <div className="logs-container">
        {stageData.logs.map((log, index) => (
          <div key={index} className="log-entry">
            {log}
          </div>
        ))}
      </div>
    );
  };

  const renderData = () => {
    if (!stageData.data || Object.keys(stageData.data).length === 0) {
      return <p>No additional data available</p>;
    }

    // For specific stages, render custom views
    if (stage === 'data_understanding' && stageData.data.semantic_analysis) {
      const semantic = stageData.data.semantic_analysis;
      return (
        <div className="semantic-analysis">
          <h4>Dataset Purpose</h4>
          <p>{semantic.dataset_purpose || 'Unknown'}</p>

          <h4>Domain</h4>
          <p>{semantic.domain || 'Unknown'}</p>

          {semantic.insights && semantic.insights.length > 0 && (
            <>
              <h4>Key Insights</h4>
              <ul>
                {semantic.insights.map((insight, index) => (
                  <li key={index}>{insight}</li>
                ))}
              </ul>
            </>
          )}

          {semantic.suggested_target && (
            <>
              <h4>Suggested Target</h4>
              <p>
                <strong>{semantic.suggested_target}</strong> (
                {semantic.suggested_problem_type})
              </p>
            </>
          )}
        </div>
      );
    }

    if (stage === 'model_selection' && stageData.data.comparison_table) {
      const table = stageData.data.comparison_table;
      const models = Object.keys(table.Model || {});

      if (models.length === 0) {
        return <p>No model comparison data available</p>;
      }

      return (
        <div className="model-comparison">
          <h4>Model Comparison</h4>
          <table className="comparison-table">
            <thead>
              <tr>
                {Object.keys(table).map((key) => (
                  <th key={key}>{key}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {models.map((index) => (
                <tr key={index}>
                  {Object.entries(table).map(([key, values]) => (
                    <td key={key}>
                      {typeof values[index] === 'number'
                        ? values[index].toFixed(4)
                        : values[index]}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    }

    if (stage === 'hyperparameter_tuning' && stageData.data.best_params) {
      return (
        <div className="hyperparameters">
          <h4>Best Hyperparameters</h4>
          <div className="params-grid">
            {Object.entries(stageData.data.best_params).map(([key, value]) => (
              <div key={key} className="param-item">
                <span className="param-key">{key}:</span>
                <span className="param-value">{JSON.stringify(value)}</span>
              </div>
            ))}
          </div>

          {stageData.data.best_score && (
            <div className="best-score">
              <h4>Best Cross-Validation Score</h4>
              <p className="score-value">{stageData.data.best_score.toFixed(4)}</p>
            </div>
          )}
        </div>
      );
    }

    // Default: show JSON
    return (
      <pre className="data-json">{JSON.stringify(stageData.data, null, 2)}</pre>
    );
  };

  return (
    <div className="stage-details">
      <div className="details-header">
        <h3>{stageTitles[stage] || stage}</h3>
        <div className="status-badge-large" data-status={stageData.status}>
          {stageData.status}
        </div>
      </div>

      {stageData.timestamp && (
        <div className="timestamp">
          Last updated: {new Date(stageData.timestamp).toLocaleString()}
        </div>
      )}

      <div className="details-tabs">
        <div className="tab-section">
          <h4>📊 Metrics</h4>
          {renderMetrics()}
        </div>

        <div className="tab-section">
          <h4>📈 Visualizations</h4>
          {renderVisualizations()}
        </div>

        <div className="tab-section">
          <h4>📋 Details</h4>
          {renderData()}
        </div>

        <div className="tab-section">
          <h4>📝 Logs</h4>
          {renderLogs()}
        </div>
      </div>

      {stage === 'preprocessing' && (
        <div className="actions-section">
          <button
            className="rerun-button"
            onClick={() => {
              const config = {
                handle_missing: true,
                handle_outliers: true,
                scale_features: true,
              };
              onRerun(stage, config);
            }}
          >
            🔄 Re-run Preprocessing
          </button>
        </div>
      )}
    </div>
  );
}

export default StageDetails;
