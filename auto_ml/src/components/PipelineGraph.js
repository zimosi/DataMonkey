import React from 'react';
import './PipelineGraph.css';

function PipelineGraph({ graphData, onStageClick, selectedStage, isRunning }) {
  if (!graphData || !graphData.nodes) {
    return <div>Loading...</div>;
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return '✅';
      case 'in_progress':
        return '⏳';
      case 'failed':
        return '❌';
      case 'pending':
      default:
        return '⭕';
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed':
        return '#4caf50';
      case 'in_progress':
        return '#ff9800';
      case 'failed':
        return '#f44336';
      case 'pending':
      default:
        return '#9e9e9e';
    }
  };

  return (
    <div className="pipeline-graph">
      <h3>ML Pipeline Stages</h3>
      <div className="graph-container">
        {graphData.nodes.map((node, index) => (
          <React.Fragment key={node.id}>
            <div
              className={`pipeline-node ${selectedStage === node.id ? 'selected' : ''} ${
                node.status === 'in_progress' ? 'pulsing' : ''
              }`}
              onClick={() => onStageClick(node.id)}
              style={{
                borderColor: getStatusColor(node.status),
              }}
            >
              <div className="node-header">
                <span className="node-icon">{getStatusIcon(node.status)}</span>
                <span className="node-label">{node.label}</span>
              </div>
              <p className="node-description">{node.description}</p>
              <div className="node-status">
                <span
                  className="status-badge"
                  style={{
                    backgroundColor: getStatusColor(node.status),
                  }}
                >
                  {node.status}
                </span>
              </div>
              {node.metrics && Object.keys(node.metrics).length > 0 && (
                <div className="node-metrics">
                  {Object.entries(node.metrics).slice(0, 2).map(([key, value]) => (
                    <div key={key} className="metric-item">
                      <span className="metric-key">{key}:</span>
                      <span className="metric-value">
                        {typeof value === 'number' ? value.toFixed(3) : value}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {index < graphData.nodes.length - 1 && (
              <div className="pipeline-arrow">
                <div className="arrow-line"></div>
                <div className="arrow-head">▼</div>
              </div>
            )}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

export default PipelineGraph;
