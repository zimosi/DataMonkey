import React, { useState, useEffect } from 'react';
import './App.css';
import PipelineGraph from './components/PipelineGraph';
import StageDetails from './components/StageDetails';
import FileUpload from './components/FileUpload';
import InteractiveDAG from './components/InteractiveDAG';
import AgentChat from './components/AgentChat';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [jobId, setJobId] = useState('');
  const [pipelineState, setPipelineState] = useState(null);
  const [graphData, setGraphData] = useState(null);
  const [selectedStage, setSelectedStage] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const [currentView, setCurrentView] = useState('upload'); // 'upload' or 'pipeline'
  const [viewMode, setViewMode] = useState('classic'); // 'classic' or 'dag'
  const [chatOpen, setChatOpen] = useState(false);
  const [chatAgent, setChatAgent] = useState({ id: '', name: '' });

  // Handle file upload
  const handleFileUpload = async (file) => {
    setSelectedFile(file);
    setUploadStatus('Uploading...');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (response.ok) {
        setJobId(result.job_id);
        setUploadStatus(`File uploaded successfully! Job ID: ${result.job_id}`);
      } else {
        setUploadStatus(`Error: ${result.detail || 'Upload failed'}`);
      }
    } catch (error) {
      setUploadStatus(`Error: ${error.message}`);
    }
  };

  // Run the complete pipeline
  const handleRunPipeline = async () => {
    if (!jobId) {
      alert('Please upload a file first');
      return;
    }

    setIsRunning(true);
    setCurrentView('pipeline');
    setUploadStatus('Running pipeline...');

    try {
      const response = await fetch('http://localhost:8000/api/pipeline/run', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          jobId: jobId,
          userPrompt: 'Analyze this dataset and build the best model',
        }),
      });

      const result = await response.json();

      if (response.ok) {
        setPipelineState(result.state);
        await fetchGraphData(jobId);
        setUploadStatus('Pipeline completed successfully!');
      } else {
        setUploadStatus(`Error: ${result.detail || 'Pipeline execution failed'}`);
      }
    } catch (error) {
      setUploadStatus(`Error: ${error.message}`);
    } finally {
      setIsRunning(false);
    }
  };

  // Fetch graph data
  const fetchGraphData = async (jid) => {
    try {
      const response = await fetch(`http://localhost:8000/api/pipeline/graph/${jid}`);
      const data = await response.json();
      setGraphData(data);
    } catch (error) {
      console.error('Error fetching graph data:', error);
    }
  };

  // Handle stage click
  const handleStageClick = (stageId) => {
    setSelectedStage(stageId);
  };

  // Re-run a specific stage
  const handleRerunStage = async (stageId, config) => {
    try {
      const response = await fetch('http://localhost:8000/api/pipeline/stage/rerun', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          jobId: jobId,
          stage: stageId,
          config: config || {},
        }),
      });

      const result = await response.json();

      if (response.ok) {
        setPipelineState(result.state);
        await fetchGraphData(jobId);
        alert(`Stage ${stageId} re-run successfully!`);
      } else {
        alert(`Error: ${result.detail}`);
      }
    } catch (error) {
      alert(`Error: ${error.message}`);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <div className="header-content">
          <h1>🐵 Data Monkey</h1>
          <p className="subtitle">Interactive Machine Learning Pipeline</p>
        </div>
      </header>

      <main className="main-content">
        {currentView === 'upload' ? (
          <div className="upload-view">
            <FileUpload
              onFileUpload={handleFileUpload}
              selectedFile={selectedFile}
              uploadStatus={uploadStatus}
            />

            {jobId && (
              <div className="action-section">
                <button
                  className="run-pipeline-button"
                  onClick={handleRunPipeline}
                  disabled={isRunning}
                >
                  {isRunning ? '⏳ Running Pipeline...' : '🚀 Run ML Pipeline'}
                </button>
              </div>
            )}
          </div>
        ) : (
          <div className="pipeline-view">
            <div className="pipeline-header">
              <h2>Pipeline Execution</h2>
              <div className="pipeline-controls">
                <div className="view-mode-toggle">
                  <button
                    className={viewMode === 'classic' ? 'active' : ''}
                    onClick={() => setViewMode('classic')}
                  >
                    Classic View
                  </button>
                  <button
                    className={viewMode === 'dag' ? 'active' : ''}
                    onClick={() => setViewMode('dag')}
                  >
                    DAG View
                  </button>
                </div>
                <button
                  className="back-button"
                  onClick={() => setCurrentView('upload')}
                >
                  ← Back to Upload
                </button>
                <div className="status-indicator">
                  <span className={`status-dot ${isRunning ? 'running' : 'idle'}`}></span>
                  {uploadStatus}
                </div>
              </div>
            </div>

            <div className="pipeline-container">
              <div className="graph-section">
                {viewMode === 'dag' ? (
                  <InteractiveDAG
                    jobId={jobId}
                    onNodeClick={(node) => {
                      // Open chat for clicked node if it's a main agent
                      if (node.type === 'main_agent') {
                        const agentNames = {
                          'data_understanding': 'Data Understanding Agent',
                          'preprocessing': 'Preprocessing Agent',
                          'model_selection': 'Model Selection Agent',
                          'hyperparameter_tuning': 'Hyperparameter Tuning Agent',
                          'prediction': 'Prediction Agent'
                        };
                        setChatAgent({
                          id: node.id,
                          name: agentNames[node.id] || node.label
                        });
                        setChatOpen(true);
                      }
                      setSelectedStage(node.id);
                    }}
                  />
                ) : graphData ? (
                  <PipelineGraph
                    graphData={graphData}
                    onStageClick={(stageId) => {
                      handleStageClick(stageId);
                      // Also enable chat for the stage
                      const agentNames = {
                        'data_understanding': 'Data Understanding Agent',
                        'preprocessing': 'Preprocessing Agent',
                        'model_selection': 'Model Selection Agent',
                        'hyperparameter_tuning': 'Hyperparameter Tuning Agent',
                        'prediction': 'Prediction Agent'
                      };
                      setChatAgent({
                        id: stageId,
                        name: agentNames[stageId] || stageId
                      });
                    }}
                    selectedStage={selectedStage}
                    isRunning={isRunning}
                  />
                ) : (
                  <div className="loading-graph">
                    <p>Loading pipeline...</p>
                  </div>
                )}
              </div>

              <div className="details-section">
                {selectedStage && pipelineState ? (
                  <>
                    <StageDetails
                      stage={selectedStage}
                      stageData={pipelineState[selectedStage]}
                      onRerun={handleRerunStage}
                      jobId={jobId}
                    />
                    <div className="chat-action">
                      <button
                        className="chat-button"
                        onClick={() => setChatOpen(true)}
                      >
                        💬 Chat with {chatAgent.name}
                      </button>
                    </div>
                  </>
                ) : (
                  <div className="no-selection">
                    <p>👆 Click on a pipeline stage to view details</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Agent Chat Modal */}
      <AgentChat
        jobId={jobId}
        agentId={chatAgent.id}
        agentName={chatAgent.name}
        isOpen={chatOpen}
        onClose={() => setChatOpen(false)}
      />

      <footer className="App-footer">
        <p>Data Monkey MVP - Making ML accessible through interactive pipelines</p>
      </footer>
    </div>
  );
}

export default App;
