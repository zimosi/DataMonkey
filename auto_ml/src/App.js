import React, { useState } from 'react';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [llmResponse, setLlmResponse] = useState('');
  const [prompt, setPrompt] = useState('');
  const [jobId, setJobId] = useState('');
  const [llmResponse_data_analysis, setLlmResponse_data_analysis] = useState('');
  const [plots, setPlots] = useState('');

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (file.type === 'text/csv' || file.name.endsWith('.csv')) {
        setSelectedFile(file);
        setUploadStatus('');
        setLlmResponse('');
      } else {
        setUploadStatus('Please select a CSV file.');
        setSelectedFile(null);
      }
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setUploadStatus('Please select a file first.');
      return;
    }

    setIsUploading(true);
    setUploadStatus('Uploading and analyzing...');
    setLlmResponse('');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();
      setJobId(result.job_id);

      if (response.ok) {
        setUploadStatus('Analysis completed successfully!');
        setLlmResponse(result.llm_analysis);
      } else {
        setUploadStatus(`Error: ${result.detail || 'Upload failed'}`);
      }
    } catch (error) {
      setUploadStatus(`Error: ${error.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setUploadStatus('');
    setLlmResponse('');
    setIsUploading(false);
    // Reset file input
    const fileInput = document.getElementById('fileInput');
    if (fileInput) fileInput.value = '';
  };

  const handlePrompt = async () => {
    if (!prompt) {
      setUploadStatus('Please enter a prompt first.');
      return;
    }
    try {
      const response = await fetch('http://localhost:8000/api/prompt', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          prompt: prompt.trim(), 
          jobId: jobId })
      });
      const result = await response.json();
      
      // Get the last plot from the array (if any plots exist)
      if (result.plots && result.plots.length > 0) {
        const plotPath = result.plots[result.plots.length - 1];
        // Make sure it's a full URL
        const plotUrl = plotPath.startsWith('http') 
          ? plotPath 
          : `http://localhost:8000/${plotPath}`;
        setPlots(plotUrl);
      }
      
      setLlmResponse_data_analysis(result.result);

    } catch (error) {
      setUploadStatus(`Error: ${error.message}`);
    }
  };


  return (
    <div className="App">
      <header className="App-header">
        <h1>Auto ML analysis</h1>
        <p>Upload a CSV file and get AI-powered insights</p>
        
        <div className="upload-section">
          <div className="file-upload-area">
            <input
              id="fileInput"
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              style={{ display: 'none' }}
            />
            <label htmlFor="fileInput" className="file-upload-label">
              {selectedFile ? selectedFile.name : 'Click to select CSV file'}
            </label>
          </div>

          {selectedFile && (
            <div className="file-info">
              <p><strong>File:</strong> {selectedFile.name}</p>
              <p><strong>Size:</strong> {(selectedFile.size / 1024).toFixed(2)} KB</p>
            </div>
          )}

          <div className="button-group">
            <button 
              onClick={handleUpload} 
              disabled={!selectedFile || isUploading}
              className="upload-button"
            >
              {isUploading ? 'Analyzing...' : 'Upload & Analyze'}
            </button>
            <button onClick={handleReset} className="reset-button">
              Reset
            </button>
          </div>



          {llmResponse && (
            <div className="analysis-results">
              <div className="llm-response">
                <pre>{llmResponse}</pre>
              </div>
            </div>
          )}

          {uploadStatus && (
            <div className="status-message">
              <input type="text" placeholder="Enter a prompt" value={prompt} onChange={(e) => setPrompt(e.target.value)} />
              <button onClick={handlePrompt} disabled={!prompt || isUploading}>submit</button>
            </div>
          )}
          {llmResponse_data_analysis && (
            <div className="analysis-results">
              <div className="llm-response">
                <pre>{llmResponse_data_analysis}</pre>
              </div>
            </div>
          )}
          {plots && (
            <div className="analysis-results">
              <h3>Generated Plot:</h3>
              <img src={plots} alt="Data Visualization" style={{ maxWidth: '100%', height: 'auto' }} />
            </div>
          )}
        </div>
      </header>
    </div>
  );
}

export default App;
