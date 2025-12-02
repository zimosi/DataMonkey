import React, { useState } from 'react';
import { useData } from '../context/DataContext';
import './FileUpload.css';

const FileUpload = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [llmResponse, setLlmResponse] = useState('');
  const {
    setJobId,
    setFileName,
    uploadStatus,
    setUploadStatus,
    isUploading,
    setIsUploading,
    setFilePath,
  } = useData();

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
      setFileName(result.filename);
      setFilePath(result.filePath)

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
    const fileInput = document.getElementById('fileInput');
    if (fileInput) fileInput.value = '';
  };

  return (
    <div className="card upload-container">
      <h2>Upload CSV File</h2>
      <div className="file-upload-area">
        <input
          id="fileInput"
          type="file"
          accept=".csv"
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />
        <label htmlFor="fileInput" className="file-upload-label">
          {selectedFile ? (
            <>
              <span style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>✅</span>
              <span>{selectedFile.name}</span>
            </>
          ) : (
            <>
              <span>Click to select CSV file</span>
              <span style={{ fontSize: '0.85rem', marginTop: '0.5rem', opacity: 0.7 }}>
                or drag and drop
              </span>
            </>
          )}
        </label>
      </div>

      {selectedFile && (
        <div className="file-info">
          <div>
            <p><strong>File:</strong> {selectedFile.name}</p>
            <p><strong>Size:</strong> {(selectedFile.size / 1024).toFixed(2)} KB</p>
          </div>
        </div>
      )}

      <div className="button-group">
        <button
          onClick={handleUpload}
          disabled={!selectedFile || isUploading}
          className="btn-primary"
        >
          {isUploading ? 'Analyzing...' : 'Upload & Analyze'}
        </button>
        {selectedFile && (
          <button onClick={handleReset} className="btn-secondary">
            ↻ Reset
          </button>
        )}
      </div>

      {uploadStatus && !llmResponse && (
        <div className="upload-status">
          {uploadStatus}
        </div>
      )}

      {llmResponse && (
        <div className="analysis-results" style={{ marginTop: '1.5rem' }}>
          <h3>Data Summary</h3>
          <div className="llm-response">
            <pre>{llmResponse}</pre>
          </div>
        </div>
      )}
    </div>
  );
};

export default FileUpload;
