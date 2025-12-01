import React, { useRef } from 'react';
import './FileUpload.css';

function FileUpload({ onFileUpload, selectedFile, uploadStatus }) {
  const fileInputRef = useRef(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (file.type === 'text/csv' || file.name.endsWith('.csv')) {
        onFileUpload(file);
      } else {
        alert('Please select a CSV file.');
      }
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file && (file.type === 'text/csv' || file.name.endsWith('.csv'))) {
      onFileUpload(file);
    } else {
      alert('Please drop a CSV file.');
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  return (
    <div className="file-upload-container">
      <h2>Upload Your Dataset</h2>
      <p className="upload-description">
        Upload a CSV file to start your machine learning pipeline
      </p>

      <div
        className={`drop-zone ${selectedFile ? 'has-file' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />

        {selectedFile ? (
          <div className="file-info-display">
            <div className="file-icon">📊</div>
            <div className="file-details">
              <p className="file-name">{selectedFile.name}</p>
              <p className="file-size">{(selectedFile.size / 1024).toFixed(2)} KB</p>
            </div>
            <button
              className="change-file-btn"
              onClick={(e) => {
                e.stopPropagation();
                fileInputRef.current?.click();
              }}
            >
              Change File
            </button>
          </div>
        ) : (
          <div className="drop-zone-content">
            <div className="upload-icon">📁</div>
            <p className="upload-text">Drop your CSV file here</p>
            <p className="upload-subtext">or click to browse</p>
          </div>
        )}
      </div>

      {uploadStatus && (
        <div className={`upload-status ${uploadStatus.includes('Error') ? 'error' : 'success'}`}>
          {uploadStatus}
        </div>
      )}
    </div>
  );
}

export default FileUpload;
