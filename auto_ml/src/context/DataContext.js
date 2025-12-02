import React, { createContext, useContext, useState } from 'react';

const DataContext = createContext();

export const useData = () => {
  const context = useContext(DataContext);
  if (!context) {
    throw new Error('useData must be used within a DataProvider');
  }
  return context;
};

export const DataProvider = ({ children }) => {
  const [jobId, setJobId] = useState('');
  const [fileName, setFileName] = useState('');
  const [uploadStatus, setUploadStatus] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [filePath, setFilePath] = useState('');

  const resetData = () => {
    setJobId('');
    setFileName('');
    setUploadStatus('');
    setIsUploading(false);
  };

  return (
    <DataContext.Provider
      value={{
        jobId,
        setJobId,
        fileName,
        setFileName,
        uploadStatus,
        setUploadStatus,
        isUploading,
        setIsUploading,
        resetData,
        filePath,
        setFilePath,
      }}
    >
      {children}
    </DataContext.Provider>
  );
};
