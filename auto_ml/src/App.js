import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { DataProvider } from './context/DataContext';
import FileUpload from './components/FileUpload';
import Navigation from './components/Navigation';
import AnalysisSection from './components/AnalysisSection';
import AutoMLSection from './components/AutoMLSection';
import './App.css';

function App() {
  return (
    <DataProvider>
      <Router>
        <div className="App">
          <header className="App-header">
            <h1>Auto ML Platform</h1>
            <p>Upload a CSV file and get AI-powered insights</p>

            <FileUpload />
            <Navigation />

            <Routes>
              <Route path="/" element={<Navigate to="/analysis" replace />} />
              <Route path="/analysis" element={<AnalysisSection />} />
              <Route path="/automl" element={<AutoMLSection />} />
            </Routes>
          </header>
        </div>
      </Router>
    </DataProvider>
  );
}

export default App;
