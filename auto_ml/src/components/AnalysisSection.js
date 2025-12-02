import React, { useState } from 'react';
import { useData } from '../context/DataContext';

const AnalysisSection = () => {
  const { jobId, filePath } = useData();
  const [prompt, setPrompt] = useState('');
  const [llmResponseDataAnalysis, setLlmResponseDataAnalysis] = useState('');
  const [plots, setPlots] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handlePrompt = async () => {
    if (!prompt) {
      alert('Please enter a prompt first.');
      return;
    }

    if (!jobId) {
      alert('Please upload a file first.');
      return;
    }

    setIsAnalyzing(true);
    console.log(filePath);
    try {
      const response = await fetch('http://localhost:8000/api/prompt', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt: prompt.trim(),
          jobId: jobId,
          filePath : filePath
        }),
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

      setLlmResponseDataAnalysis(result.result);
    } catch (error) {
      alert(`Error: ${error.message}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="card analysis-section">
      <h2>Interactive Data Analysis</h2>
      <p className="section-description">
        Ask questions about your data or request visualizations. Use natural language to explore your dataset.
      </p>

      <div className="status-message">
        <input
          type="text"
          placeholder="Ask a question or request a visualization (e.g., 'show the relationship between X and Y')"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          disabled={!jobId || isAnalyzing}
          onKeyPress={(e) => e.key === 'Enter' && !isAnalyzing && jobId && prompt && handlePrompt()}
        />
        <button
          onClick={handlePrompt}
          disabled={!prompt || !jobId || isAnalyzing}
          className="btn-primary"
        >
          {isAnalyzing ? '⏳ Analyzing...' : '🚀 Analyze'}
        </button>
      </div>

      {llmResponseDataAnalysis && (
        <div className="analysis-results">
          <h3>Analysis Result</h3>
          <div className="llm-response">
            <pre>{llmResponseDataAnalysis}</pre>
          </div>
        </div>
      )}

      {plots && (
        <div className="analysis-results">
          <h3>Generated Visualization</h3>
          <img
            src={plots}
            alt="Data Visualization"
            style={{ maxWidth: '100%', height: 'auto', borderRadius: '12px' }}
          />
        </div>
      )}
    </div>
  );
};

export default AnalysisSection;
