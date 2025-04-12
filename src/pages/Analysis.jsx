import { useState, useEffect } from 'react';
import { useLocation, Link } from 'react-router-dom';
import './Analysis.css';

function Analysis() {
  const location = useLocation();
  const { referenceImages, generatedImage, results } = location.state || {};
  
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [analysisResults, setAnalysisResults] = useState(null);
  
  // Basic validation
  if (!referenceImages || !generatedImage || !results) {
    return (
      <div className="error-container">
        <h2>No images to analyze</h2>
        <p>Please upload images on the home page first.</p>
        <Link to="/" className="btn">Go to Upload Page</Link>
      </div>
    );
  }

  // Separate results into reference and generated
  const referenceResults = results.filter(result => result.type === 'reference');
  const generatedResult = results.find(result => result.type === 'generated');
  
  useEffect(() => {
    // Only run analysis if we have the necessary data
    if (referenceResults.length > 0 && generatedResult) {
      performAnalysis();
    }
  }, []);
  
  const performAnalysis = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      // Extract the image IDs from the file paths
      const referenceIds = referenceResults.map(result => {
        // Extract filename from the path
        return result.image_path.split('/').pop() || result.image_path.split('\\').pop();
      });
      
      const generatedId = generatedResult.image_path.split('/').pop() || 
                          generatedResult.image_path.split('\\').pop();
      
      // Make API request to the analysis endpoint
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          reference_ids: referenceIds,
          generated_id: generatedId
        })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to analyze images');
      }
      
      const data = await response.json();
      setAnalysisResults(data);
    } catch (err) {
        console.error('Analysis error:', err);
        let errorDetails = err.message || 'An error occurred during analysis';
      
        // Try to get more details from the response if it exists and is not ok
        if (response && !response.ok) {
          try {
            // Attempt to read the response body as text
            const errorText = await response.text();
            console.error('Server response text:', errorText); 
            // Try parsing again IF it looks like JSON, otherwise use text
            try {
                const errorJson = JSON.parse(errorText);
                errorDetails = errorJson.detail || errorJson.message || errorText;
            } catch (parseError) {
                // If parsing fails, use the raw text (might be HTML)
                errorDetails = `Server returned status ${response.status}: ${errorText.substring(0, 200)}...`; 
            }
          } catch (textError) {
            console.error('Could not read error response text:', textError);
            errorDetails = `Failed to analyze images. Server returned status ${response?.status || 'unknown'}. Unable to read error details.`;
          }
        } else if (!response) {
            errorDetails = 'Network error or failed to fetch from the server.';
        }
        
        setError(errorDetails); 
      } finally {
        setIsLoading(false);
      }
  };
  
  return (
    <div className="analysis-container">
      <h1>Image Analysis Results</h1>
      
      <div className="images-summary">
        <div className="reference-summary">
          <h3>Reference Images</h3>
          <div className="thumbnail-grid">
            {referenceResults.map((result, index) => (
              <div key={`ref-${index}`} className="thumbnail">
                <img src={result.thumbnail_base64} alt={`Reference ${index + 1}`} />
              </div>
            ))}
          </div>
        </div>
        
        <div className="generated-summary">
          <h3>Generated Image</h3>
          <div className="thumbnail-grid">
            <div className="thumbnail">
              <img src={generatedResult.thumbnail_base64} alt="Generated" />
            </div>
          </div>
        </div>
      </div>
      
      {isLoading && (
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Analyzing images with ViP-LLaVA...</p>
          <p className="loading-note">This may take a moment as we perform detailed visual analysis</p>
        </div>
      )}
      
      {error && (
        <div className="error-message">
          <h3>Analysis Error</h3>
          <p>{error}</p>
          <button className="btn" onClick={performAnalysis}>Retry Analysis</button>
        </div>
      )}
      
      {analysisResults && (
        <div className="analysis-results">
          <div className="analysis-section">
            <h2>Facial Features Analysis</h2>
            <div className="analysis-content">
              <p>{analysisResults.face_analysis}</p>
            </div>
          </div>
          
          <div className="analysis-section">
            <h2>Body Features Analysis</h2>
            <div className="analysis-content">
              <p>{analysisResults.body_analysis}</p>
            </div>
          </div>
        </div>
      )}
      
      <div className="navigation-buttons">
        <Link to="/gallery" className="btn" state={{ referenceImages, generatedImage, results }}>
          Back to Gallery
        </Link>
        <Link to="/" className="btn">Upload New Images</Link>
      </div>
    </div>
  );
}

export default Analysis;