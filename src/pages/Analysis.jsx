import { useState, useEffect } from 'react';
import { useLocation, Link } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import './Analysis.css';

// Import an info icon from a common icon library (assuming you have lucide-react installed)
import { Info } from 'lucide-react';

function Analysis() {
  const location = useLocation();
  const { referenceImages, generatedImage, results } = location.state || {};
  
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [activeTooltip, setActiveTooltip] = useState(null);
  
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

  // Function to render quality score with a color indicator
  const renderQualityScore = (score) => {
    let colorClass = 'score-medium';
    if (score >= 0.7) colorClass = 'score-high';
    if (score < 0.4) colorClass = 'score-low';
    
    return (
      <span className={`quality-score ${colorClass}`}>
        {(score * 100).toFixed(1)}%
      </span>
    );
  };
  
  // Tooltip descriptions
  const tooltips = {
    brisque: "Blind/Referenceless Image Spatial Quality Evaluator - Measures how natural an image looks without needing a reference image. Detects blur, compression artifacts, noise, and unnatural patterns. Lower raw scores indicate better quality.",
    niqe: "Natural Image Quality Evaluator - Evaluates how closely an image matches natural image statistics. Great for detecting AI-generated content that lacks natural characteristics. Lower raw scores indicate better quality.",
    totalVariation: "Measures pixel-to-pixel changes across the image. Too low: overly smooth, lacks texture (bad). Too high: excessive noise or artifacts (bad). Middle values are best (natural textures).",
    contentScore: "Uses VGG16 neural network features to detect unnatural content arrangements. Helps identify content that might look technically fine but is semantically abnormal. Higher values indicate more natural content structure.",
    handScore: "Analyzes hand anatomy using MediaPipe. The score starts at perfect 1.0 and decreases based on detected issues: +0.2 penalty for each abnormally long/short finger, +0.15 penalty for unusual finger proportions, +0.3 penalty for merged or extra fingers. No hands detected = perfect score (no penalty)."
  };
  
  // Function to render tooltip
const renderTooltip = (id, content) => {
  return (
    <div className="tooltip-container">
      <button 
        className="info-icon-button" 
        onClick={(e) => {
          e.preventDefault();
          setActiveTooltip(activeTooltip === id ? null : id);
        }}
        aria-label="Show information"
      >
        <Info size={16} />
      </button>
      
      {activeTooltip === id && (
        <div className="tooltip-content">
          <p>{content}</p>
          <button 
            className="tooltip-close" 
            onClick={(e) => {
              e.preventDefault();
              setActiveTooltip(null);
            }}
          >
            Close
          </button>
        </div>
      )}
    </div>
  );
};

// Helper function to get metric weight display
const getMetricWeight = (metricName) => {
  const weights = {
    'brisque': '35%',
    'total_variation': '30%',
    'clip_iqa': '40%'
  };
  
  return weights[metricName] || 'N/A';
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
        <p>Analyzing images with Gemini...</p>
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
        {/* Overall Score Section */}
        <div className="analysis-section overall-score">
          <h2>Overall Score</h2>
          <div className="analysis-content">
            <div className="quality-summary">
              <div className="quality-score-container">
                <h3>Overall Score: {renderQualityScore(analysisResults.overall_score)}</h3>
                <p className="quality-status">
                  Status: <span className={analysisResults.quality_analysis.is_acceptable ? 'acceptable' : 'needs-improvement'}>
                    {analysisResults.quality_analysis.is_acceptable ? 'Acceptable' : 'Needs Improvement'}
                  </span>
                </p>
                <p className="quality-level">
                  Quality Level: <span className={`level-${analysisResults.quality_analysis.quality_level}`}>
                    {analysisResults.quality_analysis.quality_level.charAt(0).toUpperCase() +
                      analysisResults.quality_analysis.quality_level.slice(1)}
                  </span>
                </p>
                <p className="acceptability-threshold">
                  <small>Images with a score of 75% or higher are considered acceptable</small>
                </p>
              </div>

              {analysisResults.quality_analysis.issues?.length > 0 && (
                <div className="quality-issues">
                  <h4>Detected Issues:</h4>
                  <ul>
                    {analysisResults.quality_analysis.issues.map((issue, index) => (
                      <li key={index}>{issue}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>

            <div className="quality-metrics">
              <h4>Contributing Metrics:</h4>
              <div className="metrics-grid">
                {/* CLIP Score */}
                <div className="metric-item">
                  <div className="metric-header">
                    <span className="metric-name">Similarity Score:</span>
                    {renderTooltip('FaceNet', 'FaceNet SCORE: This score reflects how semantically similar the generated image face is to the reference images, using FaceNet visual understanding. Higher scores (0-1) indicate the generated image closely matches the references in terms of content and structure.')}
                  </div>
                  <span className="metric-value">{analysisResults.face_similarity_score.toFixed(3)*100 + "%"}</span>
                  <div className="metric-weight"><small>Weight: 70%</small></div>
                </div>

                {/* BRISQUE Score */}
                <div className="metric-item">
                  <div className="metric-header">
                    <span className="metric-name">Image Quality:</span>
                    {renderTooltip('brisque', 'BRISQUE measures spatial distortion like blur or unnatural patterns. Lower raw scores (0–100) are better.')}
                  </div>
                  <span className="metric-value">
                    {analysisResults.quality_analysis.normalized_metrics?.brisque?.toFixed(3)*100 + "%"}
                  </span>
                  <div className="metric-weight">
                    <small>
                      Weight: {
                        analysisResults.quality_analysis.hand_analysis?.num_hands_detected > 0 ? '15%' : '30%'
                      }
                    </small>
                  </div>
                </div>

                {/* Hand Score – Only if hands exist */}
                {analysisResults.quality_analysis.hand_analysis?.num_hands_detected > 0 && (
                  <div className="metric-item">
                    <div className="metric-header">
                      <span className="metric-name">Hand Score:</span>
                      {renderTooltip('handScore', 'Assesses hand realism. 1.0 is perfect. Deductions are made for abnormalities like merged fingers or unrealistic proportions.')}
                    </div>
                    <span className="metric-value">
                      {analysisResults.quality_analysis.hand_analysis.hand_score.toFixed(3)*100 + "%"}
                    </span>
                    <div className="metric-weight"><small>Weight: 15%</small></div>
                  </div>
                )}
              </div>

             
            </div>
          </div>
        </div>

        {/* Suggestions Section */}
        <div className="analysis-section">
          <h2>Improvement Suggestions</h2>
          <div className="analysis-content markdown-content">
            <ReactMarkdown>{analysisResults.improvement_suggestions}</ReactMarkdown>
          </div>
        </div>

        {/* Face Section */}
        <div className="analysis-section">
          <h2>Facial Features Analysis</h2>
          <div className="analysis-content markdown-content">
            <ReactMarkdown>{analysisResults.face_analysis}</ReactMarkdown>
          </div>
        </div>

        {/* Body Section */}
        <div className="analysis-section">
          <h2>Body Features Analysis</h2>
          <div className="analysis-content markdown-content">
            <ReactMarkdown>{analysisResults.body_analysis}</ReactMarkdown>
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