import { useState } from 'react';
import './Gallery.css';

function Gallery() {
  const [referenceImages, setReferenceImages] = useState([]);
  const [generatedImage, setGeneratedImage] = useState(null);
  const [evaluationResult, setEvaluationResult] = useState(null);

  return (
    <div className="gallery-container">
      <section className="reference-images">
        <h2>Reference Images</h2>
        <div className="image-grid">
          {referenceImages.map((image, index) => (
            <div key={index} className="image-card">
              <img src={image.url} alt={`Reference ${index + 1}`} />
            </div>
          ))}
        </div>
      </section>

      <section className="generated-image">
        <h2>Generated Image</h2>
        {generatedImage && (
          <div className="image-card">
            <img src={generatedImage.url} alt="Generated" />
          </div>
        )}
      </section>

      <section className="evaluation-results">
        <h2>Evaluation Results</h2>
        {evaluationResult && (
          <div className="results-container">
            <p>{evaluationResult}</p>
          </div>
        )}
      </section>
    </div>
  );
}

export default Gallery;