import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import Dropzone from 'react-dropzone'
import heic2any from 'heic2any'
import './Home.css'

function Home() {
  const [referenceImages, setReferenceImages] = useState([])
  const [generatedImage, setGeneratedImage] = useState(null)
  const [isUploading, setIsUploading] = useState(false)
  const [error, setError] = useState('')
  const navigate = useNavigate()

  const handleReferenceImageDrop = async (acceptedFiles) => {
    setError('')
    // Create preview URLs for the dropped files
    const newImages = await Promise.all(acceptedFiles.map(async (file) => {
  let convertedFile = file;
  
  if (file.type === 'image/heic' || file.name.toLowerCase().endsWith('.heic')) {
    const jpegBlob = await heic2any({
      blob: file,
      toType: 'image/jpeg',
      quality: 0.8
    });
    
    convertedFile = new File([jpegBlob], file.name.replace(/\.heic$/i, '.jpg'), {
      type: 'image/jpeg',
      lastModified: Date.now()
    });
  }

  return Object.assign(convertedFile, {
    preview: URL.createObjectURL(convertedFile)
  });
}))
    setReferenceImages([...referenceImages, ...newImages])
  }

  const handleGeneratedImageDrop = async (acceptedFiles) => {
    setError('')
    if (acceptedFiles.length > 0) {
      let file = acceptedFiles[0];
      
      if (file.type === 'image/heic' || file.name.toLowerCase().endsWith('.heic')) {
        const jpegBlob = await heic2any({
          blob: file,
          toType: 'image/jpeg',
          quality: 0.8
        });
        
        file = new File([jpegBlob], file.name.replace(/\.heic$/i, '.jpg'), {
          type: 'image/jpeg',
          lastModified: Date.now()
        });
      }

      setGeneratedImage(Object.assign(file, {
        preview: URL.createObjectURL(file)
      }))
    }
  }

  const removeReferenceImage = (index) => {
    const updatedImages = [...referenceImages]
    // Revoke the object URL to avoid memory leaks
    URL.revokeObjectURL(updatedImages[index].preview)
    updatedImages.splice(index, 1)
    setReferenceImages(updatedImages)
  }

  const removeGeneratedImage = () => {
    if (generatedImage) {
      URL.revokeObjectURL(generatedImage.preview)
      setGeneratedImage(null)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    // Validate inputs
    if (referenceImages.length === 0) {
      setError('Please upload at least one reference image')
      return
    }
    
    if (!generatedImage) {
      setError('Please upload a generated image')
      return
    }
    
    setIsUploading(true)
    setError('')
    
    try {
      // In a real implementation, we would upload the images to the server here
      // For now, we'll just simulate a successful upload and navigate to the gallery
      
      // Simulating API call delay
      await new Promise(resolve => setTimeout(resolve, 1500))
      
      // Navigate to gallery page after successful upload
      navigate('/gallery')
    } catch (err) {
      console.error('Upload error:', err)
      setError('Failed to upload images. Please try again.')
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <div className="home-page">
      <div className="container">
        <h2 className="page-title">Upload Images for Evaluation</h2>
        <p className="page-description">
          Upload reference photos of a real person and a generated image to evaluate how closely they match.
        </p>
        
        {error && <div className="error-message">{error}</div>}
        
        <form onSubmit={handleSubmit} className="upload-form">
          <div className="upload-section">
            <h3>Reference Images</h3>
            <p className="upload-hint">Upload 1-15 photos of the real person</p>
            
            <Dropzone 
              onDrop={handleReferenceImageDrop}
              accept={{
                'image/*': ['.jpeg', '.jpg', '.png', '.heic'],
                'image/heic': ['.heic']
              }}
              maxFiles={15}
              disabled={referenceImages.length >= 15 || isUploading}
            >
              {({getRootProps, getInputProps}) => (
                <div 
                  {...getRootProps()} 
                  className={`dropzone ${referenceImages.length >= 5 ? 'dropzone-disabled' : ''}`}
                >
                  <input {...getInputProps()} />
                  <p>Drag & drop reference images here, or click to select files</p>
                </div>
              )}
            </Dropzone>
            
            {referenceImages.length > 0 && (
              <div className="preview-grid">
                {referenceImages.map((file, index) => (
                  <div key={index} className="preview-item">
                    <img src={file.preview} alt={`Reference ${index + 1}`} className="preview-image" />
                    <button 
                      type="button" 
                      className="remove-btn"
                      onClick={() => removeReferenceImage(index)}
                      disabled={isUploading}
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
          
          <div className="upload-section">
            <h3>Generated Image</h3>
            <p className="upload-hint">Upload the AI-generated image to evaluate</p>
            
            <Dropzone 
              onDrop={handleGeneratedImageDrop}
              accept={{
                'image/*': ['.jpeg', '.jpg', '.png', '.heic'],
                'image/heic': ['.heic']
              }}
              maxFiles={1}
              disabled={isUploading}
            >
              {({getRootProps, getInputProps}) => (
                <div {...getRootProps()} className="dropzone">
                  <input {...getInputProps()} />
                  <p>Drag & drop generated image here, or click to select file</p>
                </div>
              )}
            </Dropzone>
            
            {generatedImage && (
              <div className="preview-grid">
                <div className="preview-item">
                  <img src={generatedImage.preview} alt="Generated" className="preview-image" />
                  <button 
                    type="button" 
                    className="remove-btn"
                    onClick={removeGeneratedImage}
                    disabled={isUploading}
                  >
                    ×
                  </button>
                </div>
              </div>
            )}
          </div>
          
          <div className="form-actions">
            <button 
              type="submit" 
              className="btn submit-btn"
              disabled={isUploading || referenceImages.length === 0 || !generatedImage}
            >
              {isUploading ? 'Uploading...' : 'Evaluate Images'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default Home