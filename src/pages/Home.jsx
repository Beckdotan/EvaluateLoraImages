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
  const [isConverting, setIsConverting] = useState(false)
  const navigate = useNavigate()

  // Helper function to check if a file is HEIC
  const isHeicFile = (file) => {
    return file.type === 'image/heic' || 
           file.type === 'image/heif' || 
           file.name.toLowerCase().endsWith('.heic') || 
           file.name.toLowerCase().endsWith('.heif');
  }

  // Helper function to convert HEIC to JPEG
  const convertHeicToJpeg = async (file) => {
    try {
      const jpegBlob = await heic2any({
        blob: file,
        toType: 'image/jpeg',
        quality: 0.8
      });
      
      // If the result is an array (multiple images), take the first one
      const blob = Array.isArray(jpegBlob) ? jpegBlob[0] : jpegBlob;
      
      return new File([blob], file.name.replace(/\.(heic|heif)$/i, '.jpg'), {
        type: 'image/jpeg',
        lastModified: Date.now()
      });
    } catch (error) {
      console.error('HEIC conversion error:', error);
      throw new Error(`Failed to convert HEIC image: ${error.message}. Please try converting it to JPEG manually.`);
    }
  }

  const handleReferenceImageDrop = async (acceptedFiles) => {
    setError('');
    setIsConverting(true);
    
    try {
      // Process files in parallel
      const newImages = await Promise.all(acceptedFiles.map(async (file) => {
        try {
          let processedFile = file;
          
          // Convert HEIC files to JPEG
          if (isHeicFile(file)) {
            processedFile = await convertHeicToJpeg(file);
          }
          
          // Create preview URL
          return Object.assign(processedFile, {
            preview: URL.createObjectURL(processedFile)
          });
        } catch (error) {
          console.error(`Error processing file ${file.name}:`, error);
          setError(error.message);
          return null;
        }
      }));
      
      // Filter out any null values from failed conversions
      const validImages = newImages.filter(img => img !== null);
      setReferenceImages([...referenceImages, ...validImages]);
    } catch (error) {
      console.error('Error processing dropped files:', error);
      setError('Error processing images. Please try again with different files.');
    } finally {
      setIsConverting(false);
    }
  }

  const handleGeneratedImageDrop = async (acceptedFiles) => {
    setError('');
    
    if (acceptedFiles.length > 0) {
      setIsConverting(true);
      
      try {
        let file = acceptedFiles[0];
        
        // Convert HEIC file to JPEG if necessary
        if (isHeicFile(file)) {
          file = await convertHeicToJpeg(file);
        }
        
        setGeneratedImage(Object.assign(file, {
          preview: URL.createObjectURL(file)
        }));
      } catch (error) {
        console.error('Error processing generated image:', error);
        setError(error.message);
      } finally {
        setIsConverting(false);
      }
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
      // Create FormData and append all images
      const formData = new FormData()
      
      // Append all reference images
      referenceImages.forEach((img) => {
        formData.append('files', img)
      })
      
      // Append generated image (should be the last file)
      formData.append('files', generatedImage)
      
      console.log('Uploading images...');
      
      // Make API call to server
      const response = await fetch('http://localhost:8000/detect', {
        method: 'POST',
        body: formData
      })
      
      const data = await response.json()
      
      if (!response.ok) {
        throw new Error(data.detail || 'Server returned an error')
      }
      
      if (data.error) {
        throw new Error(data.error)
      }

      console.log('Received data:', data);
      
      // Navigate to gallery page with results
      navigate('/gallery', { 
        state: { 
          referenceImages: referenceImages.map(img => ({ url: img.preview })),
          generatedImage: { url: generatedImage.preview },
          evaluationResult: data
        } 
      })
    } catch (err) {
      console.error('Upload error:', err)
      setError(`Failed to upload images: ${err.message}`)
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
                'image/*': ['.jpeg', '.jpg', '.png', '.heic', '.heif']
              }}
              maxFiles={15}
              disabled={referenceImages.length >= 15 || isUploading || isConverting}
            >
              {({getRootProps, getInputProps}) => (
                <div 
                  {...getRootProps()} 
                  className={`dropzone ${referenceImages.length >= 15 || isConverting ? 'dropzone-disabled' : ''}`}
                >
                  <input {...getInputProps()} />
                  {isConverting ? (
                    <p>Converting HEIC images, please wait...</p>
                  ) : (
                    <p>Drag & drop reference images here, or click to select files</p>
                  )}
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
                      disabled={isUploading || isConverting}
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
                'image/*': ['.jpeg', '.jpg', '.png', '.heic', '.heif']
              }}
              maxFiles={1}
              disabled={isUploading || isConverting}
            >
              {({getRootProps, getInputProps}) => (
                <div {...getRootProps()} className="dropzone">
                  <input {...getInputProps()} />
                  {isConverting ? (
                    <p>Converting HEIC image, please wait...</p>
                  ) : (
                    <p>Drag & drop generated image here, or click to select file</p>
                  )}
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
                    disabled={isUploading || isConverting}
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
              disabled={isUploading || isConverting || referenceImages.length === 0 || !generatedImage}
            >
              {isUploading ? 'Uploading...' : isConverting ? 'Converting...' : 'Evaluate Images'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default Home