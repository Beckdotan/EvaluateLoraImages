import { useState } from 'react'
import { Routes, Route } from 'react-router-dom'
import './App.css'
import Header from './components/Header'
import Home from './pages/Home'
import Gallery from './pages/Gallery'
import Analysis from './pages/Analysis'

function App() {
  return (
    <div className="app-container">
      <Header />
      <main className="main-content">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/gallery" element={<Gallery />} />
          <Route path="/analysis" element={<Analysis />} />
        </Routes>
      </main>
    </div>
  )
}

export default App