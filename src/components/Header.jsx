import { Link } from 'react-router-dom'
import './Header.css'

function Header() {
  return (
    <header className="header">
      <div className="container header-container">
        <h1 className="logo">Photo Evaluator</h1>
        <nav className="nav">
          <ul className="nav-list">
            <li className="nav-item">
              <Link to="/" className="nav-link">Home</Link>
            </li>
            <li className="nav-item">
              <Link to="/gallery" className="nav-link">Gallery</Link>
            </li>
          </ul>
        </nav>
      </div>
    </header>
  )
}

export default Header