import './App.css';
import React from 'react';
import NavBar from './components/navbar/NavBar';
import Footer from './components/footer/Footer';
import Home from './components/home/Home';
import About from './components/about/About';
import { Routes, Route } from "react-router-dom";

function App() {
  return (
    <div className="App">
      <NavBar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="about" element={<About />} />
      </Routes>
      <Footer />
    </div>
  );
}

export default App;
