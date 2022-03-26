import './App.css';
import React from 'react';
import NavBarComp from './components/navbarcomp/NavBarComp';
import Home from './components/home/Home';
import About from './components/about/About';
import { Routes, Route } from "react-router-dom";


function App() {
  return (
    <div className="App">
      <NavBarComp />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="about" element={<About />} />
      </Routes>
    </div>
  );
}

export default App;

