import Container from './components/container/Container';
import './App.css';
import React from 'react';
import NavBar from './components/navbar/NavBar';
import Footer from './components/footer/Footer';

function App() {
  return (
    <React.StrictMode>
      <NavBar />
      <Container />
      <Footer />
    </React.StrictMode>
  );
}

export default App;
