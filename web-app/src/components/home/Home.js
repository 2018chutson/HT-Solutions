
import "./home.css"
import MapP from "../mapP/MapP"
import MapNP from "../mapNP/MapNP"
import Settings from "../settings/Settings"
import React , { useState, useCallback } from 'react';

export default function Home() {
  const [show, setShow ] = useState(true);
  const handleToggle = useCallback(() => setShow(prevShow => !prevShow),[])

  if(show) {
    return (
      <div className="home">
        <Settings onToggle={handleToggle} txt="PREDICTED"/>
        <MapP />
      </div>
    );
  }
  return (
    <div className="home">
        <Settings onToggle={handleToggle} txt="ACTUAL"/>
        <MapNP />
    </div>
  );
}
