import "./map.css";
import React, { useRef, useEffect, useState } from 'react';
import mapboxgl from '!mapbox-gl'; // eslint-disable-line import/no-webpack-loader-syntax

mapboxgl.accessToken = 'pk.eyJ1IjoiY2h1dHNvbjEiLCJhIjoiY2wwaDZub2c2MDVwMzNkcmlzY2lveWxqdSJ9.BgzShpN9y_K-QLFbRE-NNw';

function Map() {
    const mapContainer = useRef(null);
    const map = useRef(null);
    const [lng, setLng] = useState(-107.29);
    const [lat, setLat] = useState(43.07);
    const [zoom, setZoom] = useState(6);

    useEffect(() => {
        if (map.current) return; // initialize map only once
        map.current = new mapboxgl.Map({
            container: mapContainer.current,
            style: 'mapbox://styles/mapbox/streets-v11',
            center: [lng, lat],
            zoom: zoom
        });
    });
    return (
        <div>
          <div ref={mapContainer} className="map-container" />
        </div>
      );
}
export default Map