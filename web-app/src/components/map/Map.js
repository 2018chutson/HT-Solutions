import "./map.css";
import React, { useRef, useEffect, useState } from 'react';
import mapboxgl from '!mapbox-gl'; // eslint-disable-line import/no-webpack-loader-syntax
//import Settings from "../settings/Settings.js"
//import data from './halfCrime.csv'

mapboxgl.accessToken = 'pk.eyJ1IjoiY2h1dHNvbjEiLCJhIjoiY2wwazV5M2s3MDFxZDNqcm1xeWh5MDVhZiJ9.g1UbqY2oSaOboxZ0nzmSEg';

function Map() {
    const mapContainer = useRef(null);
    const map = useRef(null);
    const [lng, setLng] = useState(-107.29);
    const [lat, setLat] = useState(43.07);
    const [zoom, setZoom] = useState(6);

    // creates the map for the first time
    useEffect(() => {
        if (map.current) return;
        map.current = new mapboxgl.Map({
            container: mapContainer.current,
            style: 'mapbox://styles/chutson1/cl0hafc5z001614ozarpqc1zq',
            center: [lng, lat],
            zoom: zoom
        });
    });

    // when the map loads, add the prediciton source, then add the visual layer of the prediction data
    /* useEffect(() => {
      if (!map.current) return;
        map.current.on('load', () => {
          map.current.addSource('prediction', {
            type: 'geojson',
            data: 'prediction.geojson'
          });
          map.current.addLayer({
          'id': 'prediction-layer',
          'type': 'circle',
          'source': 'prediction',
          'paint': {
            'circle-radius': 8,
            'circle-stroke-width': 2,
            'circle-color': 'blue',
            'circle-stroke-color': 'white'
          }
        });
      });
    }); 
     */
    // create togglable layer! Taken from mapbox GL JS documentation: https://docs.mapbox.com/mapbox-gl-js/example/toggle-layers/
    useEffect (() => {
      if (!map.current) return;
      map.current.on('load', () => {
        map.current.addSource('prediction', {
          type: 'geojson',
          data: 'prediction.geojson'
        });
        map.current.addLayer({
        'id': 'prediction-layer',
        'type': 'circle',
        'source': 'prediction',
        'paint': {
          'circle-radius': 8,
          'circle-stroke-width': 2,
          'circle-color': 'blue',
          'circle-stroke-color': 'white'
        }
      });
      // If the prediciton layer was not added to the map, abort
        // if (!map.current.getLayer('prediction-layer')) {
        //   return;
        // }
      // Enumerate the layer id.
      const toggleableLayerId = ['prediciton-layer'];
       
      // Set up the toggle button.
      for (const id of toggleableLayerId) {
        if (document.getElementById(id)) {
          continue;
        }
      
      // Create a link.
      const link = document.createElement('a');
      link.id = id;
      link.href = '#';
      link.textContent = id;
      link.className = 'active';
       
      // Show or hide layer when the toggle is clicked.
      link.onclick = function (e) {
        const clickedLayer = this.textContent;
        e.preventDefault();
        e.stopPropagation();
       
        const visibility = map.current.getLayoutProperty(
        clickedLayer,
        'visibility'
      );
       
      // Toggle layer visibility by changing the layout object's visibility property.
      if (visibility === 'visible') {
        map.current.setLayoutProperty(clickedLayer, 'visibility', 'none');
        this.className = '';
      } else {
        this.className = 'active';
        map.setLayoutProperty(
        clickedLayer,
        'visibility',
        'visible'
      );
      }
      };
       
      const layers = document.getElementById('menu');
      layers.appendChild(link);
    }
    });
  });
  
    return (
        <div>
          <nav id="menu"></nav>
          <div ref={mapContainer} className="map-container" />
        </div>
      );
}
export default Map