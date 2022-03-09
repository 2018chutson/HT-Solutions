import "./map.css";
import React, { useRef, useEffect, useState } from 'react';
import mapboxgl from '!mapbox-gl'; // eslint-disable-line import/no-webpack-loader-syntax
//import data from './halfCrime.csv'

mapboxgl.accessToken = 'pk.eyJ1IjoiY2h1dHNvbjEiLCJhIjoiY2wwazV5M2s3MDFxZDNqcm1xeWh5MDVhZiJ9.g1UbqY2oSaOboxZ0nzmSEg';

function Map() {
    // experimental
    /* const options = [
        {
            name: 'Crime by County',
            property: 'County'
        }
    ] */
    const mapContainer = useRef(null);
    //const [active, setActive] = useState(options[0]); // experimental
    //const [map, setMap] = useRef(null); // experimental
    const map = useRef(null);
    const [lng, setLng] = useState(-107.29);
    const [lat, setLat] = useState(43.07);
    const [zoom, setZoom] = useState(6);

    useEffect(() => {
        if (map.current) return; // initialize map only once
        map.current = new mapboxgl.Map({
            container: mapContainer.current,
            style: 'mapbox://styles/chutson1/cl0hafc5z001614ozarpqc1zq',
            center: [lng, lat],
            zoom: zoom
        });
    });
        /* EXPERIMENTAL STUFF HERE ----------------------------------------------------- */
        /* map.on('load', () => {
            map.addSource('County', {
              type: 'csv',
              data
            });
      
            map.setLayoutProperty('country-label', 'text-field', [
              'format',
              ['get', 'name_en'],
              { 'font-scale': 1.2 },
              '\n',
              {},
              ['get', 'name'],
              {
                'font-scale': 0.8,
                'text-font': [
                  'literal',
                  ['DIN Offc Pro Italic', 'Arial Unicode MS Regular']
                ]
              }
            ]);
      
            map.addLayer(
              {
                id: 'County',
                type: 'fill',
                source: 'countries'
              },
              'country-label'
            );
      
            map.setPaintProperty('countries', 'fill-color', {
              property: active.property,
              stops: active.stops
            });
      
            setMap(map);
          });
      
          // Clean up on unmount
          return () => map.remove();
        }, []);
      
        useEffect(() => {
          paint();
        }, [active]);
      
        const paint = () => {
          if (map) {
            map.setPaintProperty('countries', 'fill-color', {
              property: active.property,
              stops: active.stops
            });
          }
        };
      
        const changeState = i => {
          setActive(options[i]);
          map.setPaintProperty('countries', 'fill-color', {
            property: active.property,
            stops: active.stops
          });
        }; */
        /* EXPERIMENTAL STUFF HERE ----------------------------------------------------- */
    return (
        <div>
          <div ref={mapContainer} className="map-container" />
        </div>
      );
}
export default Map