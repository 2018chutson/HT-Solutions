import "./map.css";
import React, { useRef, useEffect } from 'react';
import mapboxgl from 'mapbox-gl';
import actualData from "./currentpredictions.geojson"

mapboxgl.accessToken = 'pk.eyJ1IjoiY2h1dHNvbjEiLCJhIjoiY2wwazV5M2s3MDFxZDNqcm1xeWh5MDVhZiJ9.g1UbqY2oSaOboxZ0nzmSEg';

const MapNP = () => {
  const mapContainerRef = useRef(null);

  useEffect(() => {
    const map = new mapboxgl.Map({
      container: mapContainerRef.current,
      style: "mapbox://styles/mapbox/light-v10",
      center: [-107.29, 43.07],
      zoom: 6
    });

    map.addControl(new mapboxgl.NavigationControl(), "top-right");

    map.on("load", () => {
      map.addSource("actual-data", {
        type: "geojson",
        data: actualData
      });

      map.addLayer(
        {
          id: 'predicted-heat',
          type: 'heatmap',
          source: 'actual-data',
          maxzoom: 15,
          paint: {
            'heatmap-weight': {
              property: 'probability of event',
              type: 'exponential',
              stops: [
                [0, 0],
                [.2, 10],
                [.3, 20],
                [.4, 30],
                [.5, 40],
                [.6, 50],
                [.7, 60],
                [.8, 70],
                [.9, 80],
                [1, 90]
              ]
            },
            'heatmap-intensity': {
              stops: [
                [0, 1],
                [5, 3]
              ]
            },
            'heatmap-color': [
              'interpolate',
              ['linear'],
              ['heatmap-density'],
              0,
              'rgba(236,222,239,0)',
              0.2,
              'rgb(208,209,230)',
              0.4,
              'rgb(166,189,219)',
              0.6,
              'rgb(103,169,207)',
              0.8,
              'rgb(133, 175, 255)'
            ],
            'heatmap-radius': {
              stops: [
                [0, 2],
                [.1, 4],
                [.2, 6],
                [.3, 7],
                [.4, 10],
                [.5, 11],
                [.6, 13],
                [.7, 15],
                [.8, 17],
                [.9, 19],
                [1, 20]
              ]
            },
            'heatmap-opacity': {
              default: 1,
              stops: [
                [0, 0],
                [1, 1]
              ]
            }
          }
        },
        'waterway-label'
      );
    });

    return () => map.remove();
  }, []);

  return (
    <div className="map-container" ref={mapContainerRef} />
  );
};

export default MapNP;
