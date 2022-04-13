import "./map.css";
import React, { useRef, useEffect } from 'react';
import mapboxgl from 'mapbox-gl';
import predictionData from "./prediction.geojson"

mapboxgl.accessToken = 'pk.eyJ1IjoiY2h1dHNvbjEiLCJhIjoiY2wwazV5M2s3MDFxZDNqcm1xeWh5MDVhZiJ9.g1UbqY2oSaOboxZ0nzmSEg';

const Map = () => {
  const mapContainerRef = useRef(null);

  // initialize map when component mounts
  useEffect(() => {
    const map = new mapboxgl.Map({
      container: mapContainerRef.current,
      // See style options here: https://docs.mapbox.com/api/maps/#styles
      style: "mapbox://styles/mapbox/light-v10",
      center: [-107.29, 43.07],
      zoom: 6
    });

    // add navigation control (the +/- zoom buttons)
    map.addControl(new mapboxgl.NavigationControl(), "top-right");

    map.on("load", () => {
      // add the data source for new a feature collection with no features
      map.addSource("predicted-data", {
        type: "geojson",
        data: predictionData
      });

      map.addLayer(
        {
          id: 'predicted-heat',
          type: 'heatmap',
          source: 'predicted-data',
          maxzoom: 15,
          paint: {
            // increase weight as diameter breast height increases
            'heatmap-weight': {
              property: 'predict',
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
            // increase intensity as zoom level increases
            'heatmap-intensity': {
              stops: [
                [0, 1],
                [20, 3]
              ]
            },
            // assign color values be applied to points depending on their density
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
              'rgb(68, 192, 201)'
            ],
            // increase radius as zoom increases
            'heatmap-radius': {
              stops: [
                [0, 5],
                [.1, 6],
                [.2, 8],
                [.3, 10],
                [.4, 15],
                [.5, 20],
                [.6, 25],
                [.7, 30],
                [.8, 30],
                [.9, 35],
                [1, 36]
              ]
            },
            // decrease opacity to transition into the circle layer
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

    // clean up on unmount
    return () => map.remove();
  }, []);

  return <div className="map-container" ref={mapContainerRef} />;
};

export default Map;
