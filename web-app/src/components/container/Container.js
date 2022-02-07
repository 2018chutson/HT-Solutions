import React from "react"
import "./container.css"
import Map from "../map/Map"
import Sidebar from "../sidebar/Sidebar"

class Container extends React.Component {
  render() {
    return (
      <div className="container">
          <Sidebar />
          <Map />
      </div>
    )
  }
}
export default Container