import React from "react"
import "./home.css"
import Map from "../map/Map"
import Sidebar from "../sidebar/Sidebar"

class Home extends React.Component {
  render() {
    return (
      <div className="home">
          <Sidebar />
          <Map />
      </div>
    )
  }
}
export default Home