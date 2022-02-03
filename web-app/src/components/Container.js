import React from "react"
import NavBar from "./navbar/NavBar"

class Container extends React.Component {
  render() {
    return (
      <div>
          <NavBar />
          <p>I am in a React Component!</p>
      </div>
    )
  }
}
export default Container