import "./navbar.css"
import { Link } from "react-router-dom";

function NavBar() {
    return (
        <div className="top">
            <div className="topLeft">
                <ul className="topList">
                    <li className="topListItem">
                        <Link className="link" to="/">HOME</Link>
                    </li>
                    <li className="topListItem">
                        <Link className="link" to="/about">ABOUT</Link>
                    </li>
                </ul>
            </div>
            <div className="topCenter">
                <header>
                    <h3 className="header">HT Solutions</h3>
                </header>
            </div>
            <div className="topRight"></div>
        </div>
    )
}

export default NavBar