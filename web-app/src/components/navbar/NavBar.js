import "./navbar.css"

function NavBar() {
    return (
        <div className="top">
            <div className="topLeft">
                <ul className="topList">
                    <li className="topListItem">HOME</li>
                    <li className="topListItem">ABOUT</li>
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