import Summary from "../summary/Summary";
import "./sidebar.css";
import Settings from "../settings/Settings";

function Sidebar() {
    return (
        <div className="sidebar">
            <Settings />
            <Summary />
        </div>
    )
}
export default Sidebar