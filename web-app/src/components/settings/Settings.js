import "./settings.css";
import SettingsForm from "../settings-form/settings-form.js"
import React from "react";

function Settings( { onToggle, txt }) {

    return (
        <div className="sidebar">
            <div className="settings">
                <h4>HT OUTPUT DATA</h4>
                <p>Click button to switch output data</p>
                <div>
                    <button className="dataBtn" onClick={onToggle}>{txt}</button>
                </div>
                <br />
                <h4>SETTINGS</h4>
                <SettingsForm />
            </div>
        </div>
    )
}
export default Settings