import "./settings.css";
import SettingsForm from "../settings-form/settings-form.js"
import React from "react";
import ToggleButton from "react-bootstrap/ToggleButton";
import ToggleButtonGroup from "react-bootstrap/ToggleButtonGroup";

function Settings() {
    const [value, setValue] = React.useState(2);
    const handleChange = val => setValue(val);

    return (
        <div className="settings">
            <h4>HT OUTPUT DATA</h4>
            <ToggleButtonGroup
                name="value"
                id="data-setting"
                type="radio"
                value={value}
                onChange={handleChange}
            >
                <ToggleButton value={1}>actual</ToggleButton>
                <ToggleButton value={2}>predicted</ToggleButton>
            </ToggleButtonGroup>
            <br />
            <h4>SETTINGS</h4>
            <SettingsForm />
        </div>
    )
}
export default Settings