import "./settings-form.css";
import React from "react";

class SettingsForm extends React.Component {
    constructor(props) {
      super(props);
      this.state = {value: 'coconut'};
  
      this.handleChange = this.handleChange.bind(this);
      this.handleSubmit = this.handleSubmit.bind(this);
    }
  
    handleChange(event) {
      this.setState({value: event.target.value});
    }
  
    handleSubmit(event) {
      //alert('Your favorite flavor is: ' + this.state.value);
      event.preventDefault();
    }
  
    render() {
      return (
        <form onSubmit={this.handleSubmit}>
          <label>
            Crime:
            <select value={this.state.value} onChange={this.handleChange}>
              <option value="grapefruit">Assault</option>
              <option value="lime">Sexual Assault</option>
              <option value="coconut">Violation of Protection Order</option>
              <option value="mango">Intimidation</option>
              <option value="mango">Abduction</option>
            </select>
          </label>
          <input type="submit" value="Submit" />
        </form>
      );
    }
  }

export default SettingsForm