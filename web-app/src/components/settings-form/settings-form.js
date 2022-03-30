import "./settings-form.css";
import Form from 'react-bootstrap/Form'
import Button from 'react-bootstrap/Button'
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
        // <form onSubmit={this.handleSubmit}>
        //   <label>
        //     Crime:
        //     <select value={this.state.value} onChange={this.handleChange}>
        //       <option value="grapefruit">Assault</option>
        //       <option value="lime">Sexual Assault</option>
        //       <option value="coconut">Violation of Protection Order</option>
        //       <option value="mango">Intimidation</option>
        //       <option value="mango">Abduction</option>
        //     </select>
        //   </label>
        //   <input type="submit" value="Submit" />
        // </form>
        <Form>
          <Form.Group className="mb-3" controlId="formCrime">
          <Form.Label>Adjust Data</Form.Label>
            <Form.Select aria-label="Default select example">
              <option>Select crime...</option>
              <option value="1">Assault</option>
              <option value="2">Violation of Protection Order</option>
              <option value="3">Intimidation</option>
              <option value="4">Abduction</option>
            </Form.Select>
          </Form.Group>
          <Form.Group>
            <Form.Label>Adjust Timeline</Form.Label>
            <Form.Range />
          </Form.Group>
          <Button variant="primary" type="submit">
            Submit
          </Button>
        </Form>
      );
    }
  }

export default SettingsForm