import "./about.css";
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';

export default function About() {
    return (
        <Container style={{paddingTop:"50px"}}>
            <Row>
                <h3 style={{textAlign: "center", margin: "20px auto auto"}}>ABOUT</h3>
            </Row>
            <Row>
                <Col md={{ span: 6, offset: 3}} style={{textAlign: "center"}}>
                    <p>
                            HT Map is an application that aims to provide a clear and easy to use visual interpretation of 
                        human trafficking incident data from Wyoming, and provide predictions for future incidents using our machine learning 
                        algorithm, HT Palantir. This app is intended for use by law enforcement officials. All data collected for the purposes of 
                        this project is publicly released on the Federal Bureau of Investigation’s Crime Data Explorer, but our database could be 
                        replaced with confidential data specific to the organization using HT Map.
                    </p>
                    <p>
                            Upon starting, HT Map shows all the collected data on the heatmap. To get a narrowed view of specific filters, use the toggles 
                        on the left side of the screen and the map will adjust accordingly.
                    </p>
                </Col>
            </Row>
        </Container>
    )
}
