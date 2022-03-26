import "./navbar.css"
import { Link, NavLink } from 'react-router-dom'
import Navbar from 'react-bootstrap/Navbar';
import Container from 'react-bootstrap/Container';
import Nav from 'react-bootstrap/Nav';

function NavBarComp() {
    return (
        <Navbar bg="dark" variant="dark" sticky='top'>
            <Container>
                <Navbar.Brand as={NavLink} to='/' exact>HT Solutions</Navbar.Brand>
                <Nav className="me-auto">
                    <Nav.Link as={NavLink} to='/' exact>Home</Nav.Link>
                    <Nav.Link as={NavLink} to='/about'>About</Nav.Link>
                </Nav>
            </Container>
        </Navbar>
    )
}

export default NavBarComp