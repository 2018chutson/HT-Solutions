import "./about.css";

export default function About() {
    return (
        <div className="about">
            <div className="title">
                <h1>About</h1>
            </div>
            <p className="description">HT Map is an application that aims to provide a clear and easy to use visual interpretation of 
            human trafficking incident data from Wyoming, and provide predictions for future incidents using our machine learning 
            algorithm, HT Palantir. This app is intended for use by law enforcement officials. All data collected for the purposes of 
            this project is publicly released on the Federal Bureau of Investigation’s Crime Data Explorer, but our database could be 
            replaced with confidential data specific to the organization using HT Map.

            Upon starting, HT Map shows all the collected data on the heatmap. To get a narrowed view of specific filters, use the toggles 
            on the left side of the screen and the map will adjust accordingly.
            </p>
        </div>
    )
}
