# Proyect Layout

## Network Reduction
The workflow allows for the generation of reduced network datasets according to different graph reduction techniques (GRTs).

We are interested in assesing the quality of a network reduction based on its properties and suitability for diffusion-based models. In addition to calculating the relevant network metrics of the newly obtain dataset, our workflow also simulates the SIR model on it and calculates a set of commonly used quantities of interested that can be used to asses the quality of the reduced dataset.

```mermaid
---
title: Network Reduction Process
---

flowchart TB
    config[Configuration File]-- "Network Dataset" ---> reduce((Reduce Network))
    config-- "Reduction Parameters" ---> reduce
    

    reduce-- "Network Reduction" ---> distributions((Calculate Distributions))
    distributions-- "Degree Distribution" ---> characterization((Calculate Network Metrics))
    distributions-- "Distance Distributions" ---> characterization

    reduce-- "Network Reduction" ---> simulate((Simulate Model))
    config-- "Simulation Parameters" ---> simulate
    simulate-- "Simulation Results" ---> qois((Calculate Model's Quantities of Interest))

    characterization-- "Network Metrics" ---> storage@{shape: cyl, label: "Reduction Directory"}
    qois-- "QOIs" ---> storage

    subgraph reduction [Reduction]
        reduce
        distributions
        characterization
    end

    subgraph simulation [Simulation]
        simulate
        qois
    end
```

