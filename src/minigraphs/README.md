# First Order Analysis of Models

```mermaid
---
title: Workflow
---
flowchart TD
    %% Nodes
    entity:parameters@{shape: rect, label: "Parameter file"}
    process:sample((Sample Graphs))
    process:simulate((Simulate Model))
    process:qois((Generate Calculate quantitites of Interest))
    process:plot((Generate Plots))
    ds:plots_dir@{shape: round, label: "Plots directory"}

    %% Wiring
    entity:parameters-- "Graph Characteristics" ---process:sample
    entity:parameters-- "Simulation properties" ---process:simulate
    entity:parameters-- "QOIs to calculate" ---process:qois
    entity:parameters-- "QOIs to plot" ---process:plot

    process:sample -- "Graph .npz files" ---process:simulate
    
    process:simulate -- "Simulation trajectories" ---process:qois

    process:qois -- "QOIs for each sample" ---process:plot

    process:plot -- "QOIs plots" ---ds:plots_dir
```