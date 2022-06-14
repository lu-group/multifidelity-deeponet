# Multifidelity DeepONet

The data and code for the paper [L. Lu, R. Pestourie, S. G. Johnson, & G. Romano. Multifidelity deep neural operators for efficient learning of partial differential equations with application to fast inverse design of nanoscale heat transport. *Physical Review Research*, 4(2), 023210, 2022](https://doi.org/10.1103/PhysRevResearch.4.023210).

## Data

- [Poisson equation](data/poisson)
- [Boltzmann transport equation](data/bte)

## Code

- [Poisson equation](src/poisson/deeponet_poisson.py)
- [Boltzmann transport equation](src/bte)
- Boltzmann transport equation for inverse design
    - [Genetic algorithm](src/bte_ga)
    - [Topology optimization](src/bte_to)

## Cite this work

If you use this data or code for academic research, you are encouraged to cite the following paper:

```
@article{PhysRevResearch.4.023210,
  title   = {Multifidelity deep neural operators for efficient learning of partial differential equations with application to fast inverse design of nanoscale heat transport},
  author  = {Lu, Lu and Pestourie, Rapha\"el and Johnson, Steven G. and Romano, Giuseppe},
  journal = {Phys. Rev. Research},
  volume  = {4},
  issue   = {2},
  pages   = {023210},
  year    = {2022},
  doi     = {10.1103/PhysRevResearch.4.023210}
}
```

## Questions

To get help on how to use the data or code, simply open an issue in the GitHub "Issues" section.
