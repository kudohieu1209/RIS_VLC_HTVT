# RIS-assisted VLC Simulation Summary

## Geometry

- Room size: 5.0 m x 5.0 m x 3.0 m
- AP LED position: (2.5, 2.5, 3.0) m
- Default PD position: (2.5, 1.0, 0.85) m
- RIS wall: y = 0.0 m
- Obstacle min corner: (2.15, 1.65, 0.85) m
- Obstacle max corner: (2.85, 1.95, 2.45) m

## Optimal RIS Position

- x_opt: 2.500 m
- z_opt: 1.420 m
- max SNR: 11.295 dB
- max data rate: 77.109 Mbps

## Scenario Results

| Scenario | LoS blocked | SNR (dB) | Data rate (Mbps) |
|---|---:|---:|---:|
| Scenario 1: No obstacle, without RIS | False | 32.480 | 215.807 |
| Scenario 2: Obstacle, without RIS | True | -inf | 0.000 |
| Scenario 3: Obstacle, with RIS | True | 10.862 | 74.441 |
| Scenario 4: No obstacle, with RIS | False | 33.172 | 220.406 |

## Improvement

- Delta rate in obstacle case: 74.441 Mbps

The obstacle is modeled as a 3D rectangular box. For each PD point, the simulation checks whether the AP-PD LoS segment intersects the box. If it intersects, the LoS component is blocked; otherwise the LoS component remains available.