# 6. Scenario permutations (CP-SAT helper)

Incorporating the CP-SAT permutation helper in the optimization process allows for enhanced examination of cost scenarios and benefit levels through defined uplift constants.

## Cost Scenarios
- **Base Real**
- **P95 Real**

## Benefit Levels
- **Base**
- **High**

### Uplift Constants
- **P95_COST_UPLIFT = 1.2**: This constant is applied to the P95 Real cost scenario to enhance the cost evaluation process.
- **BEN_HIGH_UPLIFT = 1.2**: Used to elevate the benefits under the High benefit level, ensuring a more favorable comparison against the base scenario.

### Example usage
The helper function `solve_with_permutations` can be utilized as follows:
```python
result = solve_with_permutations(data, P95_COST_UPLIFT, BEN_HIGH_UPLIFT)
```

This function optimally evaluates the given data under both the P95 cost uplift and the high benefit uplift, aiding in decision-making based on more nuanced financial projections.
