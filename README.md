# openml-rust
A rust interface to [OpenML](http://openml.org/).

## Goals
- [ ] get data sets
- [ ] get tasks
- [ ] get splits
- [ ] task types
  - [ ] probably requires a specialized implementation for each type
- [ ] run tasks
  - [ ] Learner/Predictor trait for use with tasks
  - [ ] Data type strategy:
    - a: burden the ML model with figuring out how to deal with dynamic types
    - b: cast everything to f64
    - c: make type casting part of the feature extraction pipeline
  
## Future Maybe-Goals
- flow support
- run support
- full OpenML API support

## Non-Goals
- machine learning algorithms
