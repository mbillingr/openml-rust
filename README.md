# openml-rust
A rust interface to [OpenML](http://openml.org/).

The aim of this crate is to give rust code access to Machine Learning data hosted by OpenML.
Thus, Machine Learning algorithms developed in Rust can be easily applied to state-of-the-art
data sets and their performance compared to existing implementations in a reproducable way.

## Example

```rust
extern crate openml;

use openml::prelude::*;
use openml::{PredictiveAccuracy, SupervisedClassification};
use openml::baseline::NaiveBayesClassifier;

fn main() {
    // Load "Supervised Classification on iris" task (https://www.openml.org/t/59)
    let task = SupervisedClassification::from_openml(59).unwrap();

    println!("Task: {}", task.name());

    // run the task
    let result: PredictiveAccuracy<_> = task.run(|train, test| {
        // train classifier
        let nbc: NaiveBayesClassifier<u8> = train
            .map(|(x, y)| (x, y))
            .collect();

        // test classifier
        let y_out: Vec<_> = test
            .map(|x| nbc.predict(x))
            .collect();

        Box::new(y_out.into_iter())
    });

    println!("Classification Accuracy: {}", result.result());
}
```

## Goals
- [x] get data sets
- [x] get tasks
- [x] get splits
- [ ] task types
  - [x] Supervised Classification
  - [x] Supervised Regression
  - [ ] Learning Curve
  - [ ] Clustering
- [x] run tasks
  - <s>[ ] Learner/Predictor trait for use with tasks</s>
  - [x] Task runner takes a closure for learning and prediction
  - [x] Data type strategy:
    - <s>a: burden the ML model with figuring out how to deal with dynamic types</s>
    - <s>b: cast everything to f64</s>
    - <s>c: make type casting part of the feature extraction pipeline</s>
    - Generics allow type selection at compile time
- [ ] make openml.org optional (manual construction of tasks)
    
  
## Future Maybe-Goals
- flow support
- run support
- full OpenML API support
- authentication
- more tasks
  - Supervised Datastream Classification
  - Machine Learning Challenge
  - Survival Analysis
  - Subgroup Discovery

## Non-Goals
- implementations of machine learning algorithms
