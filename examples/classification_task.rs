extern crate openml;

use openml::baseline::NaiveBayesClassifier;
use openml::prelude::*;
use openml::{PredictiveAccuracy, SupervisedClassification};

fn main() {
    // Load "Supervised Classification on iris" task (https://www.openml.org/t/59)
    let task = SupervisedClassification::from_openml(59).unwrap();

    println!("Task: {}", task.name());

    // run the task
    let result: PredictiveAccuracy<_> = task.run(|train, test| {
        // train classifier
        let nbc: NaiveBayesClassifier<u8> = train.map(|(x, y)| (x, y)).collect();

        // test classifier
        let y_out: Vec<_> = test.map(|x| nbc.predict(x)).collect();

        Box::new(y_out.into_iter())
    });

    println!("Classification Accuracy: {}", result.result());
}
