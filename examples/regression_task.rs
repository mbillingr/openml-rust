extern crate openml;

use openml::baseline::NaiveLinearRegression;
use openml::prelude::*;
use openml::{RootMeanSquaredError, SupervisedRegression};

fn main() {
    // Load "Supervised Regression on liver-disorders" task (https://www.openml.org/t/52948)
    let task = SupervisedRegression::from_openml(52948).unwrap();

    println!("Task: {}", task.name());

    // run the task
    let result: RootMeanSquaredError<_> = task.run(|train, test| {
        // train model
        let model: NaiveLinearRegression = train.map(|(x, y)| (x, y)).collect();

        // test model
        let y_out: Vec<_> = test.map(|x| model.predict(x)).collect();

        Box::new(y_out.into_iter())
    });

    println!("Root Mean Squared Error: {}", result.result());
}
