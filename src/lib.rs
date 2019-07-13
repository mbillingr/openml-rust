//! # openml-rust
//!
//! The openml crate provides functions to fetch tasks and data sets from https://openml.org, and
//! run them with machine learning models.
//!
//! ## Example
//!
//! ```rust
//!extern crate openml;
//!
//!use openml::prelude::*;
//!use openml::{PredictiveAccuracy, SupervisedClassification};
//!use openml::baseline::NaiveBayesClassifier;
//!
//!fn main() {
//!    // Load "Supervised Classification on iris" task (https://www.openml.org/t/59)
//!    let task = SupervisedClassification::from_openml(59).unwrap();
//!
//!    println!("Task: {}", task.name());
//!
//!    // run the task
//!    let result: PredictiveAccuracy<_> = task.run(|train, test| {
//!        // train classifier
//!        let nbc: NaiveBayesClassifier<u8> = train
//!            .map(|(x, y)| (x, y))
//!            .collect();
//!
//!        // test classifier
//!        let y_out: Vec<_> = test
//!            .map(|x| nbc.predict(x))
//!            .collect();
//!
//!        Box::new(y_out.into_iter())
//!    });
//!
//!    println!("Classification Accuracy: {}", result.result());
//!}
//! ```

extern crate app_dirs;
extern crate arff;
extern crate fs2;
extern crate reqwest;
#[macro_use]
extern crate log;
extern crate num_traits;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate serde_json;
#[cfg(test)]
extern crate simple_logger;
#[cfg(test)]
extern crate time;

pub mod baseline;
mod dataset;
mod error;
mod measure_accumulator;
mod openml_api;
pub mod prelude;
mod procedures;
mod tasks;

pub use crate::measure_accumulator::{
    MeasureAccumulator, PredictiveAccuracy, RootMeanSquaredError,
};

pub use crate::tasks::{SupervisedClassification, SupervisedRegression, Task};

#[cfg(test)]
mod tests {
    use log::Level;
    use time::PreciseTime;

    use crate::baseline::NaiveBayesClassifier;
    use crate::measure_accumulator::PredictiveAccuracy;

    use super::*;

    #[test]
    fn apidev() {
        let task = SupervisedClassification::from_openml(59).unwrap();

        println!("{}", task.name());

        let result: PredictiveAccuracy<_> = task.run_static(|_train, test| {
            let y_out: Vec<_> = test.map(|_row: &[f64; 4]| 0).collect();
            Box::new(y_out.into_iter())
        });

        println!("{:#?}", result);

        #[allow(dead_code)]
        #[derive(Deserialize)]
        struct Row {
            sepallength: f32,
            sepalwidth: f32,
            petallength: f32,
            petalwidth: f32,
        }

        let result: PredictiveAccuracy<_> = task.run_static(|train, test| {
            let (_x_train, _y_train): (Vec<&Row>, Vec<i32>) = train.unzip();
            let y_out: Vec<_> = test.map(|_row: &Row| 0).collect();
            Box::new(y_out.into_iter())
        });

        println!("{:#?}", result);

        let result: PredictiveAccuracy<_> = task.run(|train, test| {
            // train classifier
            let nbc: NaiveBayesClassifier<u8> = train.map(|(x, y)| (x, y)).collect();

            // test classifier
            let y_out: Vec<_> = test.map(|x| nbc.predict(x)).collect();

            Box::new(y_out.into_iter())
        });

        println!("{:#?}", result);
    }

    #[test]
    fn apidev2() {
        use simple_logger;
        simple_logger::init_with_level(Level::Info).unwrap();

        let start = PreciseTime::now();

        let task = SupervisedClassification::from_openml(146825).unwrap();
        //let task = SupervisedClassification::from_openml(167147).unwrap();

        let end = PreciseTime::now();

        let result: PredictiveAccuracy<_> = task.run(|_train, test| {
            let y_out: Vec<_> = test.map(|_row: &[u8]| 0).collect();
            Box::new(y_out.into_iter())
        });

        println!("{:#?}", result);

        println!("loading took {} seconds.", start.to(end));
    }
}
