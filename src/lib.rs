extern crate app_dirs;
extern crate arff;
extern crate fs2;
extern crate futures;
extern crate hyper;
extern crate hyper_tls;
#[macro_use]
extern crate log;
extern crate ndarray;
extern crate num_traits;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate serde_json;
#[cfg(test)]
extern crate simple_logger;
extern crate time;
extern crate tokio_core;

pub mod baseline;
mod dataset;
mod error;
mod measure_accumulator;
mod openml_api;
mod procedures;
mod tasks;

pub use measure_accumulator::{
    MeasureAccumulator,
    PredictiveAccuracy,
    RootMeanSquaredError
};

pub use tasks::{
    SupervisedClassification,
    SupervisedRegression,
    Task
};

#[cfg(test)]
mod tests {
    use log::Level;
    use time::PreciseTime;

    use measure_accumulator::PredictiveAccuracy;

    use super::*;

    #[test]
    fn apidev() {
        let task = SupervisedClassification::from_openml(166850).unwrap();

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

        let result: PredictiveAccuracy<_> = task.run(|_train, test| {
            let y_out: Vec<_> = test.map(|_row: &[f64]| 0).collect();
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
