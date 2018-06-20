//! Implementations of specific OpenML task types

mod supervised_classification;
mod supervised_regression;

use serde::de::DeserializeOwned;

pub use self::supervised_classification::SupervisedClassification;
pub use self::supervised_regression::SupervisedRegression;

use measure_accumulator::MeasureAccumulator;

pub trait Task {
    /// get task ID
    fn id(&self) -> &str;

    /// get task name
    fn name(&self) -> &str;

    /// run task, specifying the type of an entire feature column in `X`. This allows to run
    /// machine learning models that take features of different types, or named features in form
    /// of structs.
    fn run_static<X, Y, F, M>(&self, flow: F) -> M
    where
        F: Fn(&mut Iterator<Item = (&X, &Y)>, &mut Iterator<Item = &X>) -> Box<Iterator<Item = Y>>,
        X: DeserializeOwned,
        Y: DeserializeOwned,
        M: MeasureAccumulator<Y>;

    /// run task, specifying the feature type in `X`. This allows to run machine learning models
    /// that expect every feature to have the same type.
    fn run<X, Y, F, M>(&self, flow: F) -> M
    where
        F: Fn(&mut Iterator<Item = (&[X], &Y)>, &mut Iterator<Item = &[X]>)
            -> Box<Iterator<Item = Y>>,
        X: DeserializeOwned,
        Y: DeserializeOwned,
        M: MeasureAccumulator<Y>;
}
