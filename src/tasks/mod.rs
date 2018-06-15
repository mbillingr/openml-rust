mod supervised_classification;
mod supervised_regression;

use serde::de::DeserializeOwned;

pub use self::supervised_classification::SupervisedClassification;
pub use self::supervised_regression::SupervisedRegression;

use measure_accumulator::MeasureAccumulator;

pub trait Task {
    fn id(&self) -> &str;
    fn name(&self) -> &str;

    fn run_static<X, Y, F, M>(&self, flow: F) -> M
    where
        F: Fn(&mut Iterator<Item = (&X, &Y)>, &mut Iterator<Item = &X>) -> Box<Iterator<Item = Y>>,
        X: DeserializeOwned,
        Y: DeserializeOwned,
        M: MeasureAccumulator<Y>;

    fn run<X, Y, F, M>(&self, flow: F) -> M
    where
        F: Fn(&mut Iterator<Item = (&[X], &Y)>, &mut Iterator<Item = &[X]>)
            -> Box<Iterator<Item = Y>>,
        X: DeserializeOwned,
        Y: DeserializeOwned,
        M: MeasureAccumulator<Y>;
}
