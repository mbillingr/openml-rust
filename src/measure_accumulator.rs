use std::marker::PhantomData;
use num_traits::AsPrimitive;

pub trait MeasureAccumulator<T> {
    fn new() -> Self;
    fn update_one(&mut self, known: &T, pred: &T);
    fn result(&self) -> f64;

    fn update<I: Iterator<Item = T>>(&mut self, known: I, predicted: I) {
        for (k, p) in known.zip(predicted) {
            self.update_one(&k, &p)
        }
    }
}

#[derive(Debug)]
pub struct PredictiveAccuracy<T> {
    n_correct: usize,
    n_wrong: usize,
    _t: PhantomData<T>,
}

impl<T> MeasureAccumulator<T> for PredictiveAccuracy<T>
where
    T: PartialEq,
{
    fn new() -> Self {
        PredictiveAccuracy {
            n_correct: 0,
            n_wrong: 0,
            _t: PhantomData,
        }
    }

    fn update_one(&mut self, known: &T, pred: &T) {
        if known == pred {
            self.n_correct += 1;
        } else {
            self.n_wrong += 1;
        }
    }

    fn result(&self) -> f64 {
        self.n_correct as f64 / (self.n_correct + self.n_wrong) as f64
    }
}

#[derive(Debug)]
pub struct RootMeanSquaredError<T> {
    sum_of_squares: f64,
    n: usize,
    _t: PhantomData<T>,
}

impl<T> MeasureAccumulator<T> for RootMeanSquaredError<T>
where
    T: AsPrimitive<f64>,
{
    fn new() -> Self {
        RootMeanSquaredError {
            sum_of_squares: 0.0,
            n: 0,
            _t: PhantomData,
        }
    }

    fn update_one(&mut self, known: &T, pred: &T) {
        let diff = known.as_() - pred.as_();
        self.sum_of_squares += diff * diff;
        self.n += 1;
    }

    fn result(&self) -> f64 {
        (self.sum_of_squares / self.n as f64).sqrt()
    }
}
