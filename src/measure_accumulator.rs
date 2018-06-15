use num_traits::AsPrimitive;

pub trait MeasureAccumulator<T> {
    fn new() -> Self;
    fn update_one(&mut self, known: &T, pred: &T);

    fn update<I: Iterator<Item = T>>(&mut self, known: I, predicted: I) {
        for (k, p) in known.zip(predicted) {
            self.update_one(&k, &p)
        }
    }
}

#[derive(Debug)]
pub struct PredictiveAccuracy {
    n_correct: usize,
    n_wrong: usize,
}

impl<T> MeasureAccumulator<T> for PredictiveAccuracy
where
    T: PartialEq,
{
    fn new() -> Self {
        PredictiveAccuracy {
            n_correct: 0,
            n_wrong: 0,
        }
    }

    fn update_one(&mut self, known: &T, pred: &T) {
        if known == pred {
            self.n_correct += 1;
        } else {
            self.n_wrong += 1;
        }
    }
}

#[derive(Debug)]
pub struct RootMeanSquaredError {
    sum_of_squares: f64,
    n: usize,
}

impl<T> MeasureAccumulator<T> for RootMeanSquaredError
where
    T: AsPrimitive<f64>,
{
    fn new() -> Self {
        RootMeanSquaredError {
            sum_of_squares: 0.0,
            n: 0,
        }
    }

    fn update_one(&mut self, known: &T, pred: &T) {
        let diff = known.as_() - pred.as_();
        self.sum_of_squares += diff * diff;
        self.n += 1;
    }
}
