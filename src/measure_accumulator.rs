//! Measure accumulators are summaries of model performance, such as classification accuracy or
//! regression error.

use std::cmp::Eq;
use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;

use num_traits::AsPrimitive;

/// Trait implemented by performance measures
pub trait MeasureAccumulator<T> {
    /// initialize new measure
    fn new() -> Self;

    /// update with one prediction
    fn update_one(&mut self, known: &T, pred: &T);

    /// get resulting performance
    fn result(&self) -> f64;

    /// update with multiple predictions
    fn update<I: Iterator<Item = T>>(&mut self, known: I, predicted: I) {
        for (k, p) in known.zip(predicted) {
            self.update_one(&k, &p)
        }
    }
}

/// Classification Accuracy: relative amount of correctly classified labels
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

/// Root Mean Squared Error
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

/// Adjusted Rand Index
#[derive(Debug)]
pub struct AdjustedRandIndex<T>
where
    T: Eq + Hash,
{
    contingency_table: HashMap<(T, T), usize>,
}

impl<T> MeasureAccumulator<T> for AdjustedRandIndex<T>
where
    T: Eq + Hash + Clone,
{
    fn new() -> Self {
        AdjustedRandIndex {
            contingency_table: HashMap::new(),
        }
    }

    fn update_one(&mut self, known: &T, pred: &T) {
        let n = self
            .contingency_table
            .entry((known.clone(), pred.clone()))
            .or_insert(0);
        *n += 1;
    }

    fn result(&self) -> f64 {
        let mut a = HashMap::new();
        let mut b = HashMap::new();

        let mut ri = 0usize;
        let mut n_tot = 0usize;

        for ((ak, bk), &n) in self.contingency_table.iter() {
            n_tot += n;
            ri += combinations(n);

            *a.entry(ak).or_insert(0usize) += n;
            *b.entry(bk).or_insert(0usize) += n;
        }

        let a_sum: usize = a.iter().map(|(_, &n)| combinations(n)).sum();
        let b_sum: usize = b.iter().map(|(_, &n)| combinations(n)).sum();

        let expected_ri = (a_sum as f64) * (b_sum as f64) / combinations(n_tot) as f64;
        let max_ri = (a_sum + b_sum) as f64 / 2.0;

        (ri as f64 - expected_ri) / (max_ri - expected_ri)
    }
}

fn combinations(n: usize) -> usize {
    if n % 2 == 0 {
        (n - 1) * (n / 2)
    } else {
        n * ((n - 1) / 2)
    }
}

#[test]
fn ari() {
    let labels_true = [0, 0, 0, 1, 1, 1];
    let labels_pred = [0, 0, 1, 1, 2, 2];

    let mut ari = AdjustedRandIndex::new();
    ari.update(labels_true.iter(), labels_pred.iter());

    assert_eq!(ari.result(), 0.24242424242424246);
}
