//! Implementation of a Naive Linear Regression model

use std::f64;
use std::iter::FromIterator;

/// A Naive Linear Regression model
///
/// This is univariate regression on a single feature. During training the best feature is selected.
/// The model is trained by consuming an iterator over the training data:
/// ```
/// # use openml::baseline::NaiveLinearRegression;
/// # let data: Vec<(&[f64], &f64)> = vec![];
/// let model: NaiveLinearRegression = data
///     .into_iter()
///     .collect();
/// ```
#[derive(Debug)]
pub struct NaiveLinearRegression
{
    slope: f64,
    intercept: f64,
    feature: usize,
}

impl<'a, J> FromIterator<(J, &'a f64)> for NaiveLinearRegression
    where
        J: IntoIterator<Item=&'a f64>,
{
    fn from_iter<I: IntoIterator<Item=(J, &'a f64)>>(iter: I) -> Self {
        let mut feature_columns = Vec::new();
        let mut target_column = Vec::new();

        for (x, &y) in iter {
            target_column.push(y);
            for (i, &xi) in x.into_iter().enumerate() {
                if i >= feature_columns.len() {
                    feature_columns.push(Vec::new());
                }

                feature_columns[i].push(xi);
            }
        }

        let mut y_mean = 0.0;
        for y in &target_column {
            y_mean += *y;
        }
        y_mean /= target_column.len() as f64;

        let mut best_err = f64::INFINITY;
        let mut best_slope = f64::NAN;
        let mut best_intercept = f64::NAN;
        let mut best_feature = 0;

        for (i, feature) in feature_columns.iter().enumerate() {
            let mut x_mean = 0.0;
            for x in feature {
                x_mean += *x;
            }
            x_mean /= feature.len() as f64;

            let mut x_var = 0.0;
            let mut covar = 0.0;
            for (x, y) in feature.iter().zip(target_column.iter()) {
                let x = *x - x_mean;
                let y = *y - y_mean;

                x_var += x * x;
                covar += x * y;
            }

            let slope = covar / x_var;
            let intercept = y_mean - slope * x_mean;

            let err: f64 = feature.iter()
                .zip(target_column.iter())
                .map(|(&x, &y)| intercept + slope * x - y)
                .map(|r| r * r)
                .sum();

            if err < best_err {
                best_err = err;
                best_slope = slope;
                best_intercept = intercept;
                best_feature = i;
            }
        }

        NaiveLinearRegression {
            slope: best_slope,
            intercept: best_intercept,
            feature: best_feature,
        }
    }
}

impl NaiveLinearRegression
{
    /// predict target value for a single feature vector
    pub fn predict(&self, x: &[f64]) -> f64 {
        self.intercept + x[self.feature] * self.slope
    }
}

#[test]
fn nbc_flat() {
    let data = vec![(vec![1.0, 2.0], 3.0),
                    (vec![2.0, 1.0], 3.0),
                    (vec![1.0, 5.0], 3.0),
                    (vec![2.0, 6.0], 3.0)];

    let nlr: NaiveLinearRegression = data
        .iter()
        .map(|(x, y)| (x, y))
        .collect();

    assert_eq!(nlr.predict(&[1.5, 1.5]), 3.0);
    assert_eq!(nlr.predict(&[5.5, 1.5]), 3.0);
    assert_eq!(nlr.predict(&[1.5, 5.5]), 3.0);
    assert_eq!(nlr.predict(&[5.5, 5.5]), 3.0);
}

#[test]
fn nbc_slope() {
    let data = vec![(vec![1.0, 2.0], 8.0),
                    (vec![2.0, 1.0], 9.0),
                    (vec![1.0, 5.0], 5.0),
                    (vec![2.0, 6.0], 4.0)];

    let nlr: NaiveLinearRegression = data
        .iter()
        .map(|(x, y)| (x, y))
        .collect();

    assert_eq!(nlr.predict(&[1.5, 1.5]), 8.5);
    assert_eq!(nlr.predict(&[5.5, 1.5]), 8.5);
    assert_eq!(nlr.predict(&[1.5, 5.5]), 4.5);
    assert_eq!(nlr.predict(&[5.5, 5.5]), 4.5);
}
