//! Implementation of simple baseline models, used for testing and demonstration.

mod naive_bayes_classifier;
mod naive_linear_regression;

pub use self::naive_bayes_classifier::NaiveBayesClassifier;
pub use self::naive_linear_regression::NaiveLinearRegression;