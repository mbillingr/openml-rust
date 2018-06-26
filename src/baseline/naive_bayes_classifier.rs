//! Implementation of a Gaussian Naive Bayes Classifier

use std::cmp::Ordering;
use std::collections::HashMap;
use std::f64;
use std::fmt;
use std::hash::Hash;
use std::iter::FromIterator;

/// A Gaussian Naive Bayes Classifier
///
/// The classifier is trained by consuming an iterator over the training data:
/// ```
/// # use openml::baseline::NaiveBayesClassifier;
/// # let data: Vec<(&[f64], &u8)> = vec![];
/// let nbc: NaiveBayesClassifier<_> = data
///     .into_iter()
///     .collect();
/// ```
#[derive(Debug)]
pub struct NaiveBayesClassifier<C>
where C: Eq + Hash
{
    class_distributions: HashMap<C, FeatureDistribution>,
}

/// Distribution of each feature column
#[derive(Debug, Clone)]
struct FeatureDistribution {
    distributions: Vec<NormalDistribution>
}

/// Univariate Normal Distribution
#[derive(Copy, Clone)]
struct NormalDistribution {
    sum: f64,
    sqsum: f64,
    n: usize
}

impl<'a, C: 'a, J> FromIterator<(J, &'a C)> for NaiveBayesClassifier<C>
where
    J: IntoIterator<Item=&'a f64>,
    C: Eq + Hash + Copy,
{
    fn from_iter<I: IntoIterator<Item=(J, &'a C)>>(iter: I) -> Self {
        let mut class_distributions = HashMap::new();

        for (x, &y) in iter {
            let distributions = &mut class_distributions
                .entry(y)
                .or_insert(FeatureDistribution::new())
                .distributions;

            for (i, &xi) in x.into_iter().enumerate() {
                if i >= distributions.len() {
                    distributions.resize(1 + i, NormalDistribution::new());
                }

                distributions[i].update(xi);
            }
        }

        NaiveBayesClassifier {
            class_distributions
        }
    }
}

impl<C> NaiveBayesClassifier<C>
where  C: Eq + Hash + Copy,
{
    /// predict target class for a single feature vector
    pub fn predict(&self, x: &[f64]) -> C {
        self.class_distributions
            .iter()
            .map(|(c, dists)| {
                let mut lnprob = 0.0;
                for (&xi, dist) in x.iter().zip(dists.distributions.iter()) {
                    lnprob += dist.lnprob(xi);
                }
                (c, lnprob)
            })
            .max_by(|(_, lnp1), (_, lnp2)| {
                if lnp1 > lnp2 {
                    Ordering::Greater
                } else if lnp1 == lnp2 {
                    Ordering::Equal
                } else {
                    Ordering::Less
                }
            })
            .map(|(&c, _)| c)
            .unwrap()
    }
}

impl FeatureDistribution {
    fn new() -> Self {
        FeatureDistribution {
            distributions: Vec::new()
        }
    }
}

impl NormalDistribution {
    fn new() -> Self {
        NormalDistribution {
            sum: 0.0,
            sqsum: 0.0,
            n: 0
        }
    }

    fn update(&mut self, x: f64) {
        self.sum += x;
        self.sqsum += x * x;
        self.n += 1;
    }

    fn mean(&self) -> f64 {
        self.sum / self.n as f64
    }

    fn variance(&self) -> f64 {
        (self.sqsum - (self.sum * self.sum) / self.n as f64) / (self.n as f64 - 1.0)
    }

    fn lnprob(&self, x: f64) -> f64 {
        let v = self.variance();
        let xm = x - self.mean();

        0.5 * ((1.0 / (2.0 * f64::consts::PI * v)).ln() - (xm * xm) / v)

    }
}

impl fmt::Debug for NormalDistribution {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "N{{{}; {}}}", self.mean(), self.variance())
    }
}

#[test]
fn nbc() {
    let data = vec![(vec![1.0, 2.0], 'A'),
                    (vec![2.0, 1.0], 'A'),
                    (vec![1.0, 5.0], 'B'),
                    (vec![2.0, 6.0], 'B')];

    let nbc: NaiveBayesClassifier<_> = data
        .iter()
        .map(|(x, y)| (x, y))
        .collect();

    assert_eq!(nbc.predict(&[1.5, 1.5]), 'A');
    assert_eq!(nbc.predict(&[5.5, 1.5]), 'A');
    assert_eq!(nbc.predict(&[1.5, 5.5]), 'B');
    assert_eq!(nbc.predict(&[5.5, 5.5]), 'B');
}
