//! Validation procedures

mod frozen_sets;

pub(crate) use self::frozen_sets::FrozenSets;

/// Validation procedures support iteration over cross-validation folds
pub(crate) trait Procedure {
    fn iter<'a>(&'a self) -> Box<'a + Iterator<Item = &'a Fold>>;
}

/// A single cross-validation fold, consisting of a training set and a testing set
#[derive(Debug, Clone)]
pub(crate) struct Fold {
    pub(crate) trainset: Vec<usize>,
    pub(crate) testset: Vec<usize>,
}

impl Fold {
    pub fn new() -> Self {
        Fold {
            trainset: Vec::new(),
            testset: Vec::new(),
        }
    }
}
