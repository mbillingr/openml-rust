use super::{Fold, Procedure};

#[derive(Debug)]
pub(crate) struct FrozenSets {
    pub(crate) folds: Vec<Vec<Fold>>,
}

impl Procedure for FrozenSets {
    fn iter<'a>(&'a self) -> Box<'a + Iterator<Item = &'a Fold>> {
        let iter = self.folds.iter().flat_map(|inner| inner.iter());
        Box::new(iter)
    }
}
