use arff::dynamic::DataSet as ArffDataSet;

#[derive(Debug)]
pub(crate) struct DataSet {
    pub(crate) arff: ArffDataSet,
    pub(crate) target: Option<String>,
}

impl DataSet {
    pub(crate) fn clone_split(&self) -> Option<(ArffDataSet, ArffDataSet)> {
        match self.target {
            None => None,
            Some(ref col) => {
                let data = self.arff.clone();
                Some(data.split_one(col))
            }
        }
    }
}
