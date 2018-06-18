use arff::dynamic::de::from_dataset;
use serde::de::DeserializeOwned;

use dataset::DataSet;
use measure_accumulator::MeasureAccumulator;
use procedures::Procedure;

pub struct SupervisedClassification {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) source_data: DataSet,
    pub(crate) estimation_procedure: Box<Procedure>,
    //pub(crate) cost_matrix: CostMatrix,
}

impl SupervisedClassification {
    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn run_static<X, Y, F, M>(&self, flow: F) -> M
    where
        F: Fn(&mut Iterator<Item = (&X, &Y)>, &mut Iterator<Item = &X>) -> Box<Iterator<Item = Y>>,
        X: DeserializeOwned,
        Y: DeserializeOwned,
        M: MeasureAccumulator<Y>,
    {
        let (dx, dy) = self.source_data
            .clone_split()
            .expect("Supervised Classification requires a target column");

        let x: Vec<X> = from_dataset(&dx).unwrap();
        let y: Vec<Y> = from_dataset(&dy).unwrap();

        let mut measure = M::new();

        for fold in self.estimation_procedure.iter() {
            let mut train = fold.trainset.iter().map(|&i| (&x[i], &y[i]));

            let mut test = fold.testset.iter().map(|&i| &x[i]);

            let predictit = flow(&mut train, &mut test);

            for (known, pred) in fold.testset.iter().map(|&i| &y[i]).zip(predictit) {
                measure.update_one(known, &pred);
            }
        }

        measure
    }

    pub fn run<X, Y, F, M>(&self, flow: F) -> M
    where
        F: Fn(&mut Iterator<Item = (&[X], &Y)>, &mut Iterator<Item = &[X]>)
            -> Box<Iterator<Item = Y>>,
        X: DeserializeOwned,
        Y: DeserializeOwned,
        M: MeasureAccumulator<Y>,
    {
        let (dx, dy) = self.source_data
            .clone_split()
            .expect("Supervised Classification requires a target column");

        let x: Vec<X> = from_dataset(&dx).unwrap();
        let y: Vec<Y> = from_dataset(&dy).unwrap();

        let mut measure = M::new();

        for fold in self.estimation_procedure.iter() {
            let mut train = fold.trainset
                .iter()
                .map(|&i| (&x[i * dx.n_cols()..(i + 1) * dx.n_cols()], &y[i]));

            let mut test = fold.testset
                .iter()
                .map(|&i| &x[i * dx.n_cols()..(i + 1) * dx.n_cols()]);

            let predictit = flow(&mut train, &mut test);

            for (known, pred) in fold.testset.iter().map(|&i| &y[i]).zip(predictit) {
                measure.update_one(known, &pred);
            }
        }

        measure
    }
}
