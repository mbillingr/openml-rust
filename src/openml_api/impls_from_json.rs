//! implementations to convert the API's JSON responses into corresponding Rust structures
use arff;
use arff::dynamic::DataSet as ArffDataSet;
use serde_json;

use dataset::DataSet;
use error::Result;
use procedures::{Fold, FrozenSets};
use tasks::{SupervisedClassification, SupervisedRegression};

use super::api_types::{CrossValItem, GenericResponse, TrainTest};
use super::web_access::get_cached;

impl DataSet {
    fn from_json(item: &serde_json::Value) -> Self {
        let v = &item["data_set"];
        let id = v["data_set_id"].as_str().unwrap();
        let target = v["target_feature"].as_str();

        let info_url = format!("https://www.openml.org/api/v1/json/data/{}", id);
        let info: GenericResponse = serde_json::from_str(&get_cached(&info_url).unwrap()).unwrap();

        let default_target = info.look_up("/data_set_description/default_target_attribute")
            .and_then(|v| v.as_str());

        let target = match (default_target, target) {
            (Some(s), None) | (_, Some(s)) => Some(s.to_owned()),
            (None, None) => None,
        };

        let dset_url = info.look_up("/data_set_description/url")
            .unwrap()
            .as_str()
            .unwrap();
        let dset_str = get_cached(&dset_url).unwrap();
        let dset = ArffDataSet::from_str(&dset_str).unwrap();

        DataSet { arff: dset, target }
    }
}

impl SupervisedClassification {
    pub fn from_json(task_json: &serde_json::Value) -> Self {
        let mut source_data = None;
        let mut estimation_procedure = None;
        //let mut cost_matrix = None;

        for input_item in task_json["input"].as_array().unwrap() {
            match input_item["name"].as_str() {
                Some("source_data") => source_data = Some(DataSet::from_json(input_item)),
                Some("estimation_procedure") => {
                    estimation_procedure = Some(Box::new(FrozenSets::from_json(input_item)))
                }
                //Some("cost_matrix") => cost_matrix = Some(input_item.into()),
                Some(_) => {}
                None => panic!("/task/input/name is not a string"),
            }
        }

        SupervisedClassification {
            id: task_json["task_id"].as_str().unwrap().to_owned(),
            name: task_json["task_name"].as_str().unwrap().to_owned(),
            source_data: source_data.unwrap(),
            estimation_procedure: estimation_procedure.unwrap(),
            //cost_matrix: cost_matrix.unwrap(),
        }
    }
}

impl SupervisedRegression {
    pub fn from_json(task_json: &serde_json::Value) -> Self {
        let mut source_data = None;
        let mut estimation_procedure = None;

        for input_item in task_json["input"].as_array().unwrap() {
            match input_item["name"].as_str() {
                Some("source_data") => source_data = Some(DataSet::from_json(input_item)),
                Some("estimation_procedure") => {
                    estimation_procedure = Some(Box::new(FrozenSets::from_json(input_item)))
                }
                Some(_) => {}
                None => panic!("/task/input/name is not a string"),
            }
        }

        SupervisedRegression {
            id: task_json["task_id"].as_str().unwrap().to_owned(),
            name: task_json["task_name"].as_str().unwrap().to_owned(),
            source_data: source_data.unwrap(),
            estimation_procedure: estimation_procedure.unwrap(),
        }
    }
}

impl FrozenSets {
    fn from_json(item: &serde_json::Value) -> Self {
        let v = &item["estimation_procedure"];
        let typ = v["type"].as_str();
        let splits = v["data_splits_url"].as_str();

        match (typ, splits) {
            (_, Some(url)) => FrozenSets::from_url(url).unwrap(),
            _ => unimplemented!(),
        }
    }

    fn from_url(url: &str) -> Result<Self> {
        let raw = get_cached(url)?;
        let data: Vec<CrossValItem> = arff::from_str(&raw)?;

        let mut folds = vec![];
        for item in data {
            if item.repeat >= folds.len() {
                folds.resize(item.repeat + 1, vec![]);
            }
            let mut rep = &mut folds[item.repeat];

            if item.fold >= rep.len() {
                rep.resize(item.fold + 1, Fold::new());
            }
            let mut fold = &mut rep[item.fold];

            match item.purpose {
                TrainTest::Train => fold.trainset.push(item.rowid),
                TrainTest::Test => fold.testset.push(item.rowid),
            }
        }

        Ok(FrozenSets { folds })
    }
}
