use std;
use std::borrow::Cow;
use std::error::Error as StdError;
use std::fs;
use std::io::{Read, Write};
use std::mem;
use std::ops::Index;
use std::result;
use std::string;
use std::{thread, time};
use std::path::Path;

use arff;
use fs2::FileExt;
use futures::{Future, Stream};
use hyper;
use hyper_tls::{self, HttpsConnector};
use log::Level;
use num_traits::ToPrimitive;
use serde;
use serde_json;
use time::PreciseTime;
use tokio_core::reactor::Core;

pub type Result<T> = result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    IoError(std::io::Error),
    Utf8Error(string::FromUtf8Error),
    HyperError(hyper::Error),
    HyperUriError(hyper::error::UriError),
    HyperTlsError(hyper_tls::Error),
    JsonError(serde_json::Error),
    ArffError(arff::Error),
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self { Error::IoError(e) }
}

impl From<string::FromUtf8Error> for Error {
    fn from(e: string::FromUtf8Error) -> Self { Error::Utf8Error(e) }
}

impl From<hyper::Error> for Error {
    fn from(e: hyper::Error) -> Self { Error::HyperError(e) }
}

impl From<hyper::error::UriError> for Error {
    fn from(e: hyper::error::UriError) -> Self { Error::HyperUriError(e) }
}

impl From<hyper_tls::Error> for Error {
    fn from(e: hyper_tls::Error) -> Self { Error::HyperTlsError(e) }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self { Error::JsonError(e) }
}

impl From<arff::Error> for Error {
    fn from(e: arff::Error) -> Self { Error::ArffError(e) }
}


pub struct OpenML {
}

impl OpenML {
    pub fn new() -> Self {
        OpenML {}
    }

    pub fn task<'a, T: Id>(&mut self, id: T) -> Result<Task> {
        let url = format!("https://www.openml.org/api/v1/json/task/{}", id.as_string());
        let raw_task = get_cached(&url)?;
        let response: GenericResponse = serde_json::from_str(&raw_task)?;

        let task = response.look_up("/task").unwrap();

        Ok(Task {
            task_id: task["task_id"].as_str().unwrap().to_owned(),
            task_name: task["task_name"].as_str().unwrap().to_owned(),
            task_type: OpenML::task_type(task),
        })
    }

    fn task_type(task_json: &serde_json::Value) -> TaskType {
        let input = task_json["input"].as_array().unwrap();

        match task_json["task_type_id"].as_str() {
            Some("1") => TaskType::SupervisedClassification(SupervisedClassification::new(input)),
            Some("2") => TaskType::SupervisedRegression(SupervisedRegression::new(input)),
            tt @ _ => panic!("unsupported task type {:?}", tt)
        }
    }
}


pub trait Id {
    fn as_string(&self) -> Cow<str>;
    fn as_u32(&self) -> u32;
}

impl Id for String {
    #[inline(always)]
    fn as_string(&self) -> Cow<str> { Cow::from(self.as_ref()) }

    #[inline(always)]
    fn as_u32(&self) -> u32 { self.parse().unwrap() }
}

impl<'a> Id for &'a str {
    #[inline(always)]
    fn as_string(&self) -> Cow<str> { Cow::from(*self) }

    #[inline(always)]
    fn as_u32(&self) -> u32 { self.parse().unwrap() }
}

impl Id for u32 {
    #[inline(always)]
    fn as_string(&self) -> Cow<str> { Cow::from(format!("{}", self)) }

    #[inline(always)]
    fn as_u32(&self) -> u32 { *self }
}


#[derive(Debug, Serialize, Deserialize)]
struct GenericResponse(serde_json::Value);

impl GenericResponse {
    #[inline(always)]
    fn look_up<'a>(&'a self, p: &str) -> Option<&'a serde_json::Value> {
        self.0.pointer(p)
    }
}

#[derive(Debug)]
pub struct Task {
    task_id: String,
    task_name: String,
    task_type: TaskType,
}


type FlowFunction = Fn(arff::Array<f64>, arff::Array<f64>, arff::Array<f64>) -> Vec<f64>;


impl Task {
    pub fn name(&self) -> &str {
        &self.task_name
    }

    /// run task with statically known row size and column types
    pub fn run_static<X, Y, F>(&self, flow: F) -> MeasureAccumulator
    where F: Fn(&mut Iterator<Item=(&X, &Y)>, &mut Iterator<Item=&X>) -> Box<Iterator<Item=Y>>,
          X: serde::de::DeserializeOwned,
          Y: serde::de::DeserializeOwned,
          Y: ToPrimitive,
    {
        self.task_type.run_static(&self, flow)
    }

    /// run task with unknown row size
    pub fn run<X, Y, F>(&self, flow: F) -> MeasureAccumulator
    where F: Fn(&mut Iterator<Item=(&[X], &Y)>, &mut Iterator<Item=&[X]>) -> Box<Iterator<Item=Y>>,
          X: serde::de::DeserializeOwned,
          Y: serde::de::DeserializeOwned,
          Y: ToPrimitive,
    {
        self.task_type.run(&self, flow)
    }
}

#[derive(Debug)]
enum TaskType {
    SupervisedRegression(SupervisedRegression),
    SupervisedClassification(SupervisedClassification),
}

impl TaskType {
    fn run_static<X, Y, F>(&self, task: &Task, flow: F) -> MeasureAccumulator
    where F: Fn(&mut Iterator<Item=(&X, &Y)>, &mut Iterator<Item=&X>) -> Box<Iterator<Item=Y>>,
          X: serde::de::DeserializeOwned,
          Y: serde::de::DeserializeOwned,
          Y: ToPrimitive,
    {
        match *self {
            TaskType::SupervisedRegression(ref t) => t.run_static(task, flow),
            TaskType::SupervisedClassification(ref t) => t.run_static(task, flow),
        }
    }

    fn run<X, Y, F>(&self, task: &Task, flow: F) -> MeasureAccumulator
    where F: Fn(&mut Iterator<Item=(&[X], &Y)>, &mut Iterator<Item=&[X]>) -> Box<Iterator<Item=Y>>,
          X: serde::de::DeserializeOwned,
          Y: serde::de::DeserializeOwned,
          Y: ToPrimitive,
    {
        match *self {
            TaskType::SupervisedRegression(ref t) => t.run(task, flow),
            TaskType::SupervisedClassification(ref t) => t.run(task, flow),
        }
    }
}


#[derive(Debug)]
struct SupervisedRegression {
    source_data: DataSet,
    estimation_procedure: Procedure,
    evaluation_measures: Measure,
}

impl SupervisedRegression {
    fn new(input_json: &Vec<serde_json::Value>) -> Self {
        let mut source_data = None;
        let mut estimation_procedure = None;
        let mut evaluation_measures = None;

        for input_item in input_json {
            match input_item["name"].as_str() {
                Some("source_data") => source_data = Some(input_item.into()),
                Some("estimation_procedure") => estimation_procedure = Some(input_item.into()),
                Some("evaluation_measures") => evaluation_measures = Measure::new(input_item),
                Some(_) => {}
                None => panic!("/task/input/name is not a string")
            }
        }

        SupervisedRegression {
            source_data: source_data.unwrap(),
            estimation_procedure: estimation_procedure.unwrap(),
            evaluation_measures: evaluation_measures.unwrap(),
        }
    }

    fn run_static<X, Y, F>(&self, task: &Task, flow: F) -> MeasureAccumulator
    where F: Fn(&mut Iterator<Item=(&X, &Y)>, &mut Iterator<Item=&X>) -> Box<Iterator<Item=Y>>,
          X: serde::de::DeserializeOwned,
          Y: serde::de::DeserializeOwned,
          Y: ToPrimitive,
    {
        let (dx, dy) = self.source_data
            .clone_split()
            .expect("Supervised Regression requires a target column");

        let x: Vec<X> = arff::dynamic::de::from_dataset(&dx).unwrap();
        let y: Vec<Y> = arff::dynamic::de::from_dataset(&dy).unwrap();

        let mut measure = self.evaluation_measures.create();

        for fold in self.estimation_procedure.iter() {
            let mut train = fold.trainset
                .iter()
                .map(|&i| (&x[i], &y[i]));

            let mut test = fold.testset
                .iter()
                .map(|&i| &x[i]);

            let predictit = flow(&mut train, &mut test);

            for (known, pred) in fold.testset.iter().map(|&i| &y[i]).zip(predictit) {
                measure.update_one(known, &pred);
            }
        }

        measure
    }

    fn run<X, Y, F>(&self, task: &Task, flow: F) -> MeasureAccumulator
    where F: Fn(&mut Iterator<Item=(&[X], &Y)>, &mut Iterator<Item=&[X]>) -> Box<Iterator<Item=Y>>,
          X: serde::de::DeserializeOwned,
          Y: serde::de::DeserializeOwned,
          Y: ToPrimitive,
    {
        let (dx, dy) = self.source_data
            .clone_split()
            .expect("Supervised Regression requires a target column");

        let x: Vec<X> = arff::dynamic::de::from_dataset(&dx).unwrap();
        let y: Vec<Y> = arff::dynamic::de::from_dataset(&dy).unwrap();

        let mut measure = self.evaluation_measures.create();

        for fold in self.estimation_procedure.iter() {
            let mut train = fold.trainset
                .iter()
                .map(|&i| (&x[i * dx.n_cols() .. (i+1) * dx.n_cols()], &y[i]));

            let mut test = fold.testset
                .iter()
                .map(|&i| &x[i * dx.n_cols() .. (i+1) * dx.n_cols()]);

            let predictit = flow(&mut train, &mut test);

            for (known, pred) in fold.testset.iter().map(|&i| &y[i]).zip(predictit) {
                measure.update_one(known, &pred);
            }
        }

        measure
    }
}


#[derive(Debug)]
struct SupervisedClassification {
    source_data: DataSet,
    estimation_procedure: Procedure,
    cost_matrix: CostMatrix,
    evaluation_measures: Measure,
}

impl SupervisedClassification {
    fn new(input_json: &Vec<serde_json::Value>) -> Self {
        let mut source_data = None;
        let mut estimation_procedure = None;
        let mut cost_matrix = None;
        let mut evaluation_measures = Measure::PredictiveAccuracy;  // default

        for input_item in input_json {
            match input_item["name"].as_str() {
                Some("source_data") => source_data = Some(input_item.into()),
                Some("estimation_procedure") => estimation_procedure = Some(input_item.into()),
                Some("cost_matrix") => cost_matrix = Some(input_item.into()),
                Some("evaluation_measures") => evaluation_measures = Measure::new(input_item).unwrap_or(evaluation_measures),
                Some(_) => {}
                None => panic!("/task/input/name is not a string")
            }
        }

        SupervisedClassification {
            source_data: source_data.unwrap(),
            estimation_procedure: estimation_procedure.unwrap(),
            cost_matrix: cost_matrix.unwrap(),
            evaluation_measures: evaluation_measures,
        }
    }

    fn run_static<X, Y, F>(&self, task: &Task, flow: F) -> MeasureAccumulator
    where F: Fn(&mut Iterator<Item=(&X, &Y)>, &mut Iterator<Item=&X>) -> Box<Iterator<Item=Y>>,
          X: serde::de::DeserializeOwned,
          Y: serde::de::DeserializeOwned,
          Y: ToPrimitive,
    {
        let (dx, dy) = self.source_data
            .clone_split()
            .expect("Supervised Classification requires a target column");

        let x: Vec<X> = arff::dynamic::de::from_dataset(&dx).unwrap();
        let y: Vec<Y> = arff::dynamic::de::from_dataset(&dy).unwrap();

        let mut measure = self.evaluation_measures.create();

        for fold in self.estimation_procedure.iter() {
            let mut train = fold.trainset
                .iter()
                .map(|&i| (&x[i], &y[i]));

            let mut test = fold.testset
                .iter()
                .map(|&i| &x[i]);

            let predictit = flow(&mut train, &mut test);

            for (known, pred) in fold.testset.iter().map(|&i| &y[i]).zip(predictit) {
                measure.update_one(known, &pred);
            }
        }

        measure
    }

    fn run<X, Y, F>(&self, task: &Task, flow: F) -> MeasureAccumulator
    where F: Fn(&mut Iterator<Item=(&[X], &Y)>, &mut Iterator<Item=&[X]>) -> Box<Iterator<Item=Y>>,
          X: serde::de::DeserializeOwned,
          Y: serde::de::DeserializeOwned,
          Y: ToPrimitive,
    {
        let (dx, dy) = self.source_data
            .clone_split()
            .expect("Supervised Classification requires a target column");

        let x: Vec<X> = arff::dynamic::de::from_dataset(&dx).unwrap();
        let y: Vec<Y> = arff::dynamic::de::from_dataset(&dy).unwrap();

        let mut measure = self.evaluation_measures.create();

        for fold in self.estimation_procedure.iter() {
            let mut train = fold.trainset
                .iter()
                .map(|&i| (&x[i * dx.n_cols() .. (i+1) * dx.n_cols()], &y[i]));

            let mut test = fold.testset
                .iter()
                .map(|&i| &x[i * dx.n_cols() .. (i+1) * dx.n_cols()]);

            let predictit = flow(&mut train, &mut test);

            for (known, pred) in fold.testset.iter().map(|&i| &y[i]).zip(predictit) {
                measure.update_one(known, &pred);
            }
        }

        measure
    }
}

#[derive(Debug)]
struct DataSet {
    arff: arff::dynamic::DataSet,
    target: Option<String>,
}

impl DataSet {
    fn clone_split(&self) -> Option<(arff::dynamic::DataSet, arff::dynamic::DataSet)> {
        match self.target {
            None => None,
            Some(ref col) => {
                let data = self.arff.clone();
                Some(data.split_one(col))
            }
        }
    }
}

impl<'a> From<&'a serde_json::Value> for DataSet
{
    fn from(item: &serde_json::Value) -> Self {
        let v = &item["data_set"];
        let id = v["data_set_id"].as_str().unwrap();
        let target = v["target_feature"].as_str();

        let info_url = format!("https://www.openml.org/api/v1/json/data/{}", id.as_string());
        let info: GenericResponse =  serde_json::from_str(&get_cached(&info_url).unwrap()).unwrap();

        let default_target = info
            .look_up("/data_set_description/default_target_attribute")
            .and_then(|v| v.as_str());

        let target = match (default_target, target) {
            (Some(s), None) |
            (_, Some(s)) => Some(s.to_owned()),
            (None, None) => None,
        };

        let dset_url = info.look_up("/data_set_description/url").unwrap().as_str().unwrap();
        let dset_str = get_cached(&dset_url).unwrap();
        let dset = arff::dynamic::DataSet::from_str(&dset_str).unwrap();

        DataSet {
            arff: dset,
            target,
        }
    }
}

#[derive(Debug, Clone)]
struct CrossValidationFold {
    trainset: Vec<usize>,
    testset: Vec<usize>,
}

impl CrossValidationFold {
    fn new() -> Self {
        CrossValidationFold {
            trainset: Vec::new(),
            testset: Vec::new(),
        }
    }
}

#[derive(Debug)]
struct CrossValidation {
    folds: Vec<Vec<CrossValidationFold>>
}

impl From<CrossValSplits> for CrossValidation {
    fn from(xvs: CrossValSplits) -> Self {
        let mut folds = vec![];
        for item in xvs.data {
            if item.repeat >= folds.len() {
                folds.resize(item.repeat + 1, vec![]);
            }
            let mut rep = &mut folds[item.repeat];

            if item.fold >= rep.len() {
                rep.resize(item.fold + 1, CrossValidationFold::new());
            }
            let mut fold = &mut rep[item.fold];

            match item.purpose {
                TrainTest::Train => fold.trainset.push(item.rowid),
                TrainTest::Test => fold.testset.push(item.rowid),
            }
        }

        CrossValidation {
            folds
        }
    }
}

#[derive(Debug)]
enum Procedure {
    Frozen(CrossValidation),
}

impl Procedure {
    fn iter(&self) -> impl Iterator<Item=&CrossValidationFold> {
        match *self {
            Procedure::Frozen(ref xv) => {
                xv.folds.iter().flat_map(|inner| inner.iter())
            }
        }
    }
}

impl<'a> From<&'a serde_json::Value> for Procedure {
    fn from(item: &serde_json::Value) -> Self {
        let v = &item["estimation_procedure"];
        let typ = v["type"].as_str();
        let splits = v["data_splits_url"].as_str();
        match (typ, splits) {
            (_, Some(url)) => {
                Procedure::Frozen(CrossValSplits::load(url).unwrap().into())
            },
            _ => unimplemented!(),
        }
    }
}

#[derive(Debug)]
struct CrossValSplits {
    data: Vec<CrossValItem>,
}

impl CrossValSplits {
    fn load(url: &str) -> Result<Self> {
        let raw = get_cached(url)?;
        let data = arff::from_str(&raw)?;
        Ok(CrossValSplits {
            data
        })
    }
}

#[derive(Debug, Deserialize)]
struct CrossValItem {
    #[serde(rename = "type")]
    purpose: TrainTest,

    rowid: usize,

    repeat: usize,

    fold: usize,
}

#[derive(Debug, Deserialize)]
enum TrainTest {
    #[serde(rename = "TRAIN")]
    Train,

    #[serde(rename = "TEST")]
    Test,
}

#[derive(Debug)]
enum CostMatrix {
    None,
}

impl<'a> From<&'a serde_json::Value> for CostMatrix {
    fn from(item: &serde_json::Value) -> Self {
        let v = &item["cost_matrix"];
        match v.as_array() {
            None => panic!("invalid cots matrix"),
            Some(c) if c.is_empty() => CostMatrix::None,
            Some(_) => unimplemented!("cost matrix"),
        }
    }
}

#[derive(Debug)]
enum Measure {
    PredictiveAccuracy,
    RootMeanSquaredError,
}

impl Measure {
    fn new(item: &serde_json::Value) -> Option<Self> {
        let measure = item.pointer("/evaluation_measures/evaluation_measure").unwrap();
        match *measure {
            serde_json::Value::String(ref s) if s == "predictive_accuracy" => Some(Measure::PredictiveAccuracy),
            serde_json::Value::String(ref s) if s == "root_mean_squared_error" => Some(Measure::RootMeanSquaredError),
            serde_json::Value::Array(ref v) if v.is_empty() => None,
            _ => panic!("Invalid evaluation measure: {:?}", measure),
        }
    }

    fn create(&self) -> MeasureAccumulator {
        match *self {
            Measure::PredictiveAccuracy => MeasureAccumulator::new_accuracy(),
            Measure::RootMeanSquaredError =>  MeasureAccumulator::new_rmse(),
        }
    }
}

#[derive(Debug)]
pub enum MeasureAccumulator {
    Accuracy {
        n_correct: usize,
        n_wrong: usize,
    },

    RootMeanSquaredError {
        sum_of_squares: f64,
        n: usize,
    }
}

impl MeasureAccumulator {
    fn new_accuracy() -> Self { MeasureAccumulator::Accuracy { n_correct: 0, n_wrong: 0 } }

    fn new_rmse() -> Self { MeasureAccumulator::RootMeanSquaredError { sum_of_squares: 0.0, n: 0 } }

    fn update(&mut self, known: &[f64], predicted: &[f64]) {
        for (k, p) in known.iter().zip(predicted.iter()) {
            self.update_one(k, p);
        }
    }

    fn update_one<T: ToPrimitive>(&mut self, known: &T, pred: &T) {
        match *self {
            MeasureAccumulator::Accuracy{ref mut n_correct, ref mut n_wrong} => {
                if known.to_i64().unwrap() == pred.to_i64().unwrap() {
                    *n_correct += 1;
                } else {
                    *n_wrong += 1;
                }
            }
            MeasureAccumulator::RootMeanSquaredError{ref mut sum_of_squares, ref mut n} => {
                let diff = known.to_f64().unwrap() - pred.to_f64().unwrap();
                *sum_of_squares += diff * diff;
                *n += 1;
            }
        }
    }

    fn result(&self) -> f64{
        match *self {
            MeasureAccumulator::Accuracy{ref n_correct, ref n_wrong} => {
                *n_correct as f64 / (*n_correct as f64 + *n_wrong as f64)
            }
            MeasureAccumulator::RootMeanSquaredError{ref sum_of_squares, ref n} => {
                (*sum_of_squares / *n as f64).sqrt()
            }
        }
    }
}


fn get_cached(url: &str) -> Result<String> {
    // todo: is there a potential race condition with a process locking the file for reading while
    //       the writer has created but not yet locked the file?
    let filename = "cache/".to_owned() + &url_to_file(url);
    let path = Path::new(&filename);

    loop {

        match fs::File::open(path) {
            Ok(mut f) => {
                info!("Loading cached {}", url);
                let mut file = SharedLock::new(f)?;
                let mut data = String::new();
                file.read_to_string(&mut data)?;
                return Ok(data)
            }
            Err(e) => {}
        }

        match fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(path)
            {
                Err(e) => {
                    // todo: is this the correct io error raised if another thread has locked the file currently?
                    if let std::io::ErrorKind::PermissionDenied = e.kind() {
                        continue
                    }
                    error!("Error while opening cache for writing: {:?}", e);
                    return Err(e.into())
                },
                Ok(mut f) => {
                    info!("Downloading {}", url);
                    let mut file = ExclusiveLock::new(f)?;
                    let data = download(url)?;
                    file.write_all(data.as_bytes())?;
                    return Ok(data)
                }
            }
    }
}

fn download(url: &str) -> Result<String> {
    let mut core = Core::new()?;
    let handle = core.handle();
    let client = hyper::Client::configure()
        .connector(HttpsConnector::new(4, &handle)?)
        .build(&handle);

    let req = client.get(url.parse()?);

    let mut bytes = Vec::new();
    {
        let work = req.and_then(|res| {
            res.body().for_each(|chunk| {
                bytes.extend_from_slice(&chunk);
                Ok(())
            })
        });
        core.run(work)?
    }
    let result = String::from_utf8(bytes)?;
    Ok(result)
}

fn url_to_file(s: &str) -> String {
    s.replace('/', "_").replace(':', "")
}

struct ExclusiveLock {
    file: fs::File
}

impl ExclusiveLock {
    fn new(file: fs::File) -> Result<Self> {
        file.lock_exclusive()?;
        Ok(ExclusiveLock {
            file
        })
    }
}

impl Drop for ExclusiveLock {
    fn drop(&mut self) {
        self.file.unlock().unwrap();
    }
}

impl Read for ExclusiveLock {
    #[inline(always)]
    fn read(&mut self, data: &mut [u8]) -> std::io::Result<usize> {
        self.file.read(data)
    }
}

impl Write for ExclusiveLock {
    #[inline(always)]
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        self.file.write(data)
    }

    #[inline(always)]
    fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }
}

struct SharedLock {
    file: fs::File
}

impl SharedLock {
    fn new(file: fs::File) -> Result<Self> {
        file.lock_shared()?;
        Ok(SharedLock {
            file
        })
    }
}

impl Drop for SharedLock {
    fn drop(&mut self) {
        self.file.unlock().unwrap();
    }
}

impl Read for SharedLock {
    #[inline(always)]
    fn read(&mut self, data: &mut [u8]) -> std::io::Result<usize> {
        self.file.read(data)
    }
}

impl Write for SharedLock {
    #[inline(always)]
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        self.file.write(data)
    }

    #[inline(always)]
    fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }
}


#[test]
fn apidev() {
    let mut api = OpenML::new();
    let task = api.task(166850).unwrap();

    println!("{:#?}", task);

    let result = task.run_static(|train, test| {
        let y_out: Vec<_> = test.map(|row: &[f64; 4]| 0).collect();
        Box::new(y_out.into_iter())
    });

    println!("{:#?}", result);

    #[derive(Deserialize)]
    struct Row {
        sepallength: f32,
        sepalwidth: f32,
        petallength: f32,
        petalwidth: f32,
    }

    let result = task.run_static(|train, test| {
        let (x_train, y_train): (Vec<&Row>, Vec<i32>) = train.unzip();
        let y_out: Vec<_> = test.map(|row: &Row| 0).collect();
        Box::new(y_out.into_iter())
    });

    println!("{:#?}", result);

    let result = task.run(|train, test| {
        let y_out: Vec<_> = test.map(|row: &[f64]| 0).collect();
        Box::new(y_out.into_iter())
    });

    println!("{:#?}", result);
}


#[test]
fn apidev2() {
    use simple_logger;
    simple_logger::init_with_level(Level::Info).unwrap();
    let mut api = OpenML::new();

    let start = PreciseTime::now();

    let task = api.task(146825).unwrap();
    //let task = api.task(167147).unwrap();

    let end = PreciseTime::now();

    let result = task.run(|train, test| {
        let y_out: Vec<_> = test.map(|row: &[u8]| 0).collect();
        Box::new(y_out.into_iter())
    });

    println!("{:#?}", result);

    println!("loading took {} seconds.", start.to(end));
}
