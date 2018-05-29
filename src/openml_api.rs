use std;
use std::borrow::Cow;
use std::fs;
use std::io::{Read, Write};
use std::ops::Index;
use std::result;
use std::string;
use std::path::Path;

use arff;
use futures::{Future, Stream};
use hyper;
use hyper_tls::{self, HttpsConnector};
use serde_json;
use tokio_core::reactor::Core;

type Result<T> = result::Result<T, Error>;

#[derive(Debug)]
enum Error {
    IoError(std::io::Error),
    Utf8Error(string::FromUtf8Error),
    HyperError(hyper::Error),
    HyperUriError(hyper::error::UriError),
    HyperTlsError(hyper_tls::Error),
    JsonError(serde_json::Error),
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


struct OpenML {
}

impl OpenML {
    pub fn new() -> Self {
        OpenML {}
    }

    pub fn get_task<T: Id>(&mut self, id: T) -> Result<Task> {
        let url = format!("https://www.openml.org/api/v1/json/task/{}", id.as_string());
        let raw_task = get_cached(&url)?;
        let response: GenericResponse = serde_json::from_str(&raw_task)?;

        let task = response.look_up("/task").unwrap();

        // bring input array into a form that can be looked up by name
        let mut inputs = serde_json::Map::new();
        for input in task["input"].as_array().unwrap() {
            match input["name"].as_str() {
                None => panic!("Input missing `name` field"),
                Some("source_data") => {
                    inputs.insert(String::from("source_data"), input["data_set"].clone());
                }
                Some(name) => {
                    inputs.insert(String::from(name), input[name].clone());
                }
            }
        }

        Ok(Task {
            task_id: task["task_id"].as_str().unwrap().to_owned(),
            task_name: task["task_name"].as_str().unwrap().to_owned(),
            task_type: TaskType {},
            source_data: DataSet {},
            estimation_procedure: (&inputs["estimation_procedure"]).into(),
            cost_matrix: (&inputs["cost_matrix"]).into(),
            evaluation_measures: (&inputs["evaluation_measures"]).into(),
        })
    }
}


trait Id {
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
struct Task {
    task_id: String,
    task_name: String,
    task_type: TaskType,
    source_data: DataSet,
    estimation_procedure: Procedure,
    cost_matrix: CostMatrix,
    evaluation_measures: Measure,
}



#[derive(Debug)]
struct TaskType {

}

#[derive(Debug)]
struct DataSet {

}

#[derive(Debug)]
enum Procedure {
    Frozen(CrossValSplits),
}

impl<'a> From<&'a serde_json::Value> for Procedure {
    fn from(v: &serde_json::Value) -> Self {
        let typ = v["type"].as_str();
        let splits = v["data_splits_url"].as_str();
        match (typ, splits) {
            (_, Some(url)) => Procedure::Frozen(CrossValSplits::load(url)),
            _ => unimplemented!(),
        }
    }
}

#[derive(Debug)]
struct CrossValSplits {
    data: Vec<CrossValItem>,
}

impl CrossValSplits {
    fn load(url: &str) -> Self {
        let raw = get_cached(url).unwrap();
        unimplemented!()
    }
}

#[derive(Debug)]
struct CrossValItem {
    purpose: TrainTest,
    rowid: usize,
    repeat: usize,
    fold: usize,
}

#[derive(Debug)]
enum TrainTest {
    Train,
    Test,
}

#[derive(Debug)]
enum CostMatrix {
    None,
}

impl<'a> From<&'a serde_json::Value> for CostMatrix {
    fn from(v: &serde_json::Value) -> Self {
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
}

impl<'a> From<&'a serde_json::Value> for Measure {
    fn from(v: &serde_json::Value) -> Self {
        match v["evaluation_measure"].as_str() {
            Some("predictive_accuracy") => Measure::PredictiveAccuracy,
            _ => panic!("Invalid evaluation measure: {:?}", v),
        }
    }
}

fn get_cached(url: &str) -> Result<String> {
    // todo: this really should use some file locking mechanisms...
    let filename = "cache/".to_owned() + &url_to_file(url);
    let path = Path::new(&filename);
    if path.exists() {
        load(&path)
    } else {
        let data = download(url)?;
        store(&path, &data)?;
        Ok(data)
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

fn load(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut result = String::new();
    file.read_to_string(&mut result)?;
    Ok(result)
}

fn store(path: &Path, content: &str) -> Result<()> {
    let mut file = fs::File::create(path)?;
    file.write_all(content.as_bytes())?;
    Ok(())
}

fn url_to_file(s: &str) -> String {
    s.replace('/', "_").replace(':', "")
}


#[test]
fn apidev() {
    let mut api = OpenML::new();
    println!("{:?}", api.get_task(166850).unwrap());
}
