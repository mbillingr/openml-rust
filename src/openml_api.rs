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
            source_data: (&inputs["source_data"]).into(),
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
    arff: arff::DataSet,
}

impl<'a> From<&'a serde_json::Value> for DataSet {
    fn from(v: &serde_json::Value) -> Self {
        let id = v["data_set_id"].as_str().unwrap();
        let target = v["target_feature"].as_str();

        let info_url = format!("https://www.openml.org/api/v1/json/data/{}", id.as_string());
        let info: GenericResponse =  serde_json::from_str(&get_cached(&info_url).unwrap()).unwrap();

        let dset_url = info.look_up("/data_set_description/url").unwrap().as_str().unwrap();
        let dset = arff::DataSet::from_str(&get_cached(&dset_url).unwrap()).unwrap();

        DataSet {
            arff: dset
        }
    }
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
            (_, Some(url)) => Procedure::Frozen(CrossValSplits::load(url).unwrap()),
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
    // todo: is there a potential race condition with a process locking the file for reading while
    //       the writer has created but not yet locked the file?
    let filename = "cache/".to_owned() + &url_to_file(url);
    let path = Path::new(&filename);

    loop {
        match fs::File::open(path) {
            Ok(mut f) => {
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
                Err(_) => continue,
                Ok(mut f) => {
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
    println!("{:?}", api.get_task(166850).unwrap());
}
