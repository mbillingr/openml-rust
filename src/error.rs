use std::io::Error as IoError;
use std::result::Result as StdResult;
use std::string::FromUtf8Error;

use app_dirs::AppDirsError;
use arff::Error as ArffError;
use hyper::Error as HyperError;
use hyper::error::UriError;
use hyper_tls::Error as TlsError;
use serde_json::Error as JsonError;

pub type Result<T> = StdResult<T, Error>;

#[derive(Debug)]
pub enum Error {
    IoError(IoError),
    Utf8Error(FromUtf8Error),
    HyperError(HyperError),
    HyperUriError(UriError),
    HyperTlsError(TlsError),
    JsonError(JsonError),
    ArffError(ArffError),
    AppDirsError(AppDirsError),
}

impl From<IoError> for Error {
    fn from(e: IoError) -> Self {
        Error::IoError(e)
    }
}

impl From<FromUtf8Error> for Error {
    fn from(e: FromUtf8Error) -> Self {
        Error::Utf8Error(e)
    }
}

impl From<HyperError> for Error {
    fn from(e: HyperError) -> Self {
        Error::HyperError(e)
    }
}

impl From<UriError> for Error {
    fn from(e: UriError) -> Self {
        Error::HyperUriError(e)
    }
}

impl From<TlsError> for Error {
    fn from(e: TlsError) -> Self {
        Error::HyperTlsError(e)
    }
}

impl From<JsonError> for Error {
    fn from(e: JsonError) -> Self {
        Error::JsonError(e)
    }
}

impl From<ArffError> for Error {
    fn from(e: ArffError) -> Self {
        Error::ArffError(e)
    }
}

impl From<AppDirsError> for Error {
    fn from(e: AppDirsError) -> Self {
        match e {
            AppDirsError::Io(e) => Error::IoError(e),
            _ => Error::AppDirsError(e)
        }
    }
}
