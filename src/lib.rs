extern crate arff;
extern crate fs2;
extern crate futures;
extern crate hyper;
extern crate hyper_tls;
#[macro_use]
extern crate log;
extern crate ndarray;
extern crate serde;
#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate serde_json;
#[cfg(test)]
extern crate simple_logger;
extern crate time;
extern crate tokio_core;

mod openml_api;

pub use arff::{Array, ArrayCastInto, ArrayCastFrom};

pub use openml_api::OpenML;


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
