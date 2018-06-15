use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::Path;

use futures::{Future, Stream};
use hyper::Client;
use hyper_tls::HttpsConnector;
use tokio_core::reactor::Core;

use error::Result;

use super::file_lock::{ExclusiveLock, SharedLock};

pub fn get_cached(url: &str) -> Result<String> {
    // todo: is there a potential race condition with a process locking the file for reading while
    //       the writer has created but not yet locked the file?
    let filename = "cache/".to_owned() + &url_to_file(url);
    let path = Path::new(&filename);

    loop {
        match File::open(path) {
            Ok(f) => {
                info!("Loading cached {}", url);
                let mut file = SharedLock::new(f)?;
                let mut data = String::new();
                file.read_to_string(&mut data)?;
                return Ok(data);
            }
            Err(_) => {}
        }

        match OpenOptions::new().create_new(true).write(true).open(path) {
            Err(e) => {
                // todo: is this the correct io error raised if another thread has locked the file currently?
                if let io::ErrorKind::PermissionDenied = e.kind() {
                    continue;
                }
                error!("Error while opening cache for writing: {:?}", e);
                return Err(e.into());
            }
            Ok(f) => {
                info!("Downloading {}", url);
                let mut file = ExclusiveLock::new(f)?;
                let data = download(url)?;
                file.write_all(data.as_bytes())?;
                return Ok(data);
            }
        }
    }
}

fn download(url: &str) -> Result<String> {
    let mut core = Core::new()?;
    let handle = core.handle();
    let client = Client::configure()
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
