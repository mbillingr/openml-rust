//! Access the OpenML REST API

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};

use app_dirs::{app_root, AppDataType, AppInfo};

use error::Result;

use super::file_lock::{ExclusiveLock, SharedLock};

const APP_INFO: AppInfo = AppInfo {
    name: "openml-rust",
    author: "openml-rust",
};

/// Query a URL. If possible read the response from local cache
pub fn get_cached(url: &str) -> Result<String> {
    // todo: is there a potential race condition with a process locking the file for reading while
    //       the writer has created but not yet locked the file?

    let mut path = app_root(AppDataType::UserCache, &APP_INFO)?;
    path.push(url_to_file(url));

    loop {
        match File::open(&path) {
            Ok(f) => {
                info!("Loading cached {}", url);
                let mut file = SharedLock::new(f)?;
                let mut data = String::new();
                file.read_to_string(&mut data)?;
                return Ok(data);
            }
            Err(_) => {}
        }

        match OpenOptions::new().create_new(true).write(true).open(&path) {
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

/// Query a URL.
fn download(url: &str) -> Result<String> {
    Ok(reqwest::get(url)?.text()?)
}

/// Convert URL to file name for chching
fn url_to_file(s: &str) -> String {
    s.replace('/', "_").replace(':', "")
}
