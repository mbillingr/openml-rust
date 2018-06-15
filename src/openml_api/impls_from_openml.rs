use serde_json;

use error::Result;
use tasks::SupervisedClassification;

use super::Id;
use super::api_types::GenericResponse;
use super::web_access::get_cached;

impl SupervisedClassification {
    pub fn from_openml<'a, T: Id>(id: T) -> Result<Self> {
        let url = format!("https://www.openml.org/api/v1/json/task/{}", id.as_string());
        let raw_task = get_cached(&url)?;
        let response: GenericResponse = serde_json::from_str(&raw_task)?;

        let task = response.look_up("/task").unwrap();

        match response.look_up("/task/task_type_id").unwrap().as_str() {
            Some("1") => Ok(SupervisedClassification::from_json(task)),
            _ => panic!(),
        }
    }
}
