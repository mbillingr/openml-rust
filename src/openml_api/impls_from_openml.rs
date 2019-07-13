//! implementations to load tasks from the OpenML API.
use serde_json;

use crate::error::Result;
use crate::tasks::{SupervisedClassification, SupervisedRegression};

use super::api_types::GenericResponse;
use super::web_access::get_cached;
use super::Id;

impl SupervisedClassification {
    pub fn from_openml<'a, T: Id>(id: T) -> Result<Self> {
        let url = format!("https://www.openml.org/api/v1/json/task/{}", id.as_string());
        let raw_task = get_cached(&url)?;
        let response: GenericResponse = serde_json::from_str(&raw_task)?;

        let task = response.look_up("/task").unwrap();

        match response.look_up("/task/task_type_id").unwrap().as_str() {
            Some("1") => Ok(SupervisedClassification::from_json(task)),
            Some(id) => panic!("Wrong task type ID. Expected \"1\" but got \"{}\"", id),
            None => panic!("Invalid task type ID"),
        }
    }
}

impl SupervisedRegression {
    pub fn from_openml<'a, T: Id>(id: T) -> Result<Self> {
        let url = format!("https://www.openml.org/api/v1/json/task/{}", id.as_string());
        let raw_task = get_cached(&url)?;
        let response: GenericResponse = serde_json::from_str(&raw_task)?;

        let task = response.look_up("/task").unwrap();

        match response.look_up("/task/task_type_id").unwrap().as_str() {
            Some("2") => Ok(SupervisedRegression::from_json(task)),
            Some(id) => panic!("Wrong task type ID. Expected \"2\" but got \"{}\"", id),
            None => panic!("Invalid task type ID"),
        }
    }
}
