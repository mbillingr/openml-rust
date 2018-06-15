/*use serde::de::DeserializeOwned;

use measure_accumulator::MeasureAccumulator;
use openml_api::TaskType;*/
/*
pub struct Task {
    task_id: String,
    task_name: String,
    task_type: TaskType,
}

impl Task {
    pub(crate) fn new(task_id: String, task_name: String, task_type: TaskType) -> Self {
        Task {
            task_id,
            task_name,
            task_type,
        }
    }

    pub fn id(&self) -> &str {
        &self.task_id
    }

    pub fn name(&self) -> &str {
        &self.task_name
    }

    /// run task with statically known row size and column types
    pub fn run_static<X, Y, F, M>(&self, flow: F) -> M
    where
        F: Fn(&mut Iterator<Item = (&X, &Y)>, &mut Iterator<Item = &X>) -> Box<Iterator<Item = Y>>,
        X: DeserializeOwned,
        Y: DeserializeOwned,
        M: MeasureAccumulator<Y>,
    {
        self.task_type.run_static(flow)
    }

    /// run task with unknown row size
    pub fn run<X, Y, F, M>(&self, flow: F) -> M
    where
        F: Fn(&mut Iterator<Item = (&[X], &Y)>, &mut Iterator<Item = &[X]>)
            -> Box<Iterator<Item = Y>>,
        X: DeserializeOwned,
        Y: DeserializeOwned,
        M: MeasureAccumulator<Y>,
    {
        self.task_type.run(flow)
    }
}
*/
