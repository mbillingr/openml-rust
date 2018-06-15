use serde_json;

#[derive(Debug, Serialize, Deserialize)]
pub struct GenericResponse(serde_json::Value);

impl GenericResponse {
    #[inline(always)]
    pub fn look_up<'a>(&'a self, p: &str) -> Option<&'a serde_json::Value> {
        self.0.pointer(p)
    }
}

#[derive(Debug, Deserialize)]
pub struct CrossValItem {
    #[serde(rename = "type")]
    pub purpose: TrainTest,

    pub rowid: usize,

    pub repeat: usize,

    pub fold: usize,
}

#[derive(Debug, Deserialize)]
pub enum TrainTest {
    #[serde(rename = "TRAIN")]
    Train,

    #[serde(rename = "TEST")]
    Test,
}

#[derive(Debug)]
pub(crate) enum CostMatrix {
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
