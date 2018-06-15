mod api_types;
mod file_lock;
mod impls_from_json;
mod impls_from_openml;
mod web_access;

use std::borrow::Cow;

pub trait Id {
    fn as_string(&self) -> Cow<str>;
    fn as_u32(&self) -> u32;
}

impl Id for String {
    #[inline(always)]
    fn as_string(&self) -> Cow<str> {
        Cow::from(self.as_ref())
    }

    #[inline(always)]
    fn as_u32(&self) -> u32 {
        self.parse().unwrap()
    }
}

impl<'a> Id for &'a str {
    #[inline(always)]
    fn as_string(&self) -> Cow<str> {
        Cow::from(*self)
    }

    #[inline(always)]
    fn as_u32(&self) -> u32 {
        self.parse().unwrap()
    }
}

impl Id for u32 {
    #[inline(always)]
    fn as_string(&self) -> Cow<str> {
        Cow::from(format!("{}", self))
    }

    #[inline(always)]
    fn as_u32(&self) -> u32 {
        *self
    }
}
