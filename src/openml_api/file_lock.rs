//! file locking mechanisms

use std::fs::File;
use std::io::{self, Read, Write};

use fs2::FileExt;

/// A scoped exclusive lock for use by file writers
pub struct ExclusiveLock {
    file: File,
}

impl ExclusiveLock {
    /// acquire locked file
    pub fn new(file: File) -> io::Result<Self> {
        file.lock_exclusive()?;
        Ok(ExclusiveLock { file })
    }
}

impl Drop for ExclusiveLock {
    /// release locked file
    fn drop(&mut self) {
        self.file.unlock().unwrap();
    }
}

impl Read for ExclusiveLock {
    /// read from locked file
    #[inline(always)]
    fn read(&mut self, data: &mut [u8]) -> io::Result<usize> {
        self.file.read(data)
    }
}

impl Write for ExclusiveLock {
    /// write to locked file
    #[inline(always)]
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        self.file.write(data)
    }

    /// flush buffer of locked file
    #[inline(always)]
    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

pub struct SharedLock {
    file: File,
}

/// A scoped shared lock for use by file readers
impl SharedLock {
    /// acquire locked file
    pub fn new(file: File) -> io::Result<Self> {
        file.lock_shared()?;
        Ok(SharedLock { file })
    }
}

impl Drop for SharedLock {
    /// release locked file
    fn drop(&mut self) {
        self.file.unlock().unwrap();
    }
}

impl Read for SharedLock {
    /// read from locked file
    #[inline(always)]
    fn read(&mut self, data: &mut [u8]) -> io::Result<usize> {
        self.file.read(data)
    }
}
