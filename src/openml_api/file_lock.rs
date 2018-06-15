use std::fs::File;
use std::io::{self, Read, Write};

use fs2::FileExt;

pub struct ExclusiveLock {
    file: File,
}

impl ExclusiveLock {
    pub fn new(file: File) -> io::Result<Self> {
        file.lock_exclusive()?;
        Ok(ExclusiveLock { file })
    }
}

impl Drop for ExclusiveLock {
    fn drop(&mut self) {
        self.file.unlock().unwrap();
    }
}

impl Read for ExclusiveLock {
    #[inline(always)]
    fn read(&mut self, data: &mut [u8]) -> io::Result<usize> {
        self.file.read(data)
    }
}

impl Write for ExclusiveLock {
    #[inline(always)]
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        self.file.write(data)
    }

    #[inline(always)]
    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

pub struct SharedLock {
    file: File,
}

impl SharedLock {
    pub fn new(file: File) -> io::Result<Self> {
        file.lock_shared()?;
        Ok(SharedLock { file })
    }
}

impl Drop for SharedLock {
    fn drop(&mut self) {
        self.file.unlock().unwrap();
    }
}

impl Read for SharedLock {
    #[inline(always)]
    fn read(&mut self, data: &mut [u8]) -> io::Result<usize> {
        self.file.read(data)
    }
}

impl Write for SharedLock {
    #[inline(always)]
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        self.file.write(data)
    }

    #[inline(always)]
    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}
