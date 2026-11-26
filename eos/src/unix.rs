/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use std::{
    fs::{File, OpenOptions},
    io,
    os::{fd::AsRawFd, unix::fs::OpenOptionsExt},
    path::Path,
};

/// # Safety
///
/// `address` must be valid for writes of `bytes` bytes for the call.
pub unsafe fn read_at_raw(
    file: &File,
    address: *mut u8,
    bytes: usize,
    offset: u64,
) -> io::Result<usize> {
    let offset = libc::off_t::try_from(offset).map_err(|_| io::ErrorKind::InvalidInput)?;
    loop {
        let count = unsafe { libc::pread(file.as_raw_fd(), address.cast(), bytes, offset) };
        if count >= 0 {
            return Ok(count as usize);
        }
        let error = io::Error::last_os_error();
        if error.kind() != io::ErrorKind::Interrupted {
            return Err(error);
        }
    }
}

pub fn direct_file(path: &Path, write: bool) -> io::Result<File> {
    OpenOptions::new()
        .read(!write)
        .write(write)
        .create(write)
        .truncate(write)
        .mode(0o600)
        .custom_flags(libc::O_DIRECT)
        .open(path)
}

/// Open without following symlinks, preserving existing bytes when writing.
pub fn direct_update_file(path: &Path, write: bool) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options
        .read(true)
        .write(write)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_DIRECT);
    if write {
        options.create(true).truncate(false);
    }
    options.open(path)
}

pub fn direct_reopen(file: &File) -> io::Result<File> {
    direct_file(
        Path::new(&format!("/proc/self/fd/{}", file.as_raw_fd())),
        false,
    )
}

pub fn release_page_cache(file: &File) {
    unsafe {
        libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
    }
}

pub fn page_size() -> usize {
    usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).unwrap_or(4096)
}

pub fn lower_priority(nice: i32) {
    unsafe {
        let param = std::mem::zeroed();
        libc::sched_setscheduler(0, libc::SCHED_BATCH, &param);
        libc::setpriority(libc::PRIO_PROCESS, 0, nice);
    }
}

pub fn map_host(bytes: usize) -> io::Result<usize> {
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            bytes,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_POPULATE,
            -1,
            0,
        )
    };
    if ptr == libc::MAP_FAILED {
        return Err(io::Error::last_os_error());
    }
    unsafe {
        libc::madvise(ptr, bytes, libc::MADV_HUGEPAGE);
    }
    Ok(ptr as usize)
}

/// # Safety
///
/// `address` and `bytes` must describe a mapping made by `map_host` that nothing else
/// references any longer, such as one the driver has already unregistered.
pub unsafe fn unmap_host(address: usize, bytes: usize) {
    unsafe {
        libc::munmap(address as *mut _, bytes);
    }
}

/// Free bytes on the filesystem holding `path`.
pub fn free_space(path: &Path) -> io::Result<u64> {
    let text = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "record directory path contains a NUL byte",
        )
    })?;
    let mut stats: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: Both pointers are valid for the call; statvfs fills the struct.
    if unsafe { libc::statvfs(text.as_ptr(), &mut stats) } != 0 {
        return Err(io::Error::other(format!(
            "cannot inspect the filesystem of {}: {}",
            path.display(),
            io::Error::last_os_error()
        )));
    }
    Ok(stats.f_bavail * stats.f_frsize)
}

/// Raise the process soft open-file limit (`RLIMIT_NOFILE`) toward `desired`,
/// bounded by the current hard limit. Never lowers an existing higher soft
/// limit. Returns the resulting soft limit when `getrlimit` succeeds.
pub fn raise_open_file_limit(desired: u64) -> Option<u64> {
    let mut limit: libc::rlimit = unsafe { std::mem::zeroed() };
    // SAFETY: `getrlimit` writes into the provided `rlimit` struct.
    if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) } != 0 {
        return None;
    }
    let current = limit.rlim_cur as u64;
    if current >= desired {
        return Some(current);
    }
    let max = if limit.rlim_max == libc::RLIM_INFINITY {
        desired
    } else {
        limit.rlim_max as u64
    };
    let target = desired.min(max).min(libc::rlim_t::MAX as u64);
    if target > current {
        let updated = libc::rlimit {
            rlim_cur: target as libc::rlim_t,
            rlim_max: limit.rlim_max,
        };
        // SAFETY: `setrlimit` reads the provided `rlimit` struct.
        if unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &updated) } == 0 {
            return Some(target);
        }
    }
    Some(current)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raise_open_file_limit_never_lowers_and_respects_hard_limit() {
        let mut before: libc::rlimit = unsafe { std::mem::zeroed() };
        assert_eq!(unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut before) }, 0);
        let cur = before.rlim_cur as u64;
        assert_eq!(raise_open_file_limit(0), Some(cur));
        assert_eq!(raise_open_file_limit(cur), Some(cur));

        let desired = cur.saturating_add(16);
        let expected_max = if before.rlim_max == libc::RLIM_INFINITY {
            desired
        } else {
            before.rlim_max as u64
        };
        let raised = raise_open_file_limit(desired).expect("getrlimit succeeded");
        assert!(raised >= cur && raised <= desired.min(expected_max));
        // Restore the prior soft limit so other tests in the process see the original value.
        let _ = unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &before) };
    }
}
