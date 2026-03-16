//! This module provides conversions between `Arc<Mutex<Vec<T>>>` and
//! `Arc<RwLock<Vec<T>>>` to `DLPackTensor`, allowing you to share data between
//! Rust and DLPack while ensuring that the data is not modified by Rust while
//! it's used by DLPack and reciprocally. The locks will be held until the
//! `DLPackTensor` is dropped, ensuring safe access to the data.
//!
//! The following conversions to DLPack types are supported:
//!
//! - `Arc<Mutex<Vec<T>>> -> DLPackTensor`
//! - `Arc<RwLock<Vec<T>>> -> DLPackTensor`

use std::sync::{Arc, Mutex, MutexGuard, RwLock, RwLockWriteGuard};

use crate::sys;
use crate::{DLPackTensor, GetDLPackDataType};

use crate::vec::DLPackVecError;

// The ManagerContext structs below are self-referential: the guard borrows from
// the Arc. This is safe because:
//
// 1. MutexGuard/RwLockWriteGuard internally hold a reference to the Mutex/RwLock
//    inside the Arc, which is heap-allocated. Moving the Arc (a pointer) does not
//    invalidate the guard.
// 2. The guard field is declared BEFORE the arc field, so Rust's field drop order
//    guarantees the guard drops first (releasing the lock) before the Arc drops.
// 3. The struct is heap-allocated (Box) in the manager context and never moved.
//
// We use std::mem::transmute to extend the guard's lifetime to 'static. This is
// the standard pattern for self-referential structs where the borrow target is
// heap-allocated and outlives the borrower.

struct ManagerContextMutex<T: 'static> {
    // IMPORTANT: guard MUST be declared before arc so it drops first.
    guard: MutexGuard<'static, Vec<T>>,
    _arc: Arc<Mutex<Vec<T>>>,
    shape: i64,
    stride: i64,
}

unsafe extern "C" fn mutex_deleter_fn<T>(manager: *mut sys::DLManagedTensorVersioned) where T: 'static {
    let ctx = (*manager).manager_ctx.cast::<ManagerContextMutex<T>>();
    let _ = Box::from_raw(ctx);
}

impl<T> TryFrom<Arc<Mutex<Vec<T>>>> for DLPackTensor where T: GetDLPackDataType + 'static {
    type Error = DLPackVecError;

    fn try_from(array: Arc<Mutex<Vec<T>>>) -> Result<DLPackTensor, Self::Error> {
        let guard = array.lock()?;
        let shape = guard.len() as i64;

        // SAFETY: see module-level comment on self-referential safety.
        let guard: MutexGuard<'static, Vec<T>> = unsafe { std::mem::transmute(guard) };

        let mut ctx = Box::new(ManagerContextMutex {
            guard,
            _arc: array,
            shape,
            stride: 1,
        });

        let shape_ptr = &mut ctx.shape as *mut i64;
        let stride_ptr = &mut ctx.stride as *mut i64;
        let data = ctx.guard.as_mut_ptr().cast();

        let dl_tensor = sys::DLTensor {
            data,
            device: sys::DLDevice {
                device_type: sys::DLDeviceType::kDLCPU,
                device_id: 0,
            },
            ndim: 1,
            dtype: T::get_dlpack_data_type(),
            shape: shape_ptr,
            strides: stride_ptr,
            byte_offset: 0,
        };

        let managed_tensor = sys::DLManagedTensorVersioned {
            version: sys::DLPackVersion::current(),
            manager_ctx: Box::into_raw(ctx).cast(),
            deleter: Some(mutex_deleter_fn::<T>),
            flags: 0,
            dl_tensor,
        };

        unsafe {
            Ok(DLPackTensor::from_raw(managed_tensor))
        }
    }
}

struct ManagerContextRwLock<T: 'static> {
    // IMPORTANT: guard MUST be declared before arc so it drops first.
    guard: RwLockWriteGuard<'static, Vec<T>>,
    _arc: Arc<RwLock<Vec<T>>>,
    shape: i64,
    stride: i64,
}

unsafe extern "C" fn rwlock_deleter_fn<T>(manager: *mut sys::DLManagedTensorVersioned) where T: 'static {
    let ctx = (*manager).manager_ctx.cast::<ManagerContextRwLock<T>>();
    let _ = Box::from_raw(ctx);
}

impl<T> TryFrom<Arc<RwLock<Vec<T>>>> for DLPackTensor where T: GetDLPackDataType + 'static {
    type Error = DLPackVecError;

    fn try_from(array: Arc<RwLock<Vec<T>>>) -> Result<DLPackTensor, Self::Error> {
        let guard = array.write()?;
        let shape = guard.len() as i64;

        // SAFETY: see module-level comment on self-referential safety.
        let guard: RwLockWriteGuard<'static, Vec<T>> = unsafe { std::mem::transmute(guard) };

        let mut ctx = Box::new(ManagerContextRwLock {
            guard,
            _arc: array,
            shape,
            stride: 1,
        });

        let shape_ptr = &mut ctx.shape as *mut i64;
        let stride_ptr = &mut ctx.stride as *mut i64;
        let data = ctx.guard.as_mut_ptr().cast();

        let dl_tensor = sys::DLTensor {
            data,
            device: sys::DLDevice {
                device_type: sys::DLDeviceType::kDLCPU,
                device_id: 0,
            },
            ndim: 1,
            dtype: T::get_dlpack_data_type(),
            shape: shape_ptr,
            strides: stride_ptr,
            byte_offset: 0,
        };

        let managed_tensor = sys::DLManagedTensorVersioned {
            version: sys::DLPackVersion::current(),
            manager_ctx: Box::into_raw(ctx).cast(),
            deleter: Some(rwlock_deleter_fn::<T>),
            flags: 0,
            dl_tensor,
        };

        unsafe {
            Ok(DLPackTensor::from_raw(managed_tensor))
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mutex() {
        let data = Arc::new(Mutex::new(vec![1, 2, 3]));

        {
            let mut tensor: DLPackTensor = Arc::clone(&data).try_into().unwrap();
            assert!(data.try_lock().is_err(), "the mutex should be locked while the DLPackTensor exists");

            let tensor_mut_ref = tensor.as_mut();
            let slice: &mut[i32] = tensor_mut_ref.try_into().unwrap();

            slice[1] = 42;
        }

        let lock = data.lock().unwrap();
        assert_eq!(&*lock, &[1, 42, 3]);
    }

    #[test]
    fn test_rwlock() {
        let data = Arc::new(RwLock::new(vec![1, 2, 3]));

        {
            let mut tensor: DLPackTensor = Arc::clone(&data).try_into().unwrap();
            assert!(data.try_read().is_err(), "the rwlock should be locked while the DLPackTensor exists");
            assert!(data.try_write().is_err(), "the rwlock should be locked while the DLPackTensor exists");

            let tensor_mut_ref = tensor.as_mut();
            let slice: &mut[i32] = tensor_mut_ref.try_into().unwrap();

            slice[1] = 42;
        }

        let lock = data.read().unwrap();
        assert_eq!(&*lock, &[1, 42, 3]);
    }
}
