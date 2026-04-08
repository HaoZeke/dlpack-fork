//! This module provides conversions between `Arc<Mutex<Array<T, D>>>` and
//! `Arc<RwLock<Array<T, D>>>` to `DLPackTensor`, allowing you to share data
//! between Rust and DLPack while ensuring that the data is not modified by Rust
//! while it's used by DLPack and reciprocally. The locks will be held until the
//! `DLPackTensor` is dropped, ensuring safe access to the data.
//!
//! The following conversions to DLPack types are supported:
//!
//! - `Arc<Mutex<Array<T, D>>> -> DLPackTensor`
//! - `Arc<RwLock<Array<T, D>>> -> DLPackTensor`

use std::sync::{Arc, Mutex, MutexGuard, RwLock, RwLockWriteGuard};
use ndarray::{Array, Dimension};

use crate::sys;
use crate::{DLPackTensor, GetDLPackDataType};

use super::DLPackNDarrayError;

// See crate::sync module comment for the self-referential safety argument.
// The same reasoning applies here: guards borrow from heap-allocated Mutex/RwLock
// inside Arc, field drop order ensures guard drops before arc.

struct ManagerContextRwLock<A: 'static> {
    // IMPORTANT: guard MUST be declared before arc so it drops first.
    guard: RwLockWriteGuard<'static, A>,
    _arc: Arc<RwLock<A>>,
    shape: Vec<i64>,
    strides: Vec<i64>,
}

unsafe extern "C" fn rwlock_deleter_fn<A>(manager: *mut sys::DLManagedTensorVersioned) where A: 'static {
    let ctx = (*manager).manager_ctx.cast::<ManagerContextRwLock<A>>();
    let _ = Box::from_raw(ctx);
}

impl<T, D> TryFrom<Arc<RwLock<Array<T, D>>>> for DLPackTensor
where
    D: Dimension + 'static,
    T: GetDLPackDataType + 'static + Clone,
{
    type Error = DLPackNDarrayError;

    fn try_from(array: Arc<RwLock<Array<T, D>>>) -> Result<Self, Self::Error> {
        let guard = array.write()?;
        let shape: Vec<i64> = guard.shape().iter().map(|&s| s as i64).collect();
        let strides: Vec<i64> = guard.strides().iter().map(|&s| s as i64).collect();
        let ndim = shape.len() as i32;

        // SAFETY: see crate::sync module comment on self-referential safety.
        let guard: RwLockWriteGuard<'static, Array<T, D>> = unsafe {
            std::mem::transmute(guard)
        };

        let ctx = Box::new(ManagerContextRwLock {
            guard,
            _arc: array,
            shape,
            strides,
        });

        // SAFETY: Convert to raw pointer FIRST, then derive all field pointers
        // from it. This avoids Stacked Borrows violations where Box::into_raw
        // would invalidate pointers obtained from the Box.
        let ctx_ptr = Box::into_raw(ctx);
        let shape_ptr = unsafe { (*ctx_ptr).shape.as_mut_ptr() };
        let stride_ptr = unsafe { (*ctx_ptr).strides.as_mut_ptr() };
        let data = unsafe { (*ctx_ptr).guard.as_mut_ptr().cast() };

        let dl_tensor = sys::DLTensor {
            data,
            device: sys::DLDevice {
                device_type: sys::DLDeviceType::kDLCPU,
                device_id: 0,
            },
            ndim,
            dtype: T::get_dlpack_data_type(),
            shape: shape_ptr,
            strides: stride_ptr,
            byte_offset: 0,
        };

        let managed_tensor = sys::DLManagedTensorVersioned {
            version: sys::DLPackVersion::current(),
            manager_ctx: ctx_ptr.cast(),
            deleter: Some(rwlock_deleter_fn::<Array<T, D>>),
            flags: 0,
            dl_tensor,
        };

        unsafe {
            Ok(DLPackTensor::from_raw(managed_tensor))
        }
    }
}


struct ManagerContextMutex<A: 'static> {
    // IMPORTANT: guard MUST be declared before arc so it drops first.
    guard: MutexGuard<'static, A>,
    _arc: Arc<Mutex<A>>,
    shape: Vec<i64>,
    strides: Vec<i64>,
}

unsafe extern "C" fn mutex_deleter_fn<A>(manager: *mut sys::DLManagedTensorVersioned) where A: 'static {
    let ctx = (*manager).manager_ctx.cast::<ManagerContextMutex<A>>();
    let _ = Box::from_raw(ctx);
}

impl<T, D> TryFrom<Arc<Mutex<Array<T, D>>>> for DLPackTensor
where
    D: Dimension + 'static,
    T: GetDLPackDataType + 'static + Clone,
{
    type Error = DLPackNDarrayError;

    fn try_from(array: Arc<Mutex<Array<T, D>>>) -> Result<Self, Self::Error> {
        let guard = array.lock()?;
        let shape: Vec<i64> = guard.shape().iter().map(|&s| s as i64).collect();
        let strides: Vec<i64> = guard.strides().iter().map(|&s| s as i64).collect();
        let ndim = shape.len() as i32;

        // SAFETY: see crate::sync module comment on self-referential safety.
        let guard: MutexGuard<'static, Array<T, D>> = unsafe {
            std::mem::transmute(guard)
        };

        let ctx = Box::new(ManagerContextMutex {
            guard,
            _arc: array,
            shape,
            strides,
        });

        let ctx_ptr = Box::into_raw(ctx);
        let shape_ptr = unsafe { (*ctx_ptr).shape.as_mut_ptr() };
        let stride_ptr = unsafe { (*ctx_ptr).strides.as_mut_ptr() };
        let data = unsafe { (*ctx_ptr).guard.as_mut_ptr().cast() };

        let dl_tensor = sys::DLTensor {
            data,
            device: sys::DLDevice {
                device_type: sys::DLDeviceType::kDLCPU,
                device_id: 0,
            },
            ndim,
            dtype: T::get_dlpack_data_type(),
            shape: shape_ptr,
            strides: stride_ptr,
            byte_offset: 0,
        };

        let managed_tensor = sys::DLManagedTensorVersioned {
            version: sys::DLPackVersion::current(),
            manager_ctx: ctx_ptr.cast(),
            deleter: Some(mutex_deleter_fn::<Array<T, D>>),
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
    use ndarray::{ArrayViewMutD, arr2};

    #[test]
    fn test_mutex() {
        let array = Arc::new(Mutex::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])));

        {
            let mut tensor: DLPackTensor = Arc::clone(&array).try_into().unwrap();
            assert!(array.try_lock().is_err(), "the mutex should be locked while the DLPackTensor exists");

            let tensor_mut_ref = tensor.as_mut();
            let mut view: ArrayViewMutD<f64> = tensor_mut_ref.try_into().unwrap();

            view[[1, 1]] = 42.0;
        }

        let array = array.lock().unwrap();
        assert_eq!(*array, arr2(&[[1.0, 2.0, 3.0], [4.0, 42.0, 6.0]]));
    }

    #[test]
    fn test_rwlock() {
        let array = Arc::new(RwLock::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])));

        {
            let mut tensor: DLPackTensor = Arc::clone(&array).try_into().unwrap();
            assert!(array.try_read().is_err(), "the rwlock should be locked while the DLPackTensor exists");
            assert!(array.try_write().is_err(), "the rwlock should be locked while the DLPackTensor exists");

            let tensor_mut_ref = tensor.as_mut();
            let mut view: ArrayViewMutD<f64> = tensor_mut_ref.try_into().unwrap();

            view[[1, 1]] = 42.0;
        }

        let array = array.read().unwrap();
        assert_eq!(*array, arr2(&[[1.0, 2.0, 3.0], [4.0, 42.0, 6.0]]));
    }

    /// When the DLPackTensor holds the LAST Arc reference, the deleter must
    /// deallocate the inner Array correctly. This catches type parameter
    /// mismatches in the deleter function (e.g. using element type T instead
    /// of the full Array<T, D>).
    #[test]
    fn test_mutex_last_arc_ref() {
        let array = Arc::new(Mutex::new(arr2(&[[1.0, 2.0], [3.0, 4.0]])));
        let tensor: DLPackTensor = array.try_into().unwrap();
        drop(tensor);
    }

    #[test]
    fn test_rwlock_last_arc_ref() {
        let array = Arc::new(RwLock::new(arr2(&[[1.0, 2.0], [3.0, 4.0]])));
        let tensor: DLPackTensor = array.try_into().unwrap();
        drop(tensor);
    }
}
