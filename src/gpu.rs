//! Optional OpenCL GPU acceleration for block compression.
//!
//! The kernel is a byte-exact port of [`crate::lz::compress`]: same hash, same
//! chain walk, same lazy parse, same token emission. An archive written with
//! `--gpu` is therefore identical to one written without it — the GPU only
//! changes how fast blocks are coded, never what comes out.
//!
//! OpenCL is loaded at runtime via `LoadLibrary`/`GetProcAddress` so the binary
//! has no link-time dependency on `OpenCL.dll` and still builds (with a stub)
//! on machines that have no GPU driver.

use std::ffi::c_void;

use crate::lz::Effort;

/// One batch never exceeds these limits; the writer splits larger runs.
const MAX_BATCH_BLOCKS: usize = 32;
const MAX_BATCH_BYTES: usize = 8 * 1024 * 1024;

// ---------------------------------------------------------------------------
// OpenCL constants
// ---------------------------------------------------------------------------
const CL_SUCCESS: i32 = 0;
const CL_DEVICE_TYPE_GPU: u64 = 1 << 2;
const CL_MEM_READ_WRITE: u64 = 1 << 0;
const CL_MEM_READ_ONLY: u64 = 1 << 2;
const CL_PROGRAM_BUILD_LOG: u32 = 0x1183;
const CL_DEVICE_NAME: u32 = 0x102B;

// ---------------------------------------------------------------------------
// Windows: runtime loader
// ---------------------------------------------------------------------------
#[cfg(windows)]
mod os {
    use std::ffi::{c_void, CString};

    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryA(lp_libfilename: *const i8) -> *mut c_void;
        fn GetProcAddress(h_module: *mut c_void, lp_procname: *const i8) -> *mut c_void;
        fn FreeLibrary(h_module: *mut c_void) -> i32;
    }

    pub struct Library(*mut c_void);

    impl Library {
        pub fn open(name: &str) -> Option<Library> {
            let c = CString::new(name).ok()?;
            unsafe {
                let h = LoadLibraryA(c.as_ptr() as *const i8);
                if h.is_null() {
                    None
                } else {
                    Some(Library(h))
                }
            }
        }

        pub fn sym(&self, name: &str) -> Option<*mut c_void> {
            let c = CString::new(name).ok()?;
            unsafe { Some(GetProcAddress(self.0, c.as_ptr() as *const i8)) }
                .filter(|p| !p.is_null())
        }
    }

    impl Drop for Library {
        fn drop(&mut self) {
            unsafe {
                FreeLibrary(self.0);
            }
        }
    }

    // OpenCL uses the platform C calling convention; on x64 Windows that is the
    // same for `extern "C"` and `extern "system"`.
    pub type ClInt = i32;
    pub type ClUInt = u32;
    pub type ClMem = *mut c_void;
    pub type ClPlatformId = *mut c_void;
    pub type ClDeviceId = *mut c_void;
    pub type ClContext = *mut c_void;
    pub type ClCommandQueue = *mut c_void;
    pub type ClProgram = *mut c_void;
    pub type ClKernel = *mut c_void;

    pub struct Api {
        pub get_platform_ids: unsafe extern "C" fn(ClUInt, *mut ClPlatformId, *mut ClUInt) -> ClInt,
        pub get_device_ids:
            unsafe extern "C" fn(ClPlatformId, u64, ClUInt, *mut ClDeviceId, *mut ClUInt) -> ClInt,
        pub get_device_info:
            unsafe extern "C" fn(ClDeviceId, ClUInt, usize, *mut c_void, *mut usize) -> ClInt,
        pub create_context: unsafe extern "C" fn(
            *const c_void,
            ClUInt,
            *const ClDeviceId,
            Option<unsafe extern "C" fn(*const c_void, *const c_void, usize, *mut c_void)>,
            *mut c_void,
            *mut ClInt,
        ) -> ClContext,
        pub create_command_queue:
            unsafe extern "C" fn(ClContext, ClDeviceId, u64, *mut ClInt) -> ClCommandQueue,
        pub create_program_with_source: unsafe extern "C" fn(
            ClContext,
            ClUInt,
            *const *const i8,
            *const usize,
            *mut ClInt,
        ) -> ClProgram,
        pub build_program: unsafe extern "C" fn(
            ClProgram,
            ClUInt,
            *const ClDeviceId,
            *const i8,
            Option<unsafe extern "C" fn(ClProgram, *mut c_void)>,
            *mut c_void,
        ) -> ClInt,
        pub get_program_build_info: unsafe extern "C" fn(
            ClProgram,
            ClDeviceId,
            ClUInt,
            usize,
            *mut c_void,
            *mut usize,
        ) -> ClInt,
        pub create_kernel: unsafe extern "C" fn(ClProgram, *const i8, *mut ClInt) -> ClKernel,
        pub create_buffer:
            unsafe extern "C" fn(ClContext, u64, usize, *mut c_void, *mut ClInt) -> ClMem,
        pub set_kernel_arg: unsafe extern "C" fn(ClKernel, ClUInt, usize, *const c_void) -> ClInt,
        pub enqueue_write_buffer: unsafe extern "C" fn(
            ClCommandQueue,
            ClMem,
            ClUInt,
            usize,
            usize,
            *const c_void,
            ClUInt,
            *const *mut c_void,
            *mut *mut c_void,
        ) -> ClInt,
        pub enqueue_nd_range_kernel: unsafe extern "C" fn(
            ClCommandQueue,
            ClKernel,
            ClUInt,
            *const usize,
            *const usize,
            *const usize,
            ClUInt,
            *const *mut c_void,
            *mut *mut c_void,
        ) -> ClInt,
        pub enqueue_read_buffer: unsafe extern "C" fn(
            ClCommandQueue,
            ClMem,
            ClUInt,
            usize,
            usize,
            *mut c_void,
            ClUInt,
            *const *mut c_void,
            *mut *mut c_void,
        ) -> ClInt,
        pub finish: unsafe extern "C" fn(ClCommandQueue) -> ClInt,
        pub release_mem_object: unsafe extern "C" fn(ClMem) -> ClInt,
        pub release_kernel: unsafe extern "C" fn(ClKernel) -> ClInt,
        pub release_program: unsafe extern "C" fn(ClProgram) -> ClInt,
        pub release_command_queue: unsafe extern "C" fn(ClCommandQueue) -> ClInt,
        pub release_context: unsafe extern "C" fn(ClContext) -> ClInt,
        _lib: Library,
    }

    impl Api {
        pub fn load() -> Option<Api> {
            let lib = Library::open("OpenCL.dll")?;
            macro_rules! load_fn {
                ($name:literal, $ty:ty) => {{
                    let p = lib.sym($name)?;
                    let f: $ty = unsafe { std::mem::transmute(p) };
                    Some(f)
                }};
            }
            let get_platform_ids = load_fn!(
                "clGetPlatformIDs",
                unsafe extern "C" fn(ClUInt, *mut ClPlatformId, *mut ClUInt) -> ClInt
            )?;
            let get_device_ids = load_fn!(
                "clGetDeviceIDs",
                unsafe extern "C" fn(
                    ClPlatformId,
                    u64,
                    ClUInt,
                    *mut ClDeviceId,
                    *mut ClUInt,
                ) -> ClInt
            )?;
            let get_device_info = load_fn!(
                "clGetDeviceInfo",
                unsafe extern "C" fn(ClDeviceId, ClUInt, usize, *mut c_void, *mut usize) -> ClInt
            )?;
            let create_context = load_fn!(
                "clCreateContext",
                unsafe extern "C" fn(
                    *const c_void,
                    ClUInt,
                    *const ClDeviceId,
                    Option<unsafe extern "C" fn(*const c_void, *const c_void, usize, *mut c_void)>,
                    *mut c_void,
                    *mut ClInt,
                ) -> ClContext
            )?;
            let create_command_queue = load_fn!(
                "clCreateCommandQueue",
                unsafe extern "C" fn(ClContext, ClDeviceId, u64, *mut ClInt) -> ClCommandQueue
            )?;
            let create_program_with_source = load_fn!(
                "clCreateProgramWithSource",
                unsafe extern "C" fn(
                    ClContext,
                    ClUInt,
                    *const *const i8,
                    *const usize,
                    *mut ClInt,
                ) -> ClProgram
            )?;
            let build_program = load_fn!(
                "clBuildProgram",
                unsafe extern "C" fn(
                    ClProgram,
                    ClUInt,
                    *const ClDeviceId,
                    *const i8,
                    Option<unsafe extern "C" fn(ClProgram, *mut c_void)>,
                    *mut c_void,
                ) -> ClInt
            )?;
            let get_program_build_info = load_fn!(
                "clGetProgramBuildInfo",
                unsafe extern "C" fn(
                    ClProgram,
                    ClDeviceId,
                    ClUInt,
                    usize,
                    *mut c_void,
                    *mut usize,
                ) -> ClInt
            )?;
            let create_kernel = load_fn!(
                "clCreateKernel",
                unsafe extern "C" fn(ClProgram, *const i8, *mut ClInt) -> ClKernel
            )?;
            let create_buffer = load_fn!(
                "clCreateBuffer",
                unsafe extern "C" fn(ClContext, u64, usize, *mut c_void, *mut ClInt) -> ClMem
            )?;
            let set_kernel_arg = load_fn!(
                "clSetKernelArg",
                unsafe extern "C" fn(ClKernel, ClUInt, usize, *const c_void) -> ClInt
            )?;
            let enqueue_write_buffer = load_fn!(
                "clEnqueueWriteBuffer",
                unsafe extern "C" fn(
                    ClCommandQueue,
                    ClMem,
                    ClUInt,
                    usize,
                    usize,
                    *const c_void,
                    ClUInt,
                    *const *mut c_void,
                    *mut *mut c_void,
                ) -> ClInt
            )?;
            let enqueue_nd_range_kernel = load_fn!(
                "clEnqueueNDRangeKernel",
                unsafe extern "C" fn(
                    ClCommandQueue,
                    ClKernel,
                    ClUInt,
                    *const usize,
                    *const usize,
                    *const usize,
                    ClUInt,
                    *const *mut c_void,
                    *mut *mut c_void,
                ) -> ClInt
            )?;
            let enqueue_read_buffer = load_fn!(
                "clEnqueueReadBuffer",
                unsafe extern "C" fn(
                    ClCommandQueue,
                    ClMem,
                    ClUInt,
                    usize,
                    usize,
                    *mut c_void,
                    ClUInt,
                    *const *mut c_void,
                    *mut *mut c_void,
                ) -> ClInt
            )?;
            let finish = load_fn!("clFinish", unsafe extern "C" fn(ClCommandQueue) -> ClInt)?;
            let release_mem_object =
                load_fn!("clReleaseMemObject", unsafe extern "C" fn(ClMem) -> ClInt)?;
            let release_kernel =
                load_fn!("clReleaseKernel", unsafe extern "C" fn(ClKernel) -> ClInt)?;
            let release_program =
                load_fn!("clReleaseProgram", unsafe extern "C" fn(ClProgram) -> ClInt)?;
            let release_command_queue = load_fn!(
                "clReleaseCommandQueue",
                unsafe extern "C" fn(ClCommandQueue) -> ClInt
            )?;
            let release_context =
                load_fn!("clReleaseContext", unsafe extern "C" fn(ClContext) -> ClInt)?;
            Some(Api {
                get_platform_ids,
                get_device_ids,
                get_device_info,
                create_context,
                create_command_queue,
                create_program_with_source,
                build_program,
                get_program_build_info,
                create_kernel,
                create_buffer,
                set_kernel_arg,
                enqueue_write_buffer,
                enqueue_nd_range_kernel,
                enqueue_read_buffer,
                finish,
                release_mem_object,
                release_kernel,
                release_program,
                release_command_queue,
                release_context,
                _lib: lib,
            })
        }
    }
}

#[cfg(not(windows))]
mod os {
    use std::ffi::c_void;

    /// Stub implementation that always fails to load.
    /// This ensures that OpenCL is not supported on Linux/WSL.
    pub type ClInt = i32;
    pub type ClUInt = u32;
    pub type ClMem = *mut c_void;
    pub type ClPlatformId = *mut c_void;
    pub type ClDeviceId = *mut c_void;
    pub type ClContext = *mut c_void;
    pub type ClCommandQueue = *mut c_void;
    pub type ClProgram = *mut c_void;
    pub type ClKernel = *mut c_void;

    /// API struct with empty methods that fail when called.
    pub struct Api {
        // All function pointers are null, which will cause errors when called
        pub get_platform_ids: unsafe extern "C" fn(ClUInt, *mut ClPlatformId, *mut ClUInt) -> ClInt,
        pub get_device_ids: unsafe extern "C" fn(ClPlatformId, u64, ClUInt, *mut ClDeviceId, *mut ClUInt) -> ClInt,
        pub get_device_info: unsafe extern "C" fn(ClDeviceId, ClUInt, usize, *mut c_void, *mut usize) -> ClInt,
        pub create_context: unsafe extern "C" fn(*const c_void, ClUInt, *const ClDeviceId,
            Option<unsafe extern "C" fn(*const c_void, *const c_void, usize, *mut c_void)>,
            *mut c_void, *mut ClInt) -> ClContext,
        pub create_command_queue: unsafe extern "C" fn(ClContext, ClDeviceId, u64, *mut ClInt) -> ClCommandQueue,
        pub create_program_with_source: unsafe extern "C" fn(ClContext, ClUInt, *const *const i8,
            *const usize, *mut ClInt) -> ClProgram,
        pub build_program: unsafe extern "C" fn(ClProgram, ClUInt, *const ClDeviceId, *const i8,
            Option<unsafe extern "C" fn(ClProgram, *mut c_void)>, *mut c_void) -> ClInt,
        pub get_program_build_info: unsafe extern "C" fn(ClProgram, ClDeviceId, ClUInt, usize,
            *mut c_void, *mut usize) -> ClInt,
        pub create_kernel: unsafe extern "C" fn(ClProgram, *const i8, *mut ClInt) -> ClKernel,
        pub create_buffer: unsafe extern "C" fn(ClContext, u64, usize, *mut c_void, *mut ClInt) -> ClMem,
        pub set_kernel_arg: unsafe extern "C" fn(ClKernel, ClUInt, usize, *const c_void) -> ClInt,
        pub enqueue_write_buffer: unsafe extern "C" fn(ClCommandQueue, ClMem, ClUInt, usize, usize,
            *const c_void, ClUInt, *const *mut c_void, *mut *mut c_void) -> ClInt,
        pub enqueue_nd_range_kernel: unsafe extern "C" fn(ClCommandQueue, ClKernel, ClUInt,
            *const usize, *const usize, *const usize, ClUInt, *const *mut c_void,
            *mut *mut c_void) -> ClInt,
        pub enqueue_read_buffer: unsafe extern "C" fn(ClCommandQueue, ClMem, ClUInt, usize, usize,
            *mut c_void, ClUInt, *const *mut c_void, *mut *mut c_void) -> ClInt,
        pub finish: unsafe extern "C" fn(ClCommandQueue) -> ClInt,
        pub release_mem_object: unsafe extern "C" fn(ClMem) -> ClInt,
        pub release_kernel: unsafe extern "C" fn(ClKernel) -> ClInt,
        pub release_program: unsafe extern "C" fn(ClProgram) -> ClInt,
        pub release_command_queue: unsafe extern "C" fn(ClCommandQueue) -> ClInt,
        pub release_context: unsafe extern "C" fn(ClContext) -> ClInt,
    }

    impl Api {
        /// Always returns None since OpenCL is not supported on this platform.
        /// This forces Gpu::open() to fall back to CPU-only compression.
        pub fn load() -> Option<Api> {
            None
        }
    }
}

use os::*;

/// What the GPU worker needs from whatever accelerator is available.
///
/// Three backends are planned over the life of the project: OpenCL today
/// (which is what covers Intel and AMD, and older NVIDIA), ROCm/HIP if an AMD
/// part turns out to need it, and CUDA for NVIDIA at 1.0. Only the two methods
/// below are load-bearing, so adding a backend is a new file plus one arm in
/// [`Gpu::open`]; `writer.rs` does not change.
pub trait GpuBackend {
    /// Human-readable device name for the start banner.
    fn device_name(&self) -> &str;

    /// Compress a batch of independent blocks.
    ///
    /// Returns one entry per input block: `Some(coded)` when the accelerator
    /// produced a smaller encoding, `None` when the block should be stored
    /// verbatim. Both are the same decline rule the CPU codec uses, so the
    /// choice of backend cannot change the archive.
    fn compress_batch(
        &mut self,
        blocks: &[&[u8]],
        level: u8,
    ) -> Result<Vec<Option<Vec<u8>>>, String>;

    /// Which backend this is, for the banner and for `--verbose`.
    fn backend_name(&self) -> &'static str {
        "OpenCL"
    }
}

/// An OpenCL context + queue + prebuilt compression kernel.
pub struct OpenCl {
    api: Api,
    context: ClContext,
    queue: ClCommandQueue,
    program: ClProgram,
    kernel: ClKernel,
    device_name: String,
}

// The raw handles are plain pointers; the GPU worker thread owns the backend
// after creation, and the unit tests create one on the test thread and drop it
// there.
unsafe impl Send for OpenCl {}

/// Picks whichever accelerator backend the current system has.
pub struct Gpu {
    backend: Box<dyn GpuBackend + Send>,
}

impl Gpu {
    /// Open the best available backend, trying CUDA first, then OpenCL.
    ///
    /// CUDA is absent from this build for now and the arm below is written
    /// out so that adding it later is a single line plus a new type. When
    /// everything fails the caller falls back to CPU-only compression, so a
    /// machine with no GPU is a slower run, never a failed one.
    pub fn open() -> Result<Gpu, String> {
        // Future: if let Ok(c) = Cuda::open() { return Ok(c.into()); }
        match OpenCl::new() {
            Ok(ocl) => Ok(Gpu {
                backend: Box::new(ocl),
            }),
            Err(e) => Err(e),
        }
    }
}

impl GpuBackend for Gpu {
    fn device_name(&self) -> &str {
        self.backend.device_name()
    }
    fn compress_batch(
        &mut self,
        blocks: &[&[u8]],
        level: u8,
    ) -> Result<Vec<Option<Vec<u8>>>, String> {
        self.backend.compress_batch(blocks, level)
    }
    fn backend_name(&self) -> &'static str {
        self.backend.backend_name()
    }
}

impl OpenCl {
    /// Find the first GPU, build the kernel, and return a ready context.
    fn new() -> Result<OpenCl, String> {
        let api = Api::load().ok_or_else(|| {
            if cfg!(windows) {
                "OpenCL.dll not found or not loadable".to_string()
            } else {
                "GPU / OpenCL not supported on this platform (Linux/WSL)".to_string()
            }
        })?;

        let mut nplat: ClUInt = 0;
        let err = unsafe { (api.get_platform_ids)(0, std::ptr::null_mut(), &mut nplat) };
        if err != CL_SUCCESS || nplat == 0 {
            return Err(format!("no OpenCL platforms (err {err})"));
        }
        let mut platforms = vec![std::ptr::null_mut(); nplat as usize];
        let err =
            unsafe { (api.get_platform_ids)(nplat, platforms.as_mut_ptr(), std::ptr::null_mut()) };
        if err != CL_SUCCESS {
            return Err(format!("clGetPlatformIDs failed (err {err})"));
        }

        let mut device: ClDeviceId = std::ptr::null_mut();
        let mut chosen_platform: ClPlatformId = std::ptr::null_mut();
        for &plat in &platforms {
            let mut ndev: ClUInt = 0;
            let err = unsafe {
                (api.get_device_ids)(plat, CL_DEVICE_TYPE_GPU, 0, std::ptr::null_mut(), &mut ndev)
            };
            if err != CL_SUCCESS || ndev == 0 {
                continue;
            }
            let mut devs = vec![std::ptr::null_mut(); ndev as usize];
            let err = unsafe {
                (api.get_device_ids)(
                    plat,
                    CL_DEVICE_TYPE_GPU,
                    ndev,
                    devs.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            };
            if err == CL_SUCCESS && ndev > 0 {
                device = devs[0];
                chosen_platform = plat;
                break;
            }
        }
        let _ = chosen_platform;
        if device.is_null() {
            return Err("no OpenCL GPU device found".into());
        }

        // Device name for the banner.
        let mut name_len: usize = 0;
        unsafe {
            (api.get_device_info)(
                device,
                CL_DEVICE_NAME,
                0,
                std::ptr::null_mut(),
                &mut name_len,
            )
        };
        let mut name_buf = vec![0u8; name_len.max(1)];
        unsafe {
            (api.get_device_info)(
                device,
                CL_DEVICE_NAME,
                name_buf.len(),
                name_buf.as_mut_ptr() as *mut c_void,
                std::ptr::null_mut(),
            )
        };
        let device_name = std::ffi::CStr::from_bytes_until_nul(&name_buf)
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|_| "OpenCL GPU".into());

        // Context with a single device.
        let mut err: ClInt = 0;
        let context = unsafe {
            (api.create_context)(
                std::ptr::null(),
                1,
                &device,
                None,
                std::ptr::null_mut(),
                &mut err,
            )
        };
        if err != CL_SUCCESS || context.is_null() {
            return Err(format!("clCreateContext failed (err {err})"));
        }

        let queue = unsafe { (api.create_command_queue)(context, device, 0, &mut err) };
        if err != CL_SUCCESS || queue.is_null() {
            unsafe { (api.release_context)(context) };
            return Err(format!("clCreateCommandQueue failed (err {err})"));
        }

        // Build the kernel from the embedded source.
        let src = KERNEL_SOURCE;
        let src_ptr: *const i8 = src.as_ptr() as *const i8;
        let src_len = src.len();
        let program =
            unsafe { (api.create_program_with_source)(context, 1, &src_ptr, &src_len, &mut err) };
        if err != CL_SUCCESS || program.is_null() {
            unsafe {
                (api.release_command_queue)(queue);
                (api.release_context)(context);
            }
            return Err(format!("clCreateProgramWithSource failed (err {err})"));
        }

        let opts = b"-cl-std=CL1.2\0";
        let err = unsafe {
            (api.build_program)(
                program,
                1,
                &device,
                opts.as_ptr() as *const i8,
                None,
                std::ptr::null_mut(),
            )
        };
        if err != CL_SUCCESS {
            // Pull the build log for a useful error.
            let mut log_size: usize = 0;
            unsafe {
                (api.get_program_build_info)(
                    program,
                    device,
                    CL_PROGRAM_BUILD_LOG,
                    0,
                    std::ptr::null_mut(),
                    &mut log_size,
                );
            }
            let mut log_buf = vec![0u8; log_size.max(1)];
            unsafe {
                (api.get_program_build_info)(
                    program,
                    device,
                    CL_PROGRAM_BUILD_LOG,
                    log_buf.len(),
                    log_buf.as_mut_ptr() as *mut c_void,
                    std::ptr::null_mut(),
                );
            }
            let log = std::ffi::CStr::from_bytes_until_nul(&log_buf)
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            unsafe {
                (api.release_program)(program);
                (api.release_command_queue)(queue);
                (api.release_context)(context);
            }
            return Err(format!("clBuildProgram failed (err {err}): {log}"));
        }

        let mut err2: ClInt = 0;
        let kernel = unsafe {
            (api.create_kernel)(
                program,
                b"compress_blocks\0".as_ptr() as *const i8,
                &mut err2,
            )
        };
        if err2 != CL_SUCCESS || kernel.is_null() {
            unsafe {
                (api.release_program)(program);
                (api.release_command_queue)(queue);
                (api.release_context)(context);
            }
            return Err(format!("clCreateKernel failed (err {err2})"));
        }

        Ok(OpenCl {
            api,
            context,
            queue,
            program,
            kernel,
            device_name,
        })
    }
}

impl GpuBackend for OpenCl {
    /// Human-readable device name for the start banner.
    fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Compress a batch of independent blocks.
    ///
    /// Returns one entry per input block: `Some(coded)` when the GPU produced a
    /// smaller encoding, `None` when the block should be stored verbatim (same
    /// decline rule as the CPU codec).
    fn compress_batch(
        &mut self,
        blocks: &[&[u8]],
        level: u8,
    ) -> Result<Vec<Option<Vec<u8>>>, String> {
        let n = blocks.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if n > MAX_BATCH_BLOCKS {
            return Err(format!("batch of {n} blocks exceeds {MAX_BATCH_BLOCKS}"));
        }
        let total_in: usize = blocks.iter().map(|b| b.len()).sum();
        if total_in > MAX_BATCH_BYTES {
            return Err(format!(
                "batch of {total_in} bytes exceeds {MAX_BATCH_BYTES}"
            ));
        }

        let effort = Effort::for_level(level);
        if effort.max_chain == 0 || n == 0 {
            return Ok(vec![None; n]);
        }

        // Per-block host-side metadata.
        let mut in_off = Vec::with_capacity(n);
        let mut out_off = Vec::with_capacity(n);
        let mut block_len = Vec::with_capacity(n);
        let mut chain_off = Vec::with_capacity(n);
        let mut head_off = Vec::with_capacity(n);
        let mut slots_arr = Vec::with_capacity(n);
        let mut in_buf = Vec::with_capacity(total_in);
        // Worst case output = input size (we decline at >= input).
        let mut out_buf = vec![0u8; total_in];
        let mut out_len_arr = vec![0u32; n];

        let mut cur_in = 0usize;
        let mut cur_out = 0usize;
        let mut cur_chain = 0u32; // u32 index into chain workspace
        let mut cur_head = 0u32; // u32 index into head workspace
        for b in blocks.iter() {
            let len = b.len();
            in_off.push(cur_in as u32);
            out_off.push(cur_out as u32);
            block_len.push(len as u32);
            chain_off.push(cur_chain);
            slots_arr.push(slots_for(len) as u32);
            head_off.push(cur_head);
            in_buf.extend_from_slice(b);
            cur_in += len;
            cur_out += len; // out workspace sized to total_in
            cur_chain += len as u32;
            cur_head += slots_for(len) as u32;
        }

        let total_chain = cur_chain as usize; // u32 elements
        let total_head = cur_head as usize;

        // Upload and run.
        let api = &self.api;
        let ctx = self.context;
        let queue = self.queue;
        let kernel = self.kernel;

        // Declared without a value: each is assigned exactly once by the `mkbuf!`
        // macro below, which is why a `std::ptr::null_mut()` placeholder here
        // would only ever be dead.
        let cl_in: ClMem;
        let cl_out: ClMem;
        let cl_out_len: ClMem;
        let cl_in_off: ClMem;
        let cl_out_off: ClMem;
        let cl_block_len: ClMem;
        let cl_chain_off: ClMem;
        let cl_head_off: ClMem;
        let cl_slots: ClMem;
        let cl_chain: ClMem;
        let cl_head: ClMem;

        macro_rules! mkbuf {
            ($ptr:ident, $flags:expr, $bytes:expr) => {{
                let mut e: ClInt = 0;
                let m = unsafe {
                    (api.create_buffer)(ctx, $flags, $bytes, std::ptr::null_mut(), &mut e)
                };
                if e != CL_SUCCESS || m.is_null() {
                    return Err(format!("clCreateBuffer failed (err {e})"));
                }
                $ptr = m;
            }};
        }

        mkbuf!(cl_in, CL_MEM_READ_WRITE, total_in.max(1));
        mkbuf!(cl_out, CL_MEM_READ_WRITE, total_in.max(1));
        mkbuf!(cl_out_len, CL_MEM_READ_WRITE, n * 4);
        mkbuf!(cl_in_off, CL_MEM_READ_ONLY, n * 4);
        mkbuf!(cl_out_off, CL_MEM_READ_ONLY, n * 4);
        mkbuf!(cl_block_len, CL_MEM_READ_ONLY, n * 4);
        mkbuf!(cl_chain_off, CL_MEM_READ_ONLY, n * 4);
        mkbuf!(cl_head_off, CL_MEM_READ_ONLY, n * 4);
        mkbuf!(cl_slots, CL_MEM_READ_ONLY, n * 4);
        mkbuf!(cl_chain, CL_MEM_READ_WRITE, (total_chain.max(1)) * 4);
        mkbuf!(cl_head, CL_MEM_READ_WRITE, (total_head.max(1)) * 4);

        // Helper: write a host slice to a cl_mem and release on error.
        unsafe fn upload(
            api: &Api,
            queue: ClCommandQueue,
            mem: ClMem,
            data: *const c_void,
            bytes: usize,
        ) -> Result<(), String> {
            let e = (api.enqueue_write_buffer)(
                queue,
                mem,
                0,
                0,
                bytes,
                data,
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            );
            if e != CL_SUCCESS {
                return Err(format!("clEnqueueWriteBuffer failed (err {e})"));
            }
            Ok(())
        }
        unsafe fn check(api: &Api, queue: ClCommandQueue) -> Result<(), String> {
            let e = (api.finish)(queue);
            if e != CL_SUCCESS {
                return Err(format!("clFinish failed (err {e})"));
            }
            Ok(())
        }

        unsafe {
            upload(
                api,
                queue,
                cl_in,
                in_buf.as_ptr() as *const c_void,
                total_in.max(1),
            )?;
            upload(
                api,
                queue,
                cl_in_off,
                in_off.as_ptr() as *const c_void,
                n * 4,
            )?;
            upload(
                api,
                queue,
                cl_out_off,
                out_off.as_ptr() as *const c_void,
                n * 4,
            )?;
            upload(
                api,
                queue,
                cl_block_len,
                block_len.as_ptr() as *const c_void,
                n * 4,
            )?;
            upload(
                api,
                queue,
                cl_chain_off,
                chain_off.as_ptr() as *const c_void,
                n * 4,
            )?;
            upload(
                api,
                queue,
                cl_head_off,
                head_off.as_ptr() as *const c_void,
                n * 4,
            )?;
            upload(
                api,
                queue,
                cl_slots,
                slots_arr.as_ptr() as *const c_void,
                n * 4,
            )?;
            check(api, queue)?;
        }

        // Set kernel args.
        macro_rules! arg_buf {
            ($idx:expr, $mem:expr) => {{
                let e = unsafe {
                    (api.set_kernel_arg)(
                        kernel,
                        $idx,
                        std::mem::size_of::<ClMem>(),
                        &$mem as *const ClMem as *const c_void,
                    )
                };
                if e != CL_SUCCESS {
                    return Err(format!("clSetKernelArg({}) failed (err {e})", $idx));
                }
            }};
        }
        macro_rules! arg_u32 {
            ($idx:expr, $val:expr) => {{
                let v: u32 = $val;
                let e = unsafe {
                    (api.set_kernel_arg)(kernel, $idx, 4, &v as *const u32 as *const c_void)
                };
                if e != CL_SUCCESS {
                    return Err(format!("clSetKernelArg({}) failed (err {e})", $idx));
                }
            }};
        }

        arg_buf!(0, cl_in);
        arg_buf!(1, cl_out);
        arg_buf!(2, cl_out_len);
        arg_buf!(3, cl_in_off);
        arg_buf!(4, cl_out_off);
        arg_buf!(5, cl_block_len);
        arg_buf!(6, cl_chain_off);
        arg_buf!(7, cl_head_off);
        arg_buf!(8, cl_slots);
        arg_buf!(9, cl_chain);
        arg_buf!(10, cl_head);
        arg_u32!(11, effort.max_chain as u32);
        arg_u32!(12, effort.max_match as u32);
        arg_u32!(13, effort.window as u32);
        arg_u32!(14, if effort.lazy { 1 } else { 0 });
        arg_u32!(15, n as u32);

        // Launch: one work-item per block; let the driver choose local size.
        let global = n;
        let e = unsafe {
            (api.enqueue_nd_range_kernel)(
                queue,
                kernel,
                1,
                std::ptr::null(),
                &global,
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
        if e != CL_SUCCESS {
            return Err(format!("clEnqueueNDRangeKernel failed (err {e})"));
        }

        // Read back results.
        unsafe {
            let e = (api.enqueue_read_buffer)(
                queue,
                cl_out_len,
                0,
                0,
                n * 4,
                out_len_arr.as_mut_ptr() as *mut c_void,
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            );
            if e != CL_SUCCESS {
                return Err(format!("clEnqueueReadBuffer(out_len) failed (err {e})"));
            }
            let e = (api.enqueue_read_buffer)(
                queue,
                cl_out,
                0,
                0,
                total_in.max(1),
                out_buf.as_mut_ptr() as *mut c_void,
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            );
            if e != CL_SUCCESS {
                return Err(format!("clEnqueueReadBuffer(out) failed (err {e})"));
            }
            check(api, queue)?;
        }

        // Release per-batch buffers.
        unsafe {
            for m in [
                cl_in,
                cl_out,
                cl_out_len,
                cl_in_off,
                cl_out_off,
                cl_block_len,
                cl_chain_off,
                cl_head_off,
                cl_slots,
                cl_chain,
                cl_head,
            ] {
                (api.release_mem_object)(m);
            }
        }

        // Slice the concatenated output back into per-block results.
        let mut results = Vec::with_capacity(n);
        for i in 0..n {
            let olen = out_len_arr[i] as usize;
            let off = out_off[i] as usize;
            let blen = block_len[i] as usize;
            if olen == 0 || olen >= blen || off + olen > out_buf.len() {
                results.push(None);
            } else {
                results.push(Some(out_buf[off..off + olen].to_vec()));
            }
        }
        Ok(results)
    }
}

impl Drop for OpenCl {
    fn drop(&mut self) {
        unsafe {
            (self.api.release_kernel)(self.kernel);
            (self.api.release_program)(self.program);
            (self.api.release_command_queue)(self.queue);
            (self.api.release_context)(self.context);
        }
    }
}

/// Hash-table slot count for a block of `len` bytes, matching the CPU codec.
fn slots_for(len: usize) -> usize {
    (len / 4).next_power_of_two().clamp(1 << 10, 1 << 21)
}

// ---------------------------------------------------------------------------
// OpenCL C kernel — byte-exact port of `lz::compress`
// ---------------------------------------------------------------------------
const KERNEL_SOURCE: &str = r#"
#define MIN_MATCH 4u
#define NICE_MATCH 64u
#define NIBBLE_MAX 15u
#define NONE 0xFFFFFFFFu

inline uint key_at(__global const uchar* data, uint pos) {
    return (uint)data[pos]
         | ((uint)data[pos+1] << 8)
         | ((uint)data[pos+2] << 16)
         | ((uint)data[pos+3] << 24);
}

inline uint hash_key(uint key) {
    ulong k = (ulong)key * 0x9E3779B1UL;
    return (uint)(k >> 16);
}

inline uint varint_len(ulong v) {
    uint n = 1u;
    v >>= 7;
    while (v != 0) { n++; v >>= 7; }
    return n;
}

/* Write a LEB128 varint; returns new pos or (uint)-1 on overflow. */
inline uint emit_varint(__global uchar* out, uint pos, uint cap, ulong v) {
    while (v >= 0x80) {
        if (pos >= cap) return (uint)-1;
        out[pos++] = (uchar)((v & 0x7F) | 0x80);
        v >>= 7;
    }
    if (pos >= cap) return (uint)-1;
    out[pos++] = (uchar)v;
    return pos;
}

inline uint emit_literals(__global uchar* out, uint out_pos, uint cap,
                          __global const uchar* data, uint from, uint to) {
    uint len = to - from;
    if (len == 0) return out_pos;
    uint nibble, extra, has_extra;
    if (len < NIBBLE_MAX) { nibble = len; has_extra = 0u; extra = 0u; }
    else { nibble = NIBBLE_MAX; has_extra = 1u; extra = len - NIBBLE_MAX; }
    if (out_pos >= cap) return (uint)-1;
    out[out_pos++] = (uchar)nibble;
    if (has_extra) {
        out_pos = emit_varint(out, out_pos, cap, (ulong)extra);
        if (out_pos == (uint)-1) return (uint)-1;
    }
    if ((ulong)out_pos + len > cap) return (uint)-1;
    for (uint i = 0; i < len; i++) out[out_pos + i] = data[from + i];
    return out_pos + len;
}

inline uint emit_match(__global uchar* out, uint out_pos, uint cap, uint mlen, uint moff,
                       uint* last_off, uint* prev_off) {
    if (moff == *last_off || (moff == *prev_off && *prev_off != 0u)) {
        /* Repeat-offset form: token 0 plus one varint packing length and slot. */
        uint which = (moff != *last_off) ? 1u : 0u;
        if (which == 1u) {
            uint t = *last_off; *last_off = *prev_off; *prev_off = t;
        } else {
            *prev_off = *last_off;
        }
        if (out_pos >= cap) return (uint)-1;
        out[out_pos++] = 0u;
        out_pos = emit_varint(out, out_pos, cap,
                              (ulong)((mlen - MIN_MATCH) * 2u + which));
        return out_pos;
    }
    *prev_off = *last_off;
    *last_off = moff;
    uint lm = mlen - MIN_MATCH;
    uint len_nibble, len_extra, has_len_extra;
    uint off_nibble, off_extra, has_off_extra;
    if (lm < NIBBLE_MAX) { len_nibble = lm; has_len_extra = 0u; len_extra = 0u; }
    else { len_nibble = NIBBLE_MAX; has_len_extra = 1u; len_extra = lm - NIBBLE_MAX; }
    if (moff < NIBBLE_MAX) { off_nibble = moff; has_off_extra = 0u; off_extra = 0u; }
    else { off_nibble = NIBBLE_MAX; has_off_extra = 1u; off_extra = moff - NIBBLE_MAX; }
    if (out_pos >= cap) return (uint)-1;
    out[out_pos++] = (uchar)((off_nibble << 4) | len_nibble);
    if (has_off_extra) {
        out_pos = emit_varint(out, out_pos, cap, (ulong)off_extra);
        if (out_pos == (uint)-1) return (uint)-1;
    }
    if (has_len_extra) {
        out_pos = emit_varint(out, out_pos, cap, (ulong)len_extra);
        if (out_pos == (uint)-1) return (uint)-1;
    }
    return out_pos;
}

inline uint extension_len(uint mlen) {
    uint lm = mlen - MIN_MATCH;
    if (lm < NIBBLE_MAX) return 0u;
    return varint_len((ulong)(lm - NIBBLE_MAX));
}

inline uint saving(uint mlen, uint moff) {
    uint cost = 1u + varint_len((ulong)moff) + extension_len(mlen);
    return mlen > cost ? mlen - cost : 0u;
}

__kernel void compress_blocks(
    __global const uchar* input,
    __global uchar* output,
    __global uint* out_len,
    __global const uint* in_off,
    __global const uint* out_off,
    __global const uint* block_len,
    __global const uint* chain_off,
    __global const uint* head_off,
    __global const uint* slots_arr,
    __global uint* chain,
    __global uint* head,
    uint max_chain,
    uint max_match,
    uint window,
    uint lazy,
    uint nblocks
) {
    uint bid = get_global_id(0);
    if (bid >= nblocks) return;

    uint ioff = in_off[bid];
    uint ooff = out_off[bid];
    uint len = block_len[bid];
    uint coff = chain_off[bid];
    uint hoff = head_off[bid];
    uint slots = slots_arr[bid];

    __global const uchar* data = input + ioff;
    __global uchar* out = output + ooff;
    uint cap = len;

    out_len[bid] = 0u;
    if (max_chain == 0u || len < MIN_MATCH * 2u) return;

    uint table_mask = slots - 1u;
    for (uint s = 0; s < slots; s++) head[hoff + s] = NONE;

    uint cursor = 0u;
    uint indexed = 0u;
    uint anchor = 0u;
    int pending_pos = -1;
    uint pending_len = 0u, pending_off = 0u;
    uint out_pos = 0u;
    uint last_off = 0u;
    uint prev_off = 0u;

    while (cursor + MIN_MATCH <= len) {
        uint pos = cursor++;

        /* find at pos, BEFORE indexing this position */
        int found = 0;
        uint found_len = 0u, found_off = 0u;
        if (pos >= anchor && pos + MIN_MATCH <= len) {
            uint max_len = max_match;
            if (max_len > len - pos) max_len = len - pos;
            if (max_len >= MIN_MATCH) {
                uint floor = pos > window ? pos - window : 0u;
                uint candidate = head[hoff + (hash_key(key_at(data, pos)) & table_mask)];
                uint chain_count = 0u;
                uint best_len = 0u, best_off = 0u;
                int has_best = 0;

                while (candidate != NONE && chain_count < max_chain) {
                    uint c = candidate;
                    chain_count++;
                    if (c >= pos) break;
                    candidate = chain[coff + c];
                    if (c < floor) break;

                    /* The bucket is keyed by a hash of the anchor word, so a
                       candidate may differ in its very first byte. Verify the
                       whole word, exactly like the CPU finder. */
                    uint a = (uint)data[c] | ((uint)data[c+1] << 8) | ((uint)data[c+2] << 16) | ((uint)data[c+3] << 24);
                    uint b = (uint)data[pos] | ((uint)data[pos+1] << 8) | ((uint)data[pos+2] << 16) | ((uint)data[pos+3] << 24);
                    if (a != b) continue;
                    uint ml = MIN_MATCH;
                    while (ml + 8u <= max_len) {
                        ulong a = 0UL, b = 0UL;
                        for (int j = 0; j < 8; j++) {
                            a |= ((ulong)data[c + ml + j]) << (8 * j);
                            b |= ((ulong)data[pos + ml + j]) << (8 * j);
                        }
                        if (a != b) break;
                        ml += 8u;
                    }
                    while (ml < max_len && data[c + ml] == data[pos + ml]) ml++;

                    if (ml < MIN_MATCH) continue;
                    uint off = pos - c;
                    if (off == 0u) continue; /* offset must be >= 1 */

                    int better = 0;
                    if (!has_best) better = 1;
                    else if (ml > best_len || (ml == best_len && off < best_off)) better = 1;
                    if (better) {
                        has_best = 1; best_len = ml; best_off = off;
                        if (ml >= max_len || ml >= NICE_MATCH) break;
                    }
                }

                if (has_best) {
                    uint cost = 1u + varint_len((ulong)best_off) + extension_len(best_len);
                    if (cost < best_len) {
                        found = 1; found_len = best_len; found_off = best_off;
                    }
                }
            }
        }

        /* index up to cursor (exclusive of current pos already handled above) */
        while (indexed < cursor) {
            uint ip = indexed++;
            if (ip + MIN_MATCH <= len) {
                uint slot = hash_key(key_at(data, ip)) & table_mask;
                chain[coff + ip] = head[hoff + slot];
                head[hoff + slot] = ip;
            }
        }

        if (pos < anchor) {
            /* unreachable per CPU debug_assert; pending must be none */
            continue;
        }

        if (!lazy) {
            if (!found) continue;
            out_pos = emit_literals(out, out_pos, cap, data, anchor, pos);
            if (out_pos == (uint)-1) { out_len[bid] = 0u; return; }
            out_pos = emit_match(out, out_pos, cap, found_len, found_off, &last_off, &prev_off);
            if (out_pos == (uint)-1) { out_len[bid] = 0u; return; }
            anchor = pos + found_len;
            continue;
        }

        /* lazy parse */
        if (pending_pos < 0) {
            if (found) {
                pending_pos = (int)pos;
                pending_len = found_len;
                pending_off = found_off;
            }
        } else {
            uint ppos = (uint)pending_pos;
            if (!found) {
                out_pos = emit_literals(out, out_pos, cap, data, anchor, ppos);
                if (out_pos == (uint)-1) { out_len[bid] = 0u; return; }
                out_pos = emit_match(out, out_pos, cap, pending_len, pending_off, &last_off, &prev_off);
                if (out_pos == (uint)-1) { out_len[bid] = 0u; return; }
                anchor = ppos + pending_len;
                pending_pos = -1;
            } else if (pos == ppos + 1u) {
                uint s_new = saving(found_len, found_off);
                uint s_old = saving(pending_len, pending_off);
                if (s_new > s_old) {
                    pending_pos = (int)pos;
                    pending_len = found_len;
                    pending_off = found_off;
                } else {
                    out_pos = emit_literals(out, out_pos, cap, data, anchor, ppos);
                    if (out_pos == (uint)-1) { out_len[bid] = 0u; return; }
                    out_pos = emit_match(out, out_pos, cap, pending_len, pending_off, &last_off, &prev_off);
                    if (out_pos == (uint)-1) { out_len[bid] = 0u; return; }
                    anchor = ppos + pending_len;
                    pending_pos = -1;
                }
            } else {
                /* non-consecutive: emit pending, drop the new match */
                out_pos = emit_literals(out, out_pos, cap, data, anchor, ppos);
                if (out_pos == (uint)-1) { out_len[bid] = 0u; return; }
                out_pos = emit_match(out, out_pos, cap, pending_len, pending_off, &last_off, &prev_off);
                if (out_pos == (uint)-1) { out_len[bid] = 0u; return; }
                anchor = ppos + pending_len;
                pending_pos = -1;
            }
        }
    }

    /* flush trailing pending */
    if (pending_pos >= 0) {
        uint ppos = (uint)pending_pos;
        out_pos = emit_literals(out, out_pos, cap, data, anchor, ppos);
        if (out_pos == (uint)-1) { out_len[bid] = 0u; return; }
        out_pos = emit_match(out, out_pos, cap, pending_len, pending_off, &last_off, &prev_off);
        if (out_pos == (uint)-1) { out_len[bid] = 0u; return; }
        anchor = ppos + pending_len;
    }

    /* trailing literals to end of block */
    out_pos = emit_literals(out, out_pos, cap, data, anchor, len);
    if (out_pos == (uint)-1) { out_len[bid] = 0u; return; }

    /* decline if output not smaller (same rule as CPU) */
    if (out_pos >= len) { out_len[bid] = 0u; return; }
    out_len[bid] = out_pos;
}
"#;

#[cfg(test)]
mod tests {
    use super::GpuBackend;
    use super::*;
    use crate::lz;

    fn prose(len: usize) -> Vec<u8> {
        const WORDS: [&str; 12] = [
            "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "and", "then", "runs",
            "away",
        ];
        let mut s = 7u64;
        let mut out = Vec::with_capacity(len);
        while out.len() < len {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let w = WORDS[((s >> 33) as usize) % WORDS.len()];
            out.extend_from_slice(w.as_bytes());
            out.push(b' ');
        }
        out.truncate(len);
        out
    }

    #[test]
    fn gpu_matches_cpu_on_small_blocks() {
        let mut gpu = match Gpu::open() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping gpu_matches_cpu: {e}");
                return;
            }
        };
        eprintln!("gpu device: {}", gpu.device_name());

        let cases: Vec<Vec<u8>> = vec![prose(50_000), prose(100_000), vec![b'z'; 30_000], {
            let mut s = 42u64;
            (0..40_000)
                .map(|_| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (s >> 33) as u8
                })
                .collect()
        }];

        for data in &cases {
            for level in [1u8, 4, 7, 9] {
                let gpu_out = gpu.compress_batch(&[data.as_slice()], level).unwrap();
                let cpu_out = lz::compress(data, level);
                match (&gpu_out[0], &cpu_out) {
                    (Some(g), Some(c)) => {
                        assert_eq!(
                            g,
                            c,
                            "level {level}, {}-byte block: GPU and CPU output differ",
                            data.len()
                        );
                    }
                    (None, None) => {}
                    (g, c) => panic!(
                        "level {level}, {}-byte block: GPU={g:?} CPU={c:?}",
                        data.len()
                    ),
                }
            }
        }
    }

    #[test]
    fn gpu_batch_matches_cpu_for_multiple_blocks() {
        let mut gpu = match Gpu::open() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping gpu_batch test: {e}");
                return;
            }
        };
        let a = prose(20_000);
        let b = vec![b'q'; 15_000];
        let c = prose(8_000);
        let blocks: Vec<&[u8]> = vec![a.as_slice(), b.as_slice(), c.as_slice()];
        let gpu_out = gpu.compress_batch(&blocks, 6).unwrap();
        assert_eq!(gpu_out.len(), 3);
        for (i, block) in blocks.iter().enumerate() {
            let cpu = lz::compress(block, 6);
            match (&gpu_out[i], &cpu) {
                (Some(g), Some(c)) => assert_eq!(g, c, "block {i} differs"),
                (None, None) => {}
                (g, c) => panic!("block {i}: GPU={g:?} CPU={c:?}"),
            }
        }
    }

    #[test]
    fn level_zero_declines_everything_on_gpu() {
        let mut gpu = match Gpu::open() {
            Ok(g) => g,
            Err(_) => return,
        };
        let data = prose(10_000);
        let out = gpu.compress_batch(&[data.as_slice()], 0).unwrap();
        assert!(out[0].is_none(), "level 0 must decline");
    }

    #[test]
    fn tiny_blocks_decline() {
        let mut gpu = match Gpu::open() {
            Ok(g) => g,
            Err(_) => return,
        };
        for len in [0usize, 1, 4, 7] {
            let data = prose(len.max(1)).into_iter().take(len).collect::<Vec<u8>>();
            let out = gpu.compress_batch(&[data.as_slice()], 7).unwrap();
            assert!(out[0].is_none(), "block of {len} bytes must decline");
        }
    }
}
