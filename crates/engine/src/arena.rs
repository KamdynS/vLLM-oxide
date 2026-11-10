//! Bump arena over the backend's activation workspace (grill Q9, ADR 0003).
//!
//! **The idea.** The backend allocates one workspace of `capacity` f32 at
//! startup. The [`Arena`] hands out pieces of it by moving a `top` offset up
//! ("bump allocation"). A [`Scope`] remembers `top` when it opens and puts it
//! back when it closes, which frees everything allocated inside it at once.
//! Nothing is allocated on the device after startup.
//!
//! **Using it.** Open one scope per model step with [`Arena::scope`], nest a
//! scope per block with [`Scope::scope`], and call ops (`matmul`, `add`, ...)
//! on the scope. Each op allocates its output tensor in the scope it's called
//! on, so where you call an op decides how long its result lives.
//!
//! **Why it's safe.** The rules are enforced by the compiler, not by care:
//! - A [`Tensor<'scope>`] can't outlive its scope. [`Arena::scope`] and
//!   [`Scope::scope`] take a closure that must work for *every* lifetime
//!   `'scope`, so it can't return or stash a tensor tied to that scope.
//!   Returning a layer-local tensor out of the layer is a compile error, not
//!   a use-after-free (see the example below).
//! - An outer scope can't allocate while an inner one is open, because the
//!   inner scope borrows the outer one `&mut`. So closing the inner scope never
//!   frees an outer tensor.
//! - `Tensor` isn't `Clone`: each live region has exactly one handle, so an
//!   in-place op (which takes `&mut Tensor`) can't also get the same region as
//!   an input.
//! - Each tensor records which arena it came from, checked on every op, so a
//!   tensor from another arena can't be used to alias this one's memory.
//!
//! This is the "residual stream allocated inside a layer scope" bug, and it
//! does not build:
//!
//! ```compile_fail
//! use engine::arena::Arena;
//! use engine::{Backend, CpuBackend, LLAMA_3_2_1B, Weight};
//! let mut backend = CpuBackend::new(&LLAMA_3_2_1B, 1);
//! backend.alloc_workspace(64).unwrap();
//! let table = backend.upload(Weight::new(vec![2, 4], vec![0.0; 8]).unwrap()).unwrap();
//! let mut arena = Arena::new(&backend);
//! arena.scope(&backend, |step| {
//!     // error: a tensor allocated in the inner scope can't be returned from it.
//!     let residual = step.scope(|layer| layer.embed(&table, &[0]).unwrap());
//!     step.read(&residual).unwrap();
//! });
//! ```
//!
//! Allocating it in the outer scope and adding into it from the inner scope
//! builds:
//!
//! ```
//! # use engine::arena::Arena;
//! # use engine::{Backend, CpuBackend, LLAMA_3_2_1B, Weight};
//! # let mut backend = CpuBackend::new(&LLAMA_3_2_1B, 1);
//! # backend.alloc_workspace(64).unwrap();
//! # let table = backend.upload(Weight::new(vec![2, 4], vec![0.0; 8]).unwrap()).unwrap();
//! # let mut arena = Arena::new(&backend);
//! arena.scope(&backend, |step| {
//!     let mut residual = step.embed(&table, &[0]).unwrap();
//!     step.scope(|layer| {
//!         let layer_output = layer.embed(&table, &[1]).unwrap();
//!         layer.add(&mut residual, &layer_output).unwrap();
//!     });
//!     step.read(&residual).unwrap();
//! });
//! ```
//!
//! **Observability.** Each op runs inside a `tracing` span. Per-op timing is
//! opt-in (grill Q8): [`Arena::enable_profiling`]. CPU ops are synchronous,
//! so wall-clock timing is accurate; on CUDA this must become event-based (M1).

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::backend::{Backend, EngineError, Region, Result};

mod ops;

static NEXT_ARENA_ID: AtomicU64 = AtomicU64::new(1);

/// Per-op timing totals, collected only when profiling is enabled.
#[derive(Debug, Default, Clone, Copy)]
pub struct OpStat {
    pub calls: u64,
    pub total: Duration,
}

/// Offset bookkeeping for one workspace. Holds no memory itself: the backend
/// owns the workspace, the arena only decides which offsets are in use.
#[derive(Debug)]
pub struct Arena {
    id: u64,
    capacity: usize,
    /// First free element. Everything below is in use by an open scope.
    top: Cell<usize>,
    /// Highest `top` ever reached. Checked against `model::arena_elems`.
    high_water: Cell<usize>,
    /// Per-op timing totals, keyed by op name. `None` when profiling is off,
    /// so the hot path pays one branch. `RefCell` because every `Scope` holds
    /// the arena by `&` (see `top`), so updating the map needs interior
    /// mutability; a map isn't `Copy`, so `Cell` can't hold it.
    profile: Option<RefCell<BTreeMap<&'static str, OpStat>>>,
}

impl Arena {
    /// An arena over `backend`'s workspace. The capacity is read from the
    /// backend, not passed in, so it can't disagree with the workspace size
    /// (the "in bounds" promise of the backend's unsafe ops relies on that).
    /// Call after `backend.alloc_workspace`.
    pub fn new<B: Backend>(backend: &B) -> Self {
        Arena {
            id: NEXT_ARENA_ID.fetch_add(1, Ordering::Relaxed),
            capacity: backend.workspace_elems(),
            top: Cell::new(0),
            high_water: Cell::new(0),
            profile: None,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn high_water(&self) -> usize {
        self.high_water.get()
    }

    pub fn enable_profiling(&mut self) {
        self.profile = Some(RefCell::new(BTreeMap::new()));
    }

    /// Per-op timing totals, if profiling is enabled.
    pub fn profile(&self) -> Option<BTreeMap<&'static str, OpStat>> {
        self.profile.as_ref().map(|stats| stats.borrow().clone())
    }

    /// Open the outermost scope (one model step). Everything allocated inside
    /// is freed when `body` returns. `&mut self` means only one step scope
    /// can be open at a time.
    pub fn scope<'a, B: Backend, R>(
        &'a mut self,
        backend: &'a B,
        body: impl for<'scope> FnOnce(&mut Scope<'scope, B>) -> R,
    ) -> R {
        // Catches passing a different backend than the arena was built from.
        debug_assert!(self.capacity <= backend.workspace_elems());
        let mark = self.top.get();
        let mut scope = Scope {
            arena: &*self,
            backend,
            _lifetime: PhantomData,
        };
        let result = body(&mut scope);
        self.top.set(mark);
        result
    }
}

/// An activation: a region of the workspace, valid while scope `'scope` is open.
#[derive(Debug)]
pub struct Tensor<'scope> {
    region: Region,
    arena_id: u64,
    _lifetime: PhantomData<&'scope ()>,
}

impl Tensor<'_> {
    pub fn rows(&self) -> usize {
        self.region.rows
    }
    pub fn cols(&self) -> usize {
        self.region.cols
    }
    pub fn shape(&self) -> [usize; 2] {
        [self.region.rows, self.region.cols]
    }
}

/// An open arena scope. Ops called on it allocate their outputs in it.
pub struct Scope<'scope, B: Backend> {
    arena: &'scope Arena,
    backend: &'scope B,
    // Makes `'scope` invariant: a Scope can't be coerced to a longer or
    // shorter lifetime, so tensors can't be smuggled between scopes.
    _lifetime: PhantomData<fn(&'scope ()) -> &'scope ()>,
}

impl<'scope, B: Backend> Scope<'scope, B> {
    /// Open a nested scope. Its tensors are freed when `body` returns; tensors
    /// of this (outer) scope stay valid and usable inside `body`.
    pub fn scope<R>(&mut self, body: impl for<'inner> FnOnce(&mut Scope<'inner, B>) -> R) -> R {
        let mark = self.arena.top.get();
        let mut inner = Scope {
            arena: self.arena,
            backend: self.backend,
            _lifetime: PhantomData,
        };
        let result = body(&mut inner);
        self.arena.top.set(mark);
        result
    }

    pub fn backend(&self) -> &'scope B {
        self.backend
    }

    /// Reserve `[rows, cols]` at `top`. The contents are whatever was there
    /// before; every op overwrites its whole output.
    fn alloc(&self, rows: usize, cols: usize) -> Result<Tensor<'scope>> {
        let arena = self.arena;
        let top = arena.top.get();
        let need = rows * cols;
        if top + need > arena.capacity {
            return Err(EngineError::ArenaExhausted {
                need,
                top,
                capacity: arena.capacity,
            });
        }
        arena.top.set(top + need);
        arena.high_water.set(arena.high_water.get().max(top + need));
        Ok(Tensor {
            region: Region {
                offset: top,
                rows,
                cols,
            },
            arena_id: arena.id,
            _lifetime: PhantomData,
        })
    }

    /// The tensor's region, after checking it came from this arena.
    fn region(&self, tensor: &Tensor<'_>) -> Result<Region> {
        if tensor.arena_id != self.arena.id {
            return Err(EngineError::WrongArena);
        }
        Ok(tensor.region)
    }

    /// Run one op inside a tracing span, timing it if profiling is on.
    fn op<T>(&self, name: &'static str, run: impl FnOnce() -> Result<T>) -> Result<T> {
        let _span = tracing::trace_span!("op", name).entered();
        let Some(profile) = &self.arena.profile else {
            return run();
        };
        let started = Instant::now();
        let result = run();
        let mut profile = profile.borrow_mut();
        let stat = profile.entry(name).or_default();
        stat.calls += 1;
        stat.total += started.elapsed();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::weights::Weight;

    /// A backend with a `capacity`-element workspace and a 4×10 table to
    /// `embed` from (each embedded row allocates 10 elements).
    fn setup(capacity: usize) -> (CpuBackend, Weight) {
        let mut backend = CpuBackend::new(&crate::config::LLAMA_3_2_1B, 16);
        backend.alloc_workspace(capacity).unwrap();
        let table = Weight::new(vec![4, 10], vec![0.5; 40]).unwrap();
        (backend, table)
    }

    #[test]
    fn closing_a_scope_frees_its_tensors() {
        let (backend, table) = setup(100);
        let mut arena = Arena::new(&backend);
        arena.scope(&backend, |step| {
            let outer = step.embed(&table, &[0, 1, 2]).unwrap(); // top 30
            step.scope(|inner| {
                let _temp = inner.embed(&table, &[0; 5]).unwrap(); // top 80
            });
            // The inner scope's 50 are free again: 70 more fit.
            let _more = step.embed(&table, &[0; 7]).unwrap(); // top 100
            assert_eq!(outer.rows(), 3);
        });
        assert_eq!(arena.high_water(), 100);
        // The step scope freed everything: the whole arena is available.
        arena.scope(&backend, |step| {
            step.embed(&table, &[0; 10]).unwrap();
        });
    }

    #[test]
    fn running_out_is_an_error() {
        let (backend, table) = setup(25);
        let mut arena = Arena::new(&backend);
        arena.scope(&backend, |step| {
            step.embed(&table, &[0, 1]).unwrap();
            let err = step.embed(&table, &[0]).unwrap_err();
            assert_eq!(
                err,
                EngineError::ArenaExhausted {
                    need: 10,
                    top: 20,
                    capacity: 25
                }
            );
        });
    }

    #[test]
    fn tensor_from_another_arena_is_rejected() {
        let (backend, table) = setup(100);
        let mut first = Arena::new(&backend);
        let mut second = Arena::new(&backend);
        first.scope(&backend, |step_a| {
            let x = step_a.embed(&table, &[1]).unwrap();
            second.scope(&backend, |step_b| {
                assert_eq!(step_b.read(&x).unwrap_err(), EngineError::WrongArena);
            });
        });
    }

    #[test]
    fn profiling_counts_ops() {
        let (backend, table) = setup(100);
        let mut arena = Arena::new(&backend);
        arena.enable_profiling();
        arena.scope(&backend, |step| {
            step.embed(&table, &[0]).unwrap();
            step.embed(&table, &[1]).unwrap();
        });
        assert_eq!(arena.profile().unwrap()["embed"].calls, 2);
    }
}
