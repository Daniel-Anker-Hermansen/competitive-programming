#![allow(unused)]
mod tools {
    use std::{
        cmp::Reverse,
        collections::{BinaryHeap, HashMap},
        fmt::Debug,
        hash::{Hash, Hasher, BuildHasher},
        iter::{repeat, Peekable},
        marker::PhantomData,
        ops::{Deref, Index, IndexMut, Range, RangeBounds},
    };
    pub mod input {
        use std::{
            collections::{BTreeSet, HashSet, VecDeque},
            fmt::Debug,
            hash::Hash,
            io::{BufRead, Lines, Stdin, StdinLock},
            iter::{FromIterator, Map},
            str::FromStr,
        };

        trait LocalFromStr {
            fn local_from_str(s: &str) -> Self;
        }

        macro_rules! impl_local_from_str {
            ($x:ty) => {
                impl LocalFromStr for $x {
                    fn local_from_str(s: &str) -> $x {
                        <$x>::from_str(s).unwrap()
                    }
                }
            };
            ($($x:ty) *) => {
                $(
                    impl_local_from_str!($x);
                )*
            };
        }

        impl_local_from_str!(bool char f32 f64 i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize String
            std::num::NonZeroI8 std::num::NonZeroI16 std::num::NonZeroI32 std::num::NonZeroI64 std::num::NonZeroI128 std::num::NonZeroIsize
            std::num::NonZeroU8 std::num::NonZeroU16 std::num::NonZeroU32 std::num::NonZeroU64 std::num::NonZeroU128 std::num::NonZeroUsize);

        impl<T> LocalFromStr for Vec<T>
        where T: LocalFromStr
        {
            fn local_from_str(s: &str) -> Self {
                yuck(s)
            }
        }

        impl<T> LocalFromStr for VecDeque<T>
        where T: LocalFromStr
        {
            fn local_from_str(s: &str) -> Self {
                yuck(s)
            }
        }

        impl<T> LocalFromStr for HashSet<T>
        where T: LocalFromStr + Hash + Eq
        {
            fn local_from_str(s: &str) -> Self {
                yuck(s)
            }
        }

        impl<T> LocalFromStr for BTreeSet<T>
        where T: LocalFromStr + Ord
        {
            fn local_from_str(s: &str) -> Self {
                yuck(s)
            }
        }

        impl<A, B> LocalFromStr for (A, B)
        where
            A: LocalFromStr,
            B: LocalFromStr,
        {
            fn local_from_str(s: &str) -> Self {
                let mut iter = &mut s.split_ascii_whitespace();
                (k(iter), k(iter))
            }
        }

        impl<A, B, C> LocalFromStr for (A, B, C)
        where
            A: LocalFromStr,
            B: LocalFromStr,
            C: LocalFromStr,
        {
            fn local_from_str(s: &str) -> Self {
                let mut iter = &mut s.split_ascii_whitespace();
                (k(iter), k(iter), k(iter))
            }
        }

        impl<A, B, C, D> LocalFromStr for (A, B, C, D)
        where
            A: LocalFromStr,
            B: LocalFromStr,
            C: LocalFromStr,
            D: LocalFromStr,
        {
            fn local_from_str(s: &str) -> Self {
                let mut iter = &mut s.split_ascii_whitespace();
                (k(iter), k(iter), k(iter), k(iter))
            }
        }

        impl<A, B, C, D, E> LocalFromStr for (A, B, C, D, E)
        where
            A: LocalFromStr,
            B: LocalFromStr,
            C: LocalFromStr,
            D: LocalFromStr,
            E: LocalFromStr,
        {
            fn local_from_str(s: &str) -> Self {
                let mut iter = &mut s.split_ascii_whitespace();
                (k(iter), k(iter), k(iter), k(iter), k(iter))
            }
        }

        impl<A, B, C, D, E, F> LocalFromStr for (A, B, C, D, E, F)
        where
            A: LocalFromStr,
            B: LocalFromStr,
            C: LocalFromStr,
            D: LocalFromStr,
            E: LocalFromStr,
            F: LocalFromStr,
        {
            fn local_from_str(s: &str) -> Self {
                let mut iter = &mut s.split_ascii_whitespace();
                (k(iter), k(iter), k(iter), k(iter), k(iter), k(iter))
            }
        }

        impl<A, B, C, D, E, F, G, H> LocalFromStr for (A, B, C, D, E, F, G, H)
        where
            A: LocalFromStr,
            B: LocalFromStr,
            C: LocalFromStr,
            D: LocalFromStr,
            E: LocalFromStr,
            F: LocalFromStr,
            G: LocalFromStr,
            H: LocalFromStr,
        {
            fn local_from_str(s: &str) -> Self {
                let mut iter = &mut s.split_ascii_whitespace();
                (k(iter), k(iter), k(iter), k(iter), k(iter), k(iter), k(iter), k(iter))
            }
        }

        fn k<'a, T, I, S>(iter: &mut I) -> T
        where
            T: LocalFromStr,
            I: Iterator<Item = S>,
            S: Into<&'a str>, {
            T::local_from_str(iter.next().unwrap().into())
        }

        fn yuck<C, T>(s: &str) -> C
        where
            C: FromIterator<T>,
            T: LocalFromStr, {
            s.split_ascii_whitespace()
                .map(T::local_from_str)
                .collect()
        }

        pub trait FromStringKattis {
            fn kattis_from_str(s: &str) -> Self;
        }

        impl<T> FromStringKattis for T
        where T: LocalFromStr
        {
            fn kattis_from_str(s: &str) -> Self {
                T::local_from_str(s)
            }
        }

        /// A struct to read input from stdin to make working with Kattis easier.
        pub struct Input {
            read: Lines<StdinLock<'static>>,
        }

        impl Input {
            #[inline(always)]
            pub fn new() -> Input {
                let read = std::io::stdin().lock().lines();
                Input {
                    read,
                }
            }

            /// Reads the next line from stdin and converts it to type T. Panics if an io-error happens or the conversion fails.
            #[inline(always)]
            pub fn next<T>(&mut self) -> T
            where T: FromStringKattis {
                T::kattis_from_str(self.read.next().unwrap().unwrap().trim())
            }

            pub fn try_next<T>(&mut self) -> Option<T>
            where T: FromStringKattis {
                Some(T::kattis_from_str(&self.read.next()?.ok()?))
            }

            /// n is the amount of lines to be skipped. This function is useful when an amount in a list is specified which is often useless.
            /// This is slightly faster than calling next and throwing it away as the data is not being copied into a temporary string.
            #[inline(always)]
            pub fn skip(&mut self, n: usize) {
                debug_assert!(n > 0);
                self.read.nth(n - 1);
            }

            /// Executes the function f on the next n lines of stdin. Useful for taking many lines input that need to be treated before stored.
            #[inline(always)]
            pub fn for_each_n<F, T>(&mut self, n: usize, mut f: F)
            where
                F: FnMut(T),
                T: FromStringKattis, {
                for _ in 0..n {
                    f(self.next());
                }
            }

            /// Maps the next n lines into a collection.
            pub fn collect_n<F, T, C, R>(&mut self, n: usize, mut f: F) -> C
            where
                F: FnMut(T) -> R,
                T: FromStringKattis,
                C: FromIterator<R>, {
                (0..n).map(|_| f(self.next())).collect()
            }
        }
    }

    pub use input::*;

    pub struct Memoize<T, R, A> {
        map: HashMap<T, R>,
        data: A,
    }

    impl<T, R, A> Memoize<T, R, A>
    where
        T: Hash + Eq + Clone,
        R: Clone,
    {
        pub fn new(data: A) -> Memoize<T, R, A> {
            Memoize {
                map: HashMap::new(),
                data,
            }
        }

        pub fn add_case(mut self, arg: T, result: R) -> Memoize<T, R, A> {
            self.map.insert(arg, result);
            self
        }

        pub fn recurse<F>(&mut self, f: F, t: T) -> R
        where F: Fn(&mut Self, T) -> R {
            match self.map.get(&t) {
                None => {
                    let r = f(self, t.clone());
                    self.map.insert(t, r.clone());
                    r
                },
                Some(v) => v.clone(),
            }
        }

        pub fn aux(&self) -> &A {
            &self.data
        }
    }

    pub struct OffsetArray<T: Sized> {
        ptr: *mut T,
        range: Range<isize>,
    }

    impl<T: Sized> OffsetArray<T> {
        /// If T implements drop you must not index any index which has not been reassigned. Instead the unsafe write function should be used.
        /// If T does not implement drop you can overwrite with index but not read before reassignment, because zeroed is not a valid value.
        /// If the content needs to be dropped, which is not always neccesary even if they implement drop, this must be done manually.
        pub unsafe fn new_zeroed_unsafe(range: Range<isize>) -> OffsetArray<T> {
            let layout = std::alloc::Layout::array::<T>((range.end - range.start) as usize).unwrap();
            unsafe {
                let ptr = std::alloc::alloc(layout) as *mut T;
                OffsetArray {
                    ptr,
                    range,
                }
            }
        }
    }

    impl<T: Sized> Drop for OffsetArray<T> {
        fn drop(&mut self) {
            unsafe {
                std::alloc::dealloc(
                    self.ptr as *mut _,
                    std::alloc::Layout::array::<T>((self.range.end - self.range.start) as usize).unwrap(),
                );
            }
        }
    }

    impl<T: Sized> Index<isize> for OffsetArray<T> {
        type Output = T;

        fn index(&self, index: isize) -> &Self::Output {
            assert!(self.range.contains(&index), "index out of bounds");
            unsafe { &mut *self.ptr.offset(index) }
        }
    }

    impl<T: Sized> IndexMut<isize> for OffsetArray<T> {
        fn index_mut(&mut self, index: isize) -> &mut Self::Output {
            assert!(self.range.contains(&index), "index out of bounds");
            unsafe { &mut *self.ptr.offset(index) }
        }
    }

    pub trait Mapper<Q, R> {
        #[inline]
        fn get(&self, q: &Q) -> Option<&R>;
        #[inline]
        fn get_mut(&mut self, q: &Q) -> Option<&mut R>;
    }

    impl<Q, R> Mapper<Q, R> for HashMap<Q, R>
    where Q: Eq + Hash
    {
        fn get(&self, q: &Q) -> Option<&R> {
            self.get(q)
        }

        fn get_mut(&mut self, q: &Q) -> Option<&mut R> {
            self.get_mut(q)
        }
    }

    impl<R> Mapper<usize, R> for Vec<R> {
        fn get(&self, q: &usize) -> Option<&R> {
            if *q < self.len() {
                Some(&self[*q])
            }
            else {
                None
            }
        }

        fn get_mut(&mut self, q: &usize) -> Option<&mut R> {
            if *q < self.len() {
                Some(&mut self[*q])
            }
            else {
                None
            }
        }
    }

    pub struct JunkMapping(());

    impl JunkMapping {
        pub fn new() -> JunkMapping {
            JunkMapping(())
        }
    }

    impl<Q> Mapper<Q, ()> for JunkMapping {
        fn get(&self, q: &Q) -> Option<&()> {
            Some(&self.0)
        }

        fn get_mut(&mut self, q: &Q) -> Option<&mut ()> {
            Some(&mut self.0)
        }
    }

    pub struct AsyclicDfs<'a, 'b, 'c, 'd, EC, SC, VC, F> {
        edges: &'a EC,
        state: &'b SC,
        values: &'c mut VC,
        func: &'d mut F,
    }
    impl<'a, 'b, 'c, 'd, EC, VC, F> AsyclicDfs<'a, 'b, 'c, 'd, EC, JunkMapping, VC, F> {
        pub fn new<Q, DQ, V, S>(
            edges: &'a EC, values: &'c mut VC, combiner: &'d mut F,
        ) -> AsyclicDfs<'a, 'b, 'c, 'd, EC, JunkMapping, VC, F>
        where
            DQ: Deref<Target = [Q]>,
            EC: Mapper<Q, DQ>,
            VC: Mapper<Q, Option<V>>,
            F: FnMut(&S, &[&V]) -> V, {
            AsyclicDfs {
                edges,
                state: &JunkMapping(()),
                values,
                func: combiner,
            }
        }
    }

    impl<'a, 'b, 'c, 'd, EC, SC, VC, F> AsyclicDfs<'a, 'b, 'c, 'd, EC, SC, VC, F> {
        pub fn with_state<Q, DQ, V, S, NSC>(self, state: &'b NSC) -> AsyclicDfs<'a, 'b, 'c, 'd, EC, NSC, VC, F>
        where
            NSC: Mapper<Q, S>,
            F: FnMut(&S, &[&V]) -> V, {
            AsyclicDfs {
                edges: self.edges,
                state,
                values: self.values,
                func: self.func,
            }
        }

        pub fn execute<Q, DQ, V, S>(mut self, src: &Q) -> Self
        where
            DQ: Deref<Target = [Q]>,
            EC: Mapper<Q, DQ>,
            SC: Mapper<Q, S>,
            VC: Mapper<Q, Option<V>>,
            F: FnMut(&S, &[&V]) -> V,
            V: 'a,
            'c: 'a, {
            // SAFETY: Saves 10 lines of code reconstructing self for no reason.
            // asyclic_dfs returns a reference which is bound on values but gets dropped immediately.
            // We ensure that there is only single mutability. This would not be a problem if the function was defined on this struct.
            let copy = unsafe { &mut *(self.values as *mut _) };
            asyclic_dfs(src, self.edges, self.state, copy, &mut self.func);
            self
        }
    }

    /// I know the polymorphism might look crazy but it is not as crazy as it seems :)
    /// To use you need a collections of edges, a collections of state, which must have a value for all vertices,
    /// a function to calculate a value of a vertex from its 'children' and a collection for the values.
    /// The collection of the values are modified by the function in dfs order. The function is not graph aware,
    /// and thus does not start over from a new vertex. If you need dfs on the entire graph run this once for each,
    /// node without in-going edges with the same collections. For the collections you can use antything that implements
    /// Mapper trait e.g. HashMap and Vec. Values must have an entry for all vertices. None is the default value.
    pub fn asyclic_dfs<Q, DQ, EC, SC, S, VC, V, F>(src: &Q, edges: &EC, state: &SC, values: &mut VC, mut combiner: F)
    where
        DQ: Deref<Target = [Q]>,
        EC: Mapper<Q, DQ>,
        SC: Mapper<Q, S>,
        VC: Mapper<Q, Option<V>>,
        F: FnMut(&S, &[&V]) -> V, {
        let mut level = 0;
        let mut stack = vec![];
        stack.push((src, state.get(src).unwrap(), edges.get(src).unwrap().deref(), 0));

        loop {
            let (level_node, level_state, level_edges, level_results) = &mut stack[level];
            if level_edges.len() == *level_results {
                let gather_level_results: Vec<_> = level_edges
                    .into_iter()
                    .map(|child| {
                        values
                            .get(child)
                            .unwrap()
                            .as_ref()
                            .unwrap()
                    })
                    .collect();
                let res = combiner(level_state, &gather_level_results);
                let acc = values.get_mut(level_node).unwrap();
                *acc = Some(res);
                if level == 0 {
                    return;
                }
                else {
                    stack[level - 1].3 += 1;
                    level -= 1;
                    stack.pop();
                }
            }
            else {
                let child = &level_edges[*level_results];
                match values.get(child).unwrap() {
                    Some(_) => *level_results += 1,
                    None => {
                        let r_edges = edges.get(child).unwrap();
                        let r_state = state.get(child).unwrap();
                        stack.push((child, r_state, r_edges, 0));
                        level += 1;
                    },
                }
            }
        }
    }

    /// Recursively calculates something for a tree by proprgating the value for the parent down to calculate the child
    /// f is the function to calculate the value for a node. The first parameter is parent state. Second is node state.
    /// Default is the value presented to the root as the parent value.
    /// if root needs to be handled specially Option can easily be used for this.
    pub fn tree_propogate<Q, DQ, EC, SC, S, V, F, D, I>(
        root: &Q, edges: &EC, state: &SC, f: &mut F, d: &mut D, default: &I,
    ) -> V
    where
        DQ: Deref<Target = [Q]>,
        EC: Mapper<Q, DQ>,
        SC: Mapper<Q, S>,
        D: FnMut(&I, &S) -> I,
        F: FnMut(&I, &[V]) -> V, {
        let mut level = 0;
        let mut stack = vec![];
        stack.push((d(default, state.get(root).unwrap()), edges.get(root).unwrap().deref(), vec![]));
        loop {
            let (int, level_edges, level_results) = &stack[level];
            if level_edges.len() == level_results.len() {
                let res = f(int, level_results);
                if level == 0 {
                    return res;
                }
                else {
                    stack[level - 1].2.push(res);
                    level -= 1;
                    stack.pop();
                }
            }
            else {
                let child = &level_edges[level_results.len()];
                let r_edges = edges.get(child).unwrap();
                let r_int = d(int, state.get(child).unwrap());
                stack.push((r_int, r_edges, vec![]));
                level += 1;
            }
        }
    }

    pub fn cartesian<A, B>(a: A, b: B) -> impl Iterator<Item = (A::Item, B::Item)>
    where
        A: Iterator,
        A::Item: Clone,
        B: Iterator + Clone, {
        a.flat_map(move |a| repeat(a).zip(b.clone()))
    }

    pub struct PolynomiumHashBuilder {
        p: u64,
        n: u64,
    }

    pub struct PolynomiumHasher {
        state: u64,
        p: u64,
        n: u64,
    }

    impl BuildHasher for PolynomiumHashBuilder {
        type Hasher = PolynomiumHasher;

        fn build_hasher(&self) -> Self::Hasher {
            PolynomiumHasher { state: 0, p: self.p, n: self.n }
        }
    }

    impl Hasher for PolynomiumHasher {
        fn finish(&self) -> u64 {
            self.state
        }

        fn write(&mut self, bytes: &[u8]) {
            for &byte in bytes {
                self.state = (self.state * self.p + byte as u64) % self.n
            }
        }
    }

    pub struct SeqHasher {
        p: u64,
        n: u64,
        ps: Vec<u64>,
        hashes: Vec<u64>,
    }

    impl SeqHasher {
        pub fn new<T: Hash>(elements: &[T], p: u64, n: u64) -> SeqHasher {
            let mut ps = Vec::with_capacity(elements.len());
            ps.push(1);
            for _ in 0..elements.len() {
                ps.push((ps.last().unwrap() * p) % n);
            }
            let mut hashes = Vec::with_capacity(elements.len());
            let hash_builder = PolynomiumHashBuilder { p, n };
            hashes.push(0);
            for elem in elements {
                let mut hasher = hash_builder.build_hasher();
                elem.hash(&mut hasher);
                hashes.push((hashes.last().unwrap() * p + hasher.finish()) % n);
            }
            SeqHasher { p, n, ps, hashes }
        }

        pub fn new_str(string: &str, p: u64, n: u64) -> SeqHasher {
            SeqHasher::new(string.as_bytes(), p, n)
        }

        pub fn slice<R: RangeBounds<usize>>(&self, r: R) -> u64 {
            let start_idx = match r.start_bound() {
                std::ops::Bound::Included(&idx) => idx,
                std::ops::Bound::Excluded(&idx) => idx + 1,
                std::ops::Bound::Unbounded => 0,
            };
            let end_idx = match r.end_bound() {
                std::ops::Bound::Included(&idx) => idx + 1,
                std::ops::Bound::Excluded(&idx) => idx,
                std::ops::Bound::Unbounded => usize::MAX,
            }.min(self.hashes.len() - 1);
            let start = self.hashes[start_idx];
            let end = self.hashes[end_idx];
            let ppower = self.ps[end_idx - start_idx];
            (end - ((start * ppower) % self.n) + self.n) % self.n
        }
    }
}

use std::{hash::Hash, collections::HashMap, fmt::Debug};

#[allow(unused)]
pub use tools::*;

#[allow(unused)]
#[derive(Debug)]
pub struct SegmentTree<T, F> {
    pub data: Vec<T>,
    f: F,
}

impl<T, F> SegmentTree<T, F>
where F: Fn(&T, &T) -> T, T: Debug
{
    #[allow(unused)]
    pub fn new(mut data: Vec<T>, f: F) -> SegmentTree<T, F> {
        assert!(data.len() > 0);
        let mut vec = Vec::new();
        unsafe {
            vec.reserve(data.len() * 2 - 1);
            vec.set_len(data.len() * 2 - 1);
            let mut seg = SegmentTree {
                data: vec,
                f,
            };
            seg.inhibit(0, &data);
            data.set_len(0);
            seg
        }
    }

    unsafe fn inhibit(&mut self, idx: usize, data: &[T]) {
        if data.len() == 1 {
            std::ptr::copy(&data[0], &mut self.data[idx], 1);
        }
        else {
            let split = data.len() / 2;
            let first_len = split * 2 - 1;
            self.inhibit(idx + 1, &data[..split]);
            self.inhibit(idx + 1 + first_len, &data[split..]);
            std::ptr::write(&mut self.data[idx], (self.f)(&self.data[idx + 1], &self.data[idx + 1 + first_len]));
        }
    }

    pub fn set(&mut self, idx: usize, val: T) {
        self.set_inner(idx, val, 0, self.data.len() / 2 + 1)
    }

    fn set_inner(&mut self, idx: usize, val: T, start: usize, size: usize) {
        if size == 1 {
            self.data[start] = val;
            return;
        }
        let split = size / 2;
        if idx < split {
            self.set_inner(idx, val, start + 1, split);
        }
        else {
            self.set_inner(idx - split, val, start + 2 * split, size - split);
        }
        self.data[start] = (self.f)(&self.data[start + 1], &self.data[start + 2 * split]);
    }

    pub fn get(&self, idx: usize) -> &T {
        self.get_inner(idx, 0, self.data.len() / 2 + 1)
    }

    pub fn get_inner(&self, idx: usize, start: usize, size: usize) -> &T {
        if size == 1 {
            return &self.data[start];
        }
        if idx < size / 2 {
            self.get_inner(idx, start + 1, size / 2)
        }
        else {
            self.get_inner(idx - size / 2, start + 2 * (size / 2), size - size / 2)
        }
    }

    #[allow(unused)]
    pub fn query(&self, lower: usize, higher: usize) -> Option<T>
    where T: Clone {
        (higher > lower).then(|| self.query_inner(lower, higher, 0, self.data.len() / 2 + 1))
    }

    fn query_inner(&self, lower: usize, higher: usize, idx: usize, size: usize) -> T
    where T: Clone {
        if lower == 0 && higher == size {
            return self.data[idx].clone();
        }
        let split = size / 2;
        let first_len = split * 2 - 1;
        match (lower < split, higher > split) {
            (true, true) => {
                let a = self.query_inner(lower, split, idx + 1, split);
                let b = self.query_inner(0, higher - split, idx + first_len + 1, size - split);
                (self.f)(&a, &b)
            },
            (true, false) => self.query_inner(lower, higher, idx + 1, split),
            (false, true) => self.query_inner(lower - split, higher - split, idx + first_len + 1, size - split),
            _ => unreachable!(),
        }
    }
}

/// Lower is inclusive, higher is exclusive. Finds the first value that fulfills pred.
#[allow(unused)]
pub fn binary_search(mut lower: u64, mut higher: u64, mut pred: impl FnMut(u64) -> bool) -> u64 {
    while higher > lower + 1 {
        let mid = (higher + lower) / 2;
        if pred(mid) {
            higher = mid;
        }
        else {
            lower = mid;
        }
    }
    higher
}

/// Lower is inclusive, higher is exclusive. Finds the first value that fulfills pred within eps
/// absolute and relative distance.
#[allow(unused)]
pub fn binary_search_float(mut lower: f64, mut higher: f64, mut pred: impl FnMut(f64) -> bool, eps: f64) -> f64 {
    while higher - lower > eps || (higher - lower) / lower > eps {
        let mid = (higher + lower) / 2.0;
        if pred(mid) {
            higher = mid;
        }
        else {
            lower = mid;
        }
    }
    (higher + lower) / 2.0
}

#[allow(unused)]
pub fn gcd(mut a: u64, mut b: u64) -> u64 {
    loop {
        a %= b;
        if a == 0 {
            return b;
        }
        b %= a;
        if b == 0 {
            return a;
        }
    }
}

pub trait OrdExt: Ord + Sized {
    fn min_assign(&mut self, other: Self) {
        match Self::cmp(self, &other) {
            std::cmp::Ordering::Greater => *self = other,
            _ => ()
        }
    }

    fn max_assign(&mut self, other: Self) {
        match Self::cmp(self, &other) {
            std::cmp::Ordering::Less => *self = other,
            _ => ()
        }
    }
}

impl<T> OrdExt for T where T: Ord {}

pub fn memoize<T: Hash + Eq, R>(f: fn(&T, &mut Recursion<T, R>) -> R, arg: T) -> R {
    let mut rec = Recursion {
        map: HashMap::new(),
        f,
    };
    rec.init_call(arg)
}

pub struct Recursion<T, R> {
    map: HashMap<T, R>,
    f: fn(&T, &mut Self) -> R,
}

impl<T: Hash + Eq, R> Recursion<T, R> {
    pub fn call(&mut self, arg: T) -> &R {
        if self.map.contains_key(&arg) {
            &self.map[&arg]
        }
        else {
            let f = self.f.clone();
            let ret = f(&arg, self);
            self.map.entry(arg).or_insert(ret)
        }
    }

    fn init_call(&mut self, arg: T) -> R {
        let f = self.f.clone();
        f(&arg, self)
    }
}

static mut W: u64 = 0;

pub fn setw(w: u64) {
    unsafe {
        W = w;
    }
}

pub fn init() {
    std::panic::set_hook(Box::new(|p| {
        let now = std::time::Instant::now();
        while now.elapsed() < std::time::Duration::from_millis(unsafe { W }) {}
        eprintln!("{:?}", p);
        std::process::exit(1);
    }));
}
