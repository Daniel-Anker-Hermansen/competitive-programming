pub mod input {
    use std::{
        collections::{BTreeSet, HashSet, VecDeque},
        hash::Hash,
        io::{BufRead, Lines, StdinLock},
        iter::FromIterator,
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

    impl_local_from_str!(bool char f32 f64 i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize String);

    impl<T> LocalFromStr for Vec<T>
    where
        T: LocalFromStr,
    {
        fn local_from_str(s: &str) -> Self {
            yuck(s)
        }
    }

    impl<T> LocalFromStr for VecDeque<T>
    where
        T: LocalFromStr,
    {
        fn local_from_str(s: &str) -> Self {
            yuck(s)
        }
    }

    impl<T> LocalFromStr for HashSet<T>
    where
        T: LocalFromStr + Hash + Eq,
    {
        fn local_from_str(s: &str) -> Self {
            yuck(s)
        }
    }

    impl<T> LocalFromStr for BTreeSet<T>
    where
        T: LocalFromStr + Ord,
    {
        fn local_from_str(s: &str) -> Self {
            yuck(s)
        }
    }

    macro_rules! impl_tuple {
            (k $($t:ident), *) => {
                impl<$($t), *> LocalFromStr for ($($t), *) where $($t: LocalFromStr), * {
                    fn local_from_str(s: &str) -> Self {
                        let mut iter = s.split_ascii_whitespace();
                        ($(<$t>::local_from_str(iter.next().unwrap().into())), *)
                    }
                }
            };
            ($t0:ident) => {};
            ($t0:ident, $($t:ident), *) => {
                impl_tuple!(k $t0, $($t), *);
                impl_tuple!($($t), *);
            };
        }

    impl_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P);

    fn yuck<C, T>(s: &str) -> C
    where
        C: FromIterator<T>,
        T: LocalFromStr,
    {
        s.split_ascii_whitespace().map(T::local_from_str).collect()
    }

    pub trait FromStringKattis {
        fn kattis_from_str(s: &str) -> Self;
    }

    impl<T> FromStringKattis for T
    where
        T: LocalFromStr,
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
            Input { read }
        }

        /// Reads the next line from stdin and converts it to type T. Panics if an io-error happens or the conversion fails.
        #[inline(always)]
        pub fn next<T>(&mut self) -> T
        where
            T: FromStringKattis,
        {
            T::kattis_from_str(self.read.next().unwrap().unwrap().trim())
        }

        pub fn try_next<T>(&mut self) -> Option<T>
        where
            T: FromStringKattis,
        {
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
            T: FromStringKattis,
        {
            for _ in 0..n {
                f(self.next());
            }
        }

        /// Maps the next n lines into a collection.
        pub fn collect_n<F, T, C, R>(&mut self, n: usize, mut f: F) -> C
        where
            F: FnMut(T) -> R,
            T: FromStringKattis,
            C: FromIterator<R>,
        {
            (0..n).map(|_| f(self.next())).collect()
        }
    }
}

use std::{fmt::Debug, iter::repeat, sync::atomic::AtomicU64};

pub use input::*;

pub fn cartesian<A, B>(a: A, b: B) -> impl Iterator<Item = (A::Item, B::Item)>
where
    A: Iterator,
    A::Item: Clone,
    B: Iterator + Clone,
{
    a.flat_map(move |a| repeat(a).zip(b.clone()))
}


#[derive(Debug)]
pub struct SegmentTree<T, F> {
    pub data: Vec<T>,
    f: F,
}

impl<T, F> SegmentTree<T, F>
where
    F: Fn(&T, &T) -> T,
    T: Debug,
{
    #[allow(unused)]
    pub fn new(mut data: Vec<T>, f: F) -> SegmentTree<T, F> {
        assert!(data.len() > 0);
        let mut vec = Vec::new();
        unsafe {
            vec.reserve(data.len() * 2 - 1);
            vec.set_len(data.len() * 2 - 1);
            let mut seg = SegmentTree { data: vec, f };
            seg.inhibit(0, &data);
            data.set_len(0);
            seg
        }
    }

    unsafe fn inhibit(&mut self, idx: usize, data: &[T]) {
        if data.len() == 1 {
            std::ptr::copy(&data[0], &mut self.data[idx], 1);
        } else {
            let split = data.len() / 2;
            let first_len = split * 2 - 1;
            self.inhibit(idx + 1, &data[..split]);
            self.inhibit(idx + 1 + first_len, &data[split..]);
            std::ptr::write(
                &mut self.data[idx],
                (self.f)(&self.data[idx + 1], &self.data[idx + 1 + first_len]),
            );
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
        } else {
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
        } else {
            self.get_inner(idx - size / 2, start + 2 * (size / 2), size - size / 2)
        }
    }

    #[allow(unused)]
    pub fn query(&self, lower: usize, higher: usize) -> Option<T>
    where
        T: Clone,
    {
        (higher > lower).then(|| self.query_inner(lower, higher, 0, self.data.len() / 2 + 1))
    }

    fn query_inner(&self, lower: usize, higher: usize, idx: usize, size: usize) -> T
    where
        T: Clone,
    {
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
            }
            (true, false) => self.query_inner(lower, higher, idx + 1, split),
            (false, true) => self.query_inner(
                lower - split,
                higher - split,
                idx + first_len + 1,
                size - split,
            ),
            _ => unreachable!(),
        }
    }
}

/// Lower is inclusive, higher is exclusive. Finds the first value that fulfills pred.
pub fn binary_search(mut lower: u64, mut higher: u64, mut pred: impl FnMut(u64) -> bool) -> u64 {
    while higher > lower + 1 {
        let mid = (higher + lower) / 2;
        if pred(mid) {
            higher = mid;
        } else {
            lower = mid;
        }
    }
    higher
}

/// Finds the first value that fulfills pred within eps absolute and relative distance.
pub fn binary_search_float(
    mut lower: f64,
    mut higher: f64,
    mut pred: impl FnMut(f64) -> bool,
    eps: f64,
) -> f64 {
    while higher - lower > eps || (higher - lower) / lower > eps {
        let mid = (higher + lower) / 2.0;
        if pred(mid) {
            higher = mid;
        } else {
            lower = mid;
        }
    }
    (higher + lower) / 2.0
}

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
            _ => (),
        }
    }

    fn max_assign(&mut self, other: Self) {
        match Self::cmp(self, &other) {
            std::cmp::Ordering::Less => *self = other,
            _ => (),
        }
    }
}

impl<T> OrdExt for T where T: Ord {}

static W: AtomicU64 = AtomicU64::new(0);

/// Sets a delay after panic before termination. This can be used to know where in a program a
/// panic happened based only on the running time by setting different delays at different program
/// points. If `init` is not called before the panic this has no effect.
pub fn setw(w: u64) {
    W.store(w, std::sync::atomic::Ordering::SeqCst);
}

/// See `setw`
pub fn init() {
    std::panic::set_hook(Box::new(|p| {
        let now = std::time::Instant::now();
        while now.elapsed() < std::time::Duration::from_millis(W.load(std::sync::atomic::Ordering::SeqCst)) {}
        eprintln!("{:?}", p);
        std::process::exit(1);
    }));
}
