mod util;

use std::iter::repeat_with;

pub use util::*;

pub use std::io::Write;

use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

#[derive(Clone)]
pub struct Output(Vec<u8>);

impl Write for Output {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.extend(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub fn solve_problem<T: Send>(mut read: impl FnMut(&mut Input) -> T, solve: impl Fn(T, &mut Output) + Sync) {
    let mut input = Input::new();
    let t: usize = input.next();
    let data: Vec<T> = repeat_with(|| read(&mut input)).take(t).collect();
    let mut outputs = vec![Output(Vec::new()); t];
    data.into_par_iter().zip(&mut outputs).for_each(|(data, output)| solve(data, output));
    for (idx, output) in outputs.into_iter().enumerate() {
        std::io::stdout().write(format!("Case #{}: ", idx + 1).as_bytes()).unwrap();
        std::io::stdout().write(&output.0).unwrap();
    }
}

#[macro_export]
macro_rules! run {
    ($x:tt, $y:tt) => {
        fn main() {
            $crate::solve_problem($x, $y);
        }
    };
}

#[macro_export]
macro_rules! output {
    ($y:tt, $($x:expr),*) => {
        write!($y, $($x),*).unwrap()
    };
}

#[macro_export]
macro_rules! outputln {
    ($y:tt, $($x:expr),*) => {
        writeln!($y, $($x),*).unwrap()
    };
}
