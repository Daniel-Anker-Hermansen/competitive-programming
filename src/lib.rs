mod util;

use std::{iter::repeat_with, sync::Mutex};

use ansi_term::Color;
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

static COMPLETED: Mutex<usize> = Mutex::new(0);

fn progress(solved: usize, total: usize) {
    let mut handle = COMPLETED.lock().expect("poison what is dat bitch?");
    *handle += solved;
    let solved = *handle;
    eprint!("\x0D{}", Color::RGB(0, 0, 255).paint("["));
    for _ in 0..solved {
        eprint!("{}", Color::RGB(0, 255, 0).paint("#"));
    }
    for _ in 0..total - solved {
        eprint!("{}", Color::RGB(255, 0, 0).paint("*"));
    }
    eprint!("{}", Color::RGB(0, 0, 255).paint("]"));
    eprint!(
        "{}{}{}{}{}",
        Color::RGB(0, 0, 255).paint("["),
        Color::RGB(0, 255, 0).paint(solved.to_string()),
        Color::RGB(0, 0, 255).paint("/"),
        Color::RGB(255, 0, 0).paint(total.to_string()),
        Color::RGB(0, 0, 255).paint("]")
    );
    let _ = std::io::stderr().flush();
}

pub fn solve_problem<T: Send>(
    mut read: impl FnMut(&mut Input) -> T,
    solve: impl Fn(T, &mut Output) + Sync,
) {
    let sequential = std::env::args().len() > 1;
    let mut input = Input::new();
    let t: usize = input.next();
    let data: Vec<T> = repeat_with(|| read(&mut input)).take(t).collect();
    let mut outputs = vec![Output(Vec::new()); t];
    if sequential {
        eprintln!("{}", "Running sequential ...");
        data.into_iter()
            .zip(&mut outputs)
            .enumerate()
            .for_each(|(idx, (data, output))| {
                solve(data, output);
                std::io::stdout()
                    .write(format!("Case #{}: ", idx + 1).as_bytes())
                    .unwrap();
                std::io::stdout().write(&output.0).unwrap();
            });
    } else {
        progress(0, t);
        data.into_par_iter()
            .zip(&mut outputs)
            .for_each(|(data, output)| {
                solve(data, output);
                progress(1, t);
            });
        eprintln!();
        for (idx, output) in outputs.into_iter().enumerate() {
            std::io::stdout()
                .write(format!("Case #{}: ", idx + 1).as_bytes())
                .unwrap();
            std::io::stdout().write(&output.0).unwrap();
        }
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
