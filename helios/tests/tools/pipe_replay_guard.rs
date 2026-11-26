//! CPU-testable submission and drain checks for the remote replay fixture.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    Transitions, VerifyTransitions, Qualification, VerifyBefore, Warmup, Begin, Measure, End, VerifyAfter,
}

impl Step {
    pub fn graph(self) -> &'static str {
        match self {
            Self::Transitions => "init",
            Self::Qualification | Self::Warmup | Self::Measure => "decode",
            Self::VerifyTransitions | Self::VerifyBefore | Self::VerifyAfter => "restore",
            Self::Begin | Self::End => "snapshot",
        }
    }
    pub fn readback(self) -> bool {
        matches!(self, Self::VerifyTransitions | Self::VerifyBefore | Self::VerifyAfter)
    }

    pub fn checks_output(self, index: usize) -> bool {
        match self {
            Self::VerifyTransitions => index < 7,
            Self::VerifyBefore => index >= 7,
            Self::VerifyAfter => true,
            _ => false,
        }
    }
}

pub fn schedule(warmup: usize, rounds: usize) -> Result<Vec<Step>, &'static str> {
    if !(1..=4).contains(&warmup) || !(1..=28).contains(&rounds) {
        return Err("invalid warmup or measured round count");
    }
    let mut steps = vec![Step::Transitions, Step::VerifyTransitions, Step::Qualification, Step::VerifyBefore];
    steps.extend(std::iter::repeat_n(Step::Warmup, warmup));
    steps.push(Step::Begin);
    steps.extend(std::iter::repeat_n(Step::Measure, rounds));
    steps.extend([Step::End, Step::VerifyAfter]);
    Ok(steps)
}

/// A failed synchronization leaves both staging and native owners live. The
/// caller retains native owners independently until successful coordinated drain.
pub fn finish_transfer<T>(staging: T, copied: bool, synchronized: bool) -> Result<T, &'static str> {
    if !synchronized {
        std::mem::forget(staging);
        return Err("transfer drain unconfirmed; staging retained until process exit");
    }
    if !copied { return Err("CUDA copy failed after confirmed transfer drain"); }
    Ok(staging)
}

#[derive(Default)]
pub struct Guard {
    connected: bool,
    outstanding: bool,
    completed: usize,
    failed: bool,
}

impl Guard {
    pub fn connect(&mut self, records: &[Vec<u8>]) -> Result<(), &'static str> {
        if self.connected || records.len() != 16
            || records.iter().any(|r| r.len() < 5 || r[..4] != 14u32.to_le_bytes())
            || records.iter().any(|r| r.len() != records[0].len()) {
            return Err("requires sixteen equal-size ABI14 records, connected once");
        }
        self.connected = true;
        Ok(())
    }
    pub fn submit(&mut self) -> Result<(), &'static str> {
        if !self.connected || self.outstanding || self.failed {
            return Err("submit requires connection, no outstanding replay and no failure");
        }
        self.outstanding = true;
        Ok(())
    }
    pub fn complete(&mut self, parity: bool) -> Result<(), &'static str> {
        if !self.outstanding { return Err("completion without submission"); }
        self.outstanding = false;
        if !parity {
            self.failed = true;
            return Err("parity failure");
        }
        self.completed += 1;
        Ok(())
    }
    pub fn completed(&self) -> usize {
        self.completed
    }
    pub fn release(&self, coordinated_drain: bool) -> Result<(), &'static str> {
        if !coordinated_drain || self.outstanding {
            return Err("retain owners until coordinated drain");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn records() -> Vec<Vec<u8>> { vec![[14u32.to_le_bytes().as_slice(), &[0; 8]].concat(); 16] }
    #[test]
    fn measurement_has_no_readback_and_boundaries_are_separate() {
        use Step::*;
        for warmup in 1..=4 {
            for rounds in 1..=28 {
                let steps = schedule(warmup, rounds).unwrap();
                assert_eq!(&steps[..4], &[Transitions, VerifyTransitions, Qualification, VerifyBefore]);
                assert_eq!(&steps[4..4+warmup], vec![Warmup; warmup]);
                assert_eq!(steps[4+warmup], Begin);
                assert_eq!(&steps[5+warmup..5+warmup+rounds], vec![Measure; rounds]);
                assert_eq!(&steps[5+warmup+rounds..], &[End, VerifyAfter]);
                assert_eq!(steps.iter().filter(|s| s.readback()).count(), 3);
                for step in &steps {
                    if step.readback() { assert_eq!(step.graph(), "restore"); }
                    if matches!(step, Begin | End) { assert_eq!(step.graph(), "snapshot"); }
                    if *step == Measure { assert!(!step.readback()); }
                }
                for index in 0..9 {
                    assert_eq!(VerifyTransitions.checks_output(index), index < 7);
                    assert_eq!(VerifyBefore.checks_output(index), index >= 7);
                    assert!(VerifyAfter.checks_output(index));
                    assert!(!Measure.checks_output(index));
                }
            }
        }
        assert!(schedule(0, 28).is_err());
        assert!(schedule(2, 29).is_err());
    }
    #[test]
    fn staging_is_retained_on_unconfirmed_drain_even_when_copy_failed() {
        use std::{rc::Rc, cell::Cell};
        struct Staging(Rc<Cell<usize>>);
        impl Drop for Staging {
            fn drop(&mut self) { self.0.set(self.0.get() + 1); }
        }
        for copied in [false, true] {
            for drained in [false, true] {
                let drops = Rc::new(Cell::new(0));
                let result = finish_transfer(Staging(drops.clone()), copied, drained);
                assert_eq!(result.is_ok(), copied && drained);
                drop(result);
                assert_eq!(drops.get(), usize::from(drained));
            }
        }
    }
    #[test]
    fn submit_before_connect_and_overlapping_replays_fail() {
        let mut g = Guard::default();
        assert!(g.submit().is_err());
        g.connect(&records()).unwrap();
        g.submit().unwrap();
        assert!(g.submit().is_err());
        assert!(g.release(true).is_err());
    }
    #[test]
    fn mixed_abi_missing_rank_and_truncated_record_fail() {
        for records in [records()[..15].to_vec(), {
            let mut r = records(); r[8][0] = 11; r
        }, {
            let mut r = records(); r[1].pop(); r
        }] { assert!(Guard::default().connect(&records).is_err()); }
    }
    #[test]
    fn phase_completions_advance_without_reset() {
        let mut g = Guard::default();
        g.connect(&records()).unwrap();
        for (index, _) in schedule(2, 28).unwrap().iter().enumerate() {
            g.submit().unwrap();
            g.complete(true).unwrap();
            assert_eq!(g.completed(), index + 1);
        }
        assert!(g.release(false).is_err());
        assert!(g.release(true).is_ok());
    }
    #[test]
    fn failed_parity_blocks_further_submission_and_requires_drain() {
        let mut g = Guard::default();
        g.connect(&records()).unwrap();
        g.submit().unwrap();
        assert!(g.complete(false).is_err());
        assert!(g.submit().is_err());
        assert!(g.release(false).is_err());
        assert!(g.release(true).is_ok());
    }
}
