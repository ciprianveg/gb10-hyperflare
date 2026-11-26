/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Seeding of a request's acceptance profile from the acceptance population.
//!
//! The fixtures are one request slot with host-allocated rounds of one to four
//! target rows lasting 28, 43.7, 48.5, and 53.2 ms, and two prefix-survival
//! curves: one at which the default policy drafts and one at which it does not.

use super::*;
use speculation::{AcceptanceProfile, ProposalSource::{Copy, Mtp}};

const ROUND_NS: [u64; 4] = [28_000_000, 43_700_000, 48_500_000, 53_200_000];
/// Expected output of 1.76, 2.32, and 2.74 tokens at two, three, and four rows.
const HIGH: [f32; 7] = [0.76, 0.56, 0.42, 0.30, 0.21, 0.15, 0.11];
/// Expected output of 1.35, 1.47, and 1.51 tokens at two, three, and four rows.
const LOW: [f32; 7] = [0.35, 0.12, 0.04, 0.01, 0.0, 0.0, 0.0];

fn measured_engine() -> RequestScheduler {
    let mut s = interactive_engine(1, true);
    s.enable_speculation(&[1, 2, 3, 4]).unwrap();
    for (index, nanos) in ROUND_NS.into_iter().enumerate() {
        set_cost(&mut s, GraphKind::Speculative, 1, index + 1, nanos);
    }
    s
}

fn decoding(s: &mut RequestScheduler, id: u32, output: u32) -> RequestHandle {
    let handle = add(s, id, 64, output);
    ready(s, &[handle]);
    handle
}

/// Executes the selected plan. Each offered depth reports `curve` as its
/// survival and `accepted` drafts are emitted after the pending token.
fn round(s: &mut RequestScheduler, curve: &[f32; 7], accepted: usize) -> ScheduledPlan {
    let plan = s.plan().expect("a decoding request must make progress");
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    let duration = plan.estimated_duration_ns();
    s.advance_to(s.now_ns + duration).unwrap();
    let mut records = completion_records(s, &plan, fence, accepted);
    let slot = plan.lanes[0].request.slot as usize;
    let mut offered = 0;
    if let Some(vector) = records[slot].vector.as_mut() {
        offered = plan.lanes[0].tokens - 1;
        vector.offered_drafts = offered as i64;
    }
    let mut survival = vec![-1.0; s.capabilities.completion_token_capacity - 1];
    survival[..offered].copy_from_slice(&curve[..offered]);
    let output = s.complete_with_survival(fence, &records, gpu_time(duration), Some(&[survival])).unwrap();
    for token in output {
        s.published_output(token.request, fence).unwrap();
    }
    let request = plan.lanes[0].request;
    if s.request(request).is_ok_and(|r| r.phase == RequestPhase::Finishing && r.pending == 0) {
        s.finish(request).unwrap();
    }
    plan
}

/// Completes `requests` requests whose own estimates keep them drafting while
/// their rounds report `curve`.
fn drafting_requests(s: &mut RequestScheduler, curve: &[f32; 7], requests: u32, rounds: usize) {
    for id in 0..requests {
        let handle = decoding(s, 100 + id, rounds as u32 + 16);
        for _ in 0..128 {
            s.request_mut(handle).unwrap().acceptance.observe(Mtp, 0, 7, 7, true, None);
        }
        for _ in 0..rounds {
            round(s, curve, 0);
        }
        s.cancel(handle).unwrap();
    }
}

fn seed(s: &mut RequestScheduler) -> Vec<f64> {
    let handle = decoding(s, 1, 4096);
    let seed = (0..7).map(|depth| s.request(handle).unwrap().acceptance
        .prefix_probability(Mtp, depth).unwrap()).collect();
    s.cancel(handle).unwrap();
    seed
}

/// Index of the first drafting round among a request's first 64 decode
/// rounds, without acceptance probes.
fn first_drafting_round(s: &mut RequestScheduler) -> Option<usize> {
    s.refresh.acceptance_credit_ns = f64::NEG_INFINITY;
    (0..64).position(|_| round(s, &HIGH, 0).token_capacity > 1)
}

#[test]
fn a_request_after_high_acceptance_requests_drafts_within_its_first_rounds() {
    let mut fixed = measured_engine();
    decoding(&mut fixed, 1, 4096);
    assert_eq!(first_drafting_round(&mut fixed), None);

    // A request whose own estimate already is the reported curve still spends
    // its first two decode rounds at width one.
    let mut known = measured_engine();
    let handle = decoding(&mut known, 1, 4096);
    for _ in 0..128 {
        known.request_mut(handle).unwrap().acceptance.observe(Mtp, 0, 7, 0, true, Some(&HIGH));
    }
    assert_eq!(first_drafting_round(&mut known), Some(2));

    for (requests, expected) in [(1, Some(14)), (3, Some(5)), (16, Some(3))] {
        let mut seeded = measured_engine();
        drafting_requests(&mut seeded, &HIGH, requests, 8);
        decoding(&mut seeded, 1, 4096);
        assert_eq!(first_drafting_round(&mut seeded), expected, "{requests} requests");
    }
}

#[test]
fn a_request_stays_at_width_one_after_low_acceptance_requests() {
    let mut s = measured_engine();
    drafting_requests(&mut s, &LOW, 3, 8);
    let handle = decoding(&mut s, 1, 4096);
    let profile = s.request(handle).unwrap().acceptance();
    for depth in 0..3 {
        assert!(profile.prefix_probability(Mtp, depth).unwrap() < 0.5f64.powi(depth as i32 + 1));
    }
    for _ in 0..64 {
        let plan = round(&mut s, &LOW, 0);
        assert!(plan.token_capacity == 1 || plan.origin.is_acceptance());
    }
}

#[test]
fn requests_start_from_the_fixed_prior_until_a_drafting_round_completes() {
    let fixed = AcceptanceProfile::new(16);
    let mut s = measured_engine();
    let handle = decoding(&mut s, 1, 4096);
    s.refresh.acceptance_credit_ns = f64::NEG_INFINITY;
    for _ in 0..64 {
        assert_eq!(round(&mut s, &HIGH, 0).token_capacity, 1);
    }
    s.cancel(handle).unwrap();
    let handle = decoding(&mut s, 2, 4096);
    let profile = s.request(handle).unwrap().acceptance();
    for source in [Mtp, Copy] {
        for optimistic in [false, true] {
            assert_eq!(profile.curve(source, 0, optimistic), fixed.curve(source, 0, optimistic));
        }
        assert_eq!(profile.information(source, 0, 0), fixed.information(source, 0, 0));
    }
}

#[test]
fn one_request_moves_the_population_by_at_most_its_share() {
    let mut short = measured_engine();
    drafting_requests(&mut short, &[1.0; 7], 1, 8);
    let mut long = measured_engine();
    drafting_requests(&mut long, &[1.0; 7], 1, 2000);
    let expected = seed(&mut short);
    // Eight observations of 1.0 against the fixed prior's weight of eight.
    for depth in 0..3 {
        assert_eq!(expected[depth], (0.5f64.powi(depth as i32 + 1) + 1.0) / 2.0);
    }
    for depth in 3..7 {
        assert_eq!(expected[depth], 0.5f64.powi(depth as i32 + 1));
    }
    assert_eq!(seed(&mut long), expected);

    // At the weight cap, a ninth observation moves a mean by 1/128 of its error.
    let mut population = speculation::AcceptancePopulation::new(16);
    for _ in 0..4096 { population.observe(Mtp, 0, 0.0); }
    let before = population.mean(Mtp, 0);
    let mut request = AcceptanceProfile::seeded(&population);
    for _ in 0..2000 {
        request.observe_shared(Mtp, 0, 1, 1, true, None, Some(&mut |depth, value| population.observe(Mtp, depth, value)));
    }
    let share = 1.0 - (1.0f64 - 1.0 / 128.0).powi(8);
    assert!((population.mean(Mtp, 0) - (before + share * (1.0 - before))).abs() < 1e-12);
    assert_eq!(population.mean(Copy, 0), 0.5);
}

#[test]
fn a_low_acceptance_request_returns_to_width_one_after_a_high_acceptance_population() {
    let mut s = measured_engine();
    drafting_requests(&mut s, &HIGH, 16, 8);
    decoding(&mut s, 1, 4096);
    s.refresh.acceptance_credit_ns = f64::NEG_INFINITY;
    let widths: Vec<_> = (0..64).map(|_| round(&mut s, &LOW, 0).token_capacity).collect();
    let drafting = widths.iter().filter(|&&width| width > 1).count();
    assert!((1..=4).contains(&drafting), "{widths:?}");
    assert!(widths[32..].iter().all(|&width| width == 1), "{widths:?}");
}

#[test]
fn confidence_forecast_catalogs_keep_the_fixed_acceptance_prior() {
    let fixture = measured_engine();
    let mut capabilities = fixture.capabilities.clone();
    capabilities.speculative_allocations = vec![(1, 7, 8, 8, crate::execution::PrefixAllocation::Gpu)];
    let mut s = RequestScheduler::new([0; 32], fixture.limits.clone(), capabilities).unwrap();
    for depth in 0..7 {
        for _ in 0..64 { s.acceptance_population.observe(Mtp, depth, 1.0); }
    }
    let handle = add(&mut s, 1, 64, 64);
    let fixed = AcceptanceProfile::new(16);
    assert_eq!(s.request(handle).unwrap().acceptance().curve(Mtp, 0, false), fixed.curve(Mtp, 0, false));
}

/// Deterministic uniform draws for the simulated acceptance outcomes.
struct Draws(u64);

impl Draws {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Five consecutive 1,024-token requests on text with the `HIGH` curve. Each
/// round's accepted prefix is drawn from that curve. Prints, per request, the
/// width-one rounds before its first ordinary drafting round, its width-one
/// and total rounds, and its simulated decode time.
#[test]
fn simulated_requests_reach_drafting_sooner_as_the_population_fills() {
    let mut s = measured_engine();
    let mut draws = Draws(1);
    let mut before_drafting = Vec::new();
    let mut seconds = Vec::new();
    for id in 0..5 {
        let handle = decoding(&mut s, id, 1024);
        let start = s.now_ns;
        let (mut first, mut narrow, mut rounds) = (None, 0, 0);
        while s.request(handle).is_ok() {
            let draw = draws.next();
            let width = s.plan().unwrap().token_capacity;
            let accepted = HIGH[..width - 1].iter().take_while(|&&survival| draw < survival).count();
            let plan = round(&mut s, &HIGH, accepted);
            if plan.token_capacity == 1 { narrow += 1; }
            else if plan.origin == PlanOrigin::Ordinary { first.get_or_insert(rounds); }
            rounds += 1;
        }
        let elapsed = (s.now_ns - start) as f64 / 1e9;
        println!("request {id}: {} width-one rounds before ordinary drafting, {narrow} of {rounds} rounds at width one, {elapsed:.2} s, {:.1} tokens/s",
            first.unwrap_or(rounds), 1024.0 / elapsed);
        before_drafting.push(first.unwrap_or(rounds));
        seconds.push(elapsed);
    }
    assert!(before_drafting[0] > 100, "{before_drafting:?}");
    assert!(before_drafting[1..].iter().all(|&rounds| rounds <= 8), "{before_drafting:?}");
    assert!(seconds[1..].iter().all(|&elapsed| elapsed < seconds[0]), "{seconds:?}");
}

fn host_c2(width: usize) -> (RequestScheduler, [RequestHandle; 2]) {
    let mut s = interactive_engine(2, true);
    s.capabilities.speculative_allocations = (1..=2).flat_map(|batch|
        (1..=6).map(move |width| (batch, width - 1, width, batch * width,
            crate::execution::PrefixAllocation::Host))).collect();
    s.enable_speculation(&[1, 2, 3, 4, 5, 6]).unwrap();
    s.fixed_speculative_width(Some(width)).unwrap();
    let handles = [add(&mut s, 201, 64, 4096), add(&mut s, 202, 64, 4096)];
    ready(&mut s, &handles);
    (s, handles)
}

#[test]
fn population_rejects_invalid_c2_completion_without_partial_learning() {
    let (mut s, handles) = host_c2(6);
    let plan = s.plan().unwrap();
    assert_eq!((plan.batch_capacity, plan.token_capacity, plan.lanes.len()), (2, 6, 2));
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    let records = completion_records(&s, &plan, fence, 0);
    let before: Vec<_> = (0..15).map(|d| s.acceptance_population.mean(Mtp, d)).collect();
    let mut malformed = records.clone();
    malformed[handles[1].slot as usize].generation += 1;
    assert!(s.complete(fence, &malformed, gpu_time(MS)).is_err());
    let mut survival = vec![vec![-1.0; 15]; 2];
    for row in &mut survival { row[..5].fill(0.5); }
    survival[handles[1].slot as usize][2] = f32::NAN;
    assert!(s.complete_with_survival(fence, &records, gpu_time(MS), Some(&survival)).is_err());
    assert_eq!((0..15).map(|d| s.acceptance_population.mean(Mtp, d)).collect::<Vec<_>>(), before);
    for handle in handles { assert_eq!(s.request(handle).unwrap().acceptance.rounds(), 0); }
    s.complete(fence, &records, gpu_time(MS)).unwrap();
    for depth in 0..5 {
        assert!((s.acceptance_population.mean(Mtp, depth) - before[depth] * 8.0 / 10.0).abs() < 1e-12);
    }
}

#[test]
fn c2_requests_each_contribute_eight_observations_at_each_offered_depth() {
    let (mut s, handles) = host_c2(6);
    let curve = [0.75f32, 0.5, 0.25, 0.125, 0.0625];
    for round in 0..20 {
        let plan = s.plan().unwrap();
        assert_eq!((plan.batch_capacity, plan.token_capacity, plan.lanes.len()), (2, 6, 2));
        let fence = Fence(s.submitted.0 + 1);
        s.commit(&plan, fence).unwrap();
        let records = completion_records(&s, &plan, fence, 0);
        let mut survival = vec![vec![-1.0; 15]; 2];
        for row in &mut survival { row[..5].copy_from_slice(&curve); }
        let output = s.complete_with_survival(fence, &records, gpu_time(MS), Some(&survival)).unwrap();
        for token in output { s.published_output(token.request, fence).unwrap(); }
        let observations = 2 * (round + 1).min(8);
        for depth in 0..5 {
            let prior = 0.5f64.powi(depth as i32 + 1);
            let expected = (8.0 * prior + observations as f64 * f64::from(curve[depth]))
                / (8 + observations) as f64;
            assert!((s.acceptance_population.mean(Mtp, depth) - expected).abs() < 1e-12);
        }
        for depth in 5..15 {
            assert_eq!(s.acceptance_population.mean(Mtp, depth), 0.5f64.powi(depth as i32 + 1));
        }
    }
    for handle in handles { assert_eq!(s.request(handle).unwrap().acceptance.rounds(), 20); }
}

#[test]
fn fixed_four_and_five_drafts_enforce_width_despite_population_on_host_c2() {
    for width in [5, 6] {
        for observation in [0.0, 1.0] {
            let (mut s, handles) = host_c2(width);
            for depth in 0..5 {
                for _ in 0..128 { s.acceptance_population.observe(Mtp, depth, observation); }
            }
            for handle in handles {
                s.request_mut(handle).unwrap().acceptance = AcceptanceProfile::seeded(&s.acceptance_population);
            }
            let plan = s.plan().unwrap();
            assert_eq!((plan.batch_capacity, plan.token_capacity, plan.lanes.len()), (2, width, 2));
            assert!(!plan.requires_feedback);
        }
    }
}

#[test]
fn capacity_six_completion_seeds_a_subsequent_request_from_five_survival_entries() {
    let fixture = interactive_engine(1, true);
    let mut capabilities = fixture.capabilities.clone();
    capabilities.completion_token_capacity = 6;
    capabilities.speculative_allocations = (1..=6).map(|width|
        (1, width - 1, width, width, crate::execution::PrefixAllocation::Host)).collect();
    let mut s = RequestScheduler::new([7; 32], fixture.limits.clone(), capabilities).unwrap();
    let mut graphs = super::super::tests::duration_graphs(1);
    graphs.retain(|g| g.kind != GraphKind::Speculative || g.token_capacity <= 6);
    s.measured_graph_durations(&graphs).unwrap();
    s.enable_speculation(&[1, 2, 3, 4, 5, 6]).unwrap();
    s.fixed_speculative_width(Some(6)).unwrap();
    let handle = add(&mut s, 301, 64, 4096);
    ready(&mut s, &[handle]);
    let plan = s.plan().unwrap();
    assert_eq!(plan.token_capacity, 6);
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    let mut records = completion_records(&s, &plan, fence, 0);
    records[handle.slot as usize].vector.as_mut().unwrap().tokens.truncate(6);
    let survival = vec![vec![0.75, 0.5, 0.25, 0.125, 0.0625]];
    let output = s.complete_with_survival(fence, &records, gpu_time(MS), Some(&survival)).unwrap();
    for token in output { s.published_output(token.request, fence).unwrap(); }
    s.cancel(handle).unwrap();
    let next = add(&mut s, 302, 64, 4096);
    let acceptance = s.request(next).unwrap().acceptance();
    for depth in 0..5 {
        let expected = (8.0 * 0.5f64.powi(depth as i32 + 1) + f64::from(survival[0][depth])) / 9.0;
        assert!((acceptance.prefix_probability(Mtp, depth).unwrap() - expected).abs() < 1e-12);
    }
    assert_eq!(acceptance.prefix_probability(Mtp, 5), None);
}

#[test]
fn cancelled_submitted_c2_lane_and_replayed_completion_do_not_contribute() {
    let (mut s, handles) = host_c2(6);
    let plan = s.plan().unwrap();
    assert_eq!(plan.lanes.len(), 2);
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    let records = completion_records(&s, &plan, fence, 0);
    s.cancel(handles[0]).unwrap();
    let output = s.complete(fence, &records, gpu_time(MS)).unwrap();
    assert!(output.iter().all(|token| token.request == handles[1]));
    assert_eq!(s.request(handles[1]).unwrap().acceptance.rounds(), 1);
    let means: Vec<_> = (0..15).map(|depth| s.acceptance_population.mean(Mtp, depth)).collect();
    for depth in 0..5 {
        assert!((means[depth] - 8.0 * 0.5f64.powi(depth as i32 + 1) / 9.0).abs() < 1e-12);
    }
    assert!(s.complete(fence, &records, gpu_time(MS)).is_err());
    assert_eq!((0..15).map(|depth| s.acceptance_population.mean(Mtp, depth)).collect::<Vec<_>>(), means);
    assert_eq!(s.request(handles[1]).unwrap().acceptance.rounds(), 1);
}
