//! Deterministic BF16 fixture bytes and fixed-rank FP32 references.

const PERIOD: usize = 1 << 20;
pub fn word(index: usize, rank: usize, element: usize) -> u16 {
    0x3c00 + ((index*131 + rank*17 + element*7 + (element/1024)*13) % 1024) as u16
}
fn repeat_to(cycle: &[u8], bytes: usize) -> Vec<u8> {
    cycle.iter().copied().cycle().take(bytes).collect()
}
pub fn source(index: usize, bytes: usize, rank: usize) -> Vec<u8> {
    let cycle: Vec<u8> = (0..(bytes/2).min(PERIOD))
        .flat_map(|i| word(index,rank,i).to_le_bytes()).collect();
    repeat_to(&cycle,bytes)
}
pub fn reduced_word(index: usize, element: usize) -> u16 {
    let mut sum = 0.0f32;
    for rank in 0..16 {
        sum += f32::from_bits((word(index,rank,element) as u32) << 16);
    }
    let bits = sum.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}
pub fn expected(index: usize, kind: &str, bytes: usize, rank: usize) -> Vec<u8> {
    if kind == "SG" {
        return (0..16).flat_map(|peer| source(index,bytes,peer)).collect();
    }
    assert!(kind == "AR" || kind == "RS", "unsupported fixture reference");
    let cycle: Vec<u8> = (0..(bytes/2).min(PERIOD))
        .flat_map(|i| reduced_word(index,i).to_le_bytes()).collect();
    if kind == "RS" {
        let shard=bytes/16;
        cycle.iter().copied().cycle().skip((rank*shard)%cycle.len()).take(shard).collect()
    } else { repeat_to(&cycle,bytes) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inputs_have_rank_operation_and_shard_identity() {
        assert_ne!(word(7,0,0),word(7,1,0));
        assert_ne!(word(7,0,0),word(8,0,0));
        assert_ne!(word(7,0,0),word(7,0,917504));
        assert_eq!(word(7,3,12345),word(7,3,12345+PERIOD));
    }
    #[test]
    fn references_preserve_rs_rows_and_rank_major_shards() {
        let total=16*7168*2;
        let full=expected(1,"AR",total,0);
        for rank in 0..16 {
            assert_eq!(expected(1,"RS",total,rank),full[rank*total/16..(rank+1)*total/16]);
        }
        let gathered=expected(4,"SG",7168*2,0);
        for rank in 0..16 {
            assert_eq!(gathered[rank*14336..(rank+1)*14336],source(4,14336,rank));
        }
    }
    #[test]
    fn bf16_reference_matches_independent_nearest_even_search() {
        for index in [0,3,7,8] {
            for element in [0,1,1023,1024,131071,917503,917504] {
                let sum: f64=(0..16).map(|r| f32::from_bits((word(index,r,element) as u32)<<16) as f64).sum();
                let nearest=(0x3c00u16..0x4500).min_by(|a,b| {
                    let a_error=(f32::from_bits((*a as u32)<<16) as f64-sum).abs();
                    let b_error=(f32::from_bits((*b as u32)<<16) as f64-sum).abs();
                    a_error.total_cmp(&b_error).then((a&1).cmp(&(b&1)))
                }).unwrap();
                assert_eq!(reduced_word(index,element),nearest);
            }
        }
    }
}
