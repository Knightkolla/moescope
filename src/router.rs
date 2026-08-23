//partial-selection is a known optimization as we do not need to sort all the tokens
// we can only sort the k ones but that will be delt in the later phase ig.
//Could use select_nth_unstable_by
// for O(E) partial selection if E grows large or this becomes a bottleneck
pub fn top_k_route(scores :&[f32],k:usize) -> Vec<usize>{
    let mut indexed_scores:Vec<(usize,f32)> = scores
    .iter()
    .enumerate()
    .map(|(i,&s)|(i,s))
    .collect();
    indexed_scores.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
    indexed_scores
        .into_iter()
        .take(k)
        .map(|(i, _)|i)
        .collect()
}
//deref coercion -it let you pass other datatypes 
pub fn batch_top_k_route(scores: &[Vec<f32>], k: usize) -> Vec<Vec<usize>> {
    scores
        .iter()
        .map(|token_scores| top_k_route(token_scores, k))
        .collect()
}