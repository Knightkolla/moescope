use moescope::router::top_k_route;

#[test]
fn test_top_k_route() {
    let scores = vec![0.1, 0.9, 0.3, 0.7];
    let result = top_k_route(&scores, 2);
    println!("Result: {:?}", result);
    assert_eq!(result, vec![1, 3]);
}
use moescope::router::batch_top_k_route;

#[test]
fn test_batch_top_k_route() {
    let scores = vec![vec![0.1, 0.9, 0.3, 0.7], vec![0.5, 0.2, 0.8, 0.4]];
    let result = batch_top_k_route(&scores, 2);
    println!("Result: {:?}", result);
    assert_eq!(result, vec![vec![1, 3], vec![2, 0],]);
}
