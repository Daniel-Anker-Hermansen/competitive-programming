use std::collections::HashSet;

use competitive_programming::AvlTree;

#[test]
fn set() {
    let mut hash = HashSet::new();
    let mut avl = AvlTree::new();
    for _ in 0..200 {
        let rand = fastrand::u64(0..100);
        assert_eq!(hash.contains(&rand), avl.contains(&rand));
        hash.insert(rand);
        avl.insert(rand);
    }
}
