#[macro_use]
extern crate ndarray;

use std::f64;

use ndarray::{
    OwnedArray,
    Ix,
};

pub type V = OwnedArray<f64, Ix>;

pub struct OnlineStats {
    sum_of_weights : f64,
    mean : V,
    m2 : V,
    n : u64
}

impl OnlineStats {
    pub fn new(width: usize) -> OnlineStats {
        OnlineStats {
            sum_of_weights: 0.,
            mean: V::zeros(width),
            m2:   V::zeros(width),
            n: 0 }
    }

    pub fn next_value(&mut self, data : &V, weight : f64) {
        let temp = weight + self.sum_of_weights;
        // nice
        let delta = data - &self.mean;
        let r = &delta * (weight / temp); // Interoperation with scalars works nicely.

        self.mean.iadd(&r); // While I don't like this style, is efficient and will be += when rust supports it so fine.

        self.m2.iadd(&(self.sum_of_weights * (r * &delta)));
        self.sum_of_weights = temp;
        self.n += 1;
    }

    pub fn mean_and_variance(&self) -> (V, V) {
        //let var = if self.n > 1 {self.M2 / self.sumOfWeights * self.n / (self.n - 1) } else {Float::inf};
        let var = if self.n > 1 {
            &self.m2 * ((self.n as f64) / (self.sum_of_weights * ((self.n - 1) as f64)))
        } else {
            V::from_elem(self.m2.dim(), f64::INFINITY)
        };
        (self.mean.clone(), var)
    }
}

use ndarray::arr1;

#[test]
fn weighted_stats17(){
    let const_vec = arr1(&[1., 3., 4.]);
    let const_vec2 = arr1(&[1., 1.5, 1.]);
    let const_vec3 = arr1(&[0., 2., 4.]);
    let mut s = OnlineStats::new(3);
    s.next_value(&const_vec, 1.);
    let (m1, _) = s.mean_and_variance();
    assert_eq!(m1, const_vec);
    s.next_value(&const_vec2, 2.);
    let (m2, _) = s.mean_and_variance();
    assert_eq!(m2, arr1(&[1., 2., 2.]));
    s.next_value(&const_vec3, 3.);
    let (m3, _) = s.mean_and_variance();
    assert_eq!(m3, arr1(&[0.5, 2., 3.]));
}
