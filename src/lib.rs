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
    state : State
}


enum State {Nothing, Mean, Full}
impl OnlineStats {
    pub fn new(width: usize) -> OnlineStats {
        OnlineStats {
            sum_of_weights: 0.,
            mean: V::zeros(width),
            m2:   V::zeros(width),
            state: State::Nothing }
    }

    pub fn next_value(&mut self, data : &V, weight : f64) {
        let temp = weight + self.sum_of_weights;
        let delta = data - &self.mean;
        let r = &delta * (weight / temp);

        self.mean.iadd(&r);

        self.m2.iadd(&(self.sum_of_weights * (r * &delta)));
        self.sum_of_weights = temp;
        self.state = match self.state {
            State::Nothing => State::Mean,
            _ => State::Full
        };
    }

    fn nans(&self) -> V {V::from_elem(self.mean.dim(), f64::NAN)}

    pub fn mean_and_variance(&self) -> (V, V) {
        // With no data, there is no mean. With only one vector, there is no variance.
        match self.state {
            State::Nothing => (self.nans(), self.nans()),
            State::Mean => (self.mean.clone(), self.nans()),
            State::Full => (self.mean.clone(), &self.m2 / self.sum_of_weights),
        }
    }
}

use ndarray::arr1;

#[test]
fn weighted_stats17(){
    let const_vec = arr1(&[1., 3., 4.]);
    let const_vec2 = arr1(&[1., 1.5, 1.]);
    let const_vec3 = arr1(&[0., 2., 4.]);
    let mut s = OnlineStats::new(3);
    let (m0, v0) = s.mean_and_variance();
    let nans = V::from_elem(const_vec.dim(), f64::NAN);
    assert!(m0[0].is_nan());
    assert!(v0[0].is_nan());
    s.next_value(&const_vec, 1.);
    let (m1, v1) = s.mean_and_variance();
    assert!(v1[0].is_nan());
    assert_eq!(m1, const_vec);
    s.next_value(&const_vec2, 2.);
    let (m2, _) = s.mean_and_variance();
    assert_eq!(m2, arr1(&[1., 2., 2.]));
    s.next_value(&const_vec3, 3.);
    let (m3, _) = s.mean_and_variance();
    assert_eq!(m3, arr1(&[0.5, 2., 3.]));
}
