pub use crate::lwt::LiftingTransform;
pub use crate::wavelets::bior::*;
use wavelets_macros::implement_lifting_scheme;

implement_lifting_scheme! {
    Bior3_1,
    UpdateS(1, [-0.333333333333333333333333333333333333333333333333333333333333]),
    UpdateD(-1, [-0.375, 1.125]),
    Scale(0.942809041582063365867792482806465385713114583584632048784453)
}
