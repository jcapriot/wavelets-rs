extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::spanned::Spanned;
use syn::{Ident, LitFloat, Result, Token, parse_macro_input};

#[derive(Debug)]
enum LiftingStep<T> {
    UpdateD { offset: isize, coefs: Vec<T> },
    UpdateS { offset: isize, coefs: Vec<T> },
    Scale { scale: T },
}

struct LiftingScheme<T> {
    name: Ident,
    steps: Vec<LiftingStep<T>>,
}

impl Parse for LiftingStep<f64> {
    fn parse(input: ParseStream) -> Result<Self> {
        let ident: Ident = input.parse()?;

        let content;
        syn::parenthesized!(content in input);

        match ident.to_string().as_str() {
            "UpdateD" | "UpdateS" => {
                let offset_literal: syn::LitInt = content.parse()?;
                let offset: isize = offset_literal.base10_parse()?;

                content.parse::<Token![,]>()?;

                let coeff_content;
                syn::bracketed!(coeff_content in content);
                let coefs: Vec<f64> = coeff_content
                    .parse_terminated(syn::LitFloat::parse, Token![,])?
                    .into_iter()
                    .map(|lit| lit.base10_parse().unwrap())
                    .collect();

                if ident == "UpdateD" {
                    Ok(Self::UpdateD { offset, coefs })
                } else {
                    Ok(Self::UpdateS { offset, coefs })
                }
            }
            "Scale" => {
                let lit: LitFloat = content.parse()?;
                let scale = lit.base10_parse()?;
                Ok(Self::Scale { scale })
            }
            _ => Err(input.error("unknown lifting step")),
        }
    }
}

impl Parse for LiftingScheme<f64> {
    fn parse(input: ParseStream) -> Result<Self> {
        let name = input.parse()?;
        input.parse::<Token![,]>()?;

        let steps: Vec<LiftingStep<f64>> = input
            .parse_terminated(LiftingStep::<f64>::parse, Token![,])?
            .into_iter()
            .collect();
        Ok(Self { name, steps })
    }
}

enum LiftingDirection {
    Forward,
    Inverse,
}

fn expand_lifting_step(
    step: &LiftingStep<f64>,
    direction: LiftingDirection,
) -> proc_macro2::TokenStream {
    match step {
        LiftingStep::UpdateD { offset, coefs } | LiftingStep::UpdateS { offset, coefs } => {
            let (l, r, is_s) = match step {
                LiftingStep::UpdateS { .. } => (quote! {s}, quote! {d}, true),
                LiftingStep::UpdateD { .. } => (quote! {d}, quote! {s}, false),
                _ => unreachable!(),
            };
            let l_iter_concat = format!("{}_iter", l);
            let l_iter = syn::Ident::new(&l_iter_concat, l.span());
            let l_i_concat = format!("{}_i", l);
            let l_i = syn::Ident::new(&l_i_concat, l.span());
            let r_i_concat = format!("{}_i", r);
            let r_i = syn::Ident::new(&r_i_concat, r.span());

            let update_op = match direction {
                LiftingDirection::Forward => quote! {+=},
                LiftingDirection::Inverse => quote! {-=},
            };

            let n_coefs = coefs.len();
            let n_front = std::cmp::max(0, -offset) as usize;
            let max_offset = n_coefs as isize - 1 - offset;
            let n_back = std::cmp::max(0, max_offset) as usize;

            let terms = coefs.iter().map(|c| {
                quote! {
                    T::from(#c)
                }
            });

            let mut loop_body = quote! {
                let c = (#(#terms), * ,);
                let mut #l_iter = (0..#l.len() as isize).zip(#l.iter_mut());
            };

            if *offset < 0 {
                let accumulators =
                    coefs
                        .iter()
                        .enumerate()
                        .filter(|(_, v)| **v != 0.0)
                        .map(|(j, _)| {
                            let i_off = offset + j as isize;
                            let j = syn::Index::from(j);
                            quote! {
                                c.#j.clone() * bc.get_bc(#r, i + #i_off)
                            }
                        });
                loop_body = quote! {
                    #loop_body
                    for (i, #l_i) in #l_iter.by_ref().take(#n_front){
                        *#l_i #update_op #(#accumulators)+*;
                    }
                };
            }

            let r_iter = if *offset > 0 {
                let ind = syn::Index::from(*offset as usize);
                quote! {#r[#ind..]}
            } else {
                quote! {#r}
            };

            let is_back_loop = n_back > 0 || is_s;
            // ensure the iterator is consumed if there is no back loop
            let main_loop_l_iter = {
                if is_back_loop {
                    quote! {#l_iter.by_ref()}
                } else {
                    quote! {#l_iter}
                }
            };

            if n_coefs > 1 {
                let accumulators =
                    coefs
                        .iter()
                        .enumerate()
                        .filter(|(_, v)| **v != 0.0)
                        .map(|(j, _)| {
                            let j = syn::Index::from(j);
                            quote! {
                                c.#j.clone() * #r_i[#j].clone()
                            }
                        });

                loop_body = quote! {
                    #loop_body
                    #main_loop_l_iter
                        .zip(#r_iter.windows(#n_coefs))
                        .for_each(|((_, #l_i), #r_i)|{
                            *#l_i #update_op #(#accumulators)+*;
                        });
                }
            } else {
                loop_body = quote! {
                    #loop_body
                    #main_loop_l_iter
                        .zip(#r_iter.iter())
                        .for_each(|((_, #l_i), #r_i)|{
                            *#l_i #update_op c.0.clone() * #r_i.clone();
                        });
                };
            }
            if is_back_loop {
                let accumulators =
                    coefs
                        .iter()
                        .enumerate()
                        .filter(|(_, v)| **v != 0.0)
                        .map(|(j, _)| {
                            let i_off = offset + j as isize;
                            let j = syn::Index::from(j);
                            quote! {
                                c.#j.clone() * bc.get_bc(#r, i + #i_off)
                            }
                        });
                loop_body = quote! {
                    #loop_body
                    for (i, #l_i) in #l_iter{
                        *#l_i #update_op #(#accumulators)+*;
                    }
                };
            }
            loop_body
        }
        LiftingStep::Scale { scale } => {
            let scale_step = quote! {let scaling = T::from(#scale);};

            let (s_op, d_op) = match direction {
                LiftingDirection::Forward => (quote! {*=}, quote! {/=}),
                LiftingDirection::Inverse => (quote! {/=}, quote! {*=}),
            };

            quote! {
                #scale_step
                s.iter_mut().for_each(|s_i|{
                    *s_i #s_op scaling.clone();
                });
                d.iter_mut().for_each(|d_i|{
                    *d_i #d_op scaling.clone();
                });
            }
        }
    }
}

fn generate_forward_op(steps: &[LiftingStep<f64>]) -> proc_macro2::TokenStream {
    let mut func_body = quote! {
        assert!(d.len() == s.len() || d.len() + 1 == s.len(), "detail and scaling coefficient arrays must have compatible lengths");
    };
    for (_i, step) in steps.iter().enumerate() {
        let step_ts = expand_lifting_step(step, LiftingDirection::Forward);
        func_body.extend(step_ts);
    }

    quote! {
        fn forward<T, BC>(s: &mut [T], d: &mut [T], bc: &BC)
        where
            T: ::num_traits::Num
                + ::num_traits::NumAssignOps
                + Clone
                + From<f64>,
            BC: crate::lwt::BoundaryExtension
        {
            #func_body
        }
    }
    .into()
}

fn generate_inverse_op(steps: &[LiftingStep<f64>]) -> proc_macro2::TokenStream {
    let mut func_body = quote! {
        assert!(d.len() == s.len() || d.len() + 1 == s.len(), "detail and scaling coefficient arrays must have compatible lengths");
    };
    for (_i, step) in steps.iter().enumerate().rev() {
        let step_ts = expand_lifting_step(step, LiftingDirection::Inverse);
        func_body.extend(step_ts);
    }

    quote! {
        fn inverse<T, BC>(s: &mut [T], d: &mut [T], bc: &BC)
        where
            T: ::num_traits::Num
                + ::num_traits::NumAssignOps
                + Clone
                + From<f64>,
            BC: crate::lwt::BoundaryExtension
        {
            #func_body
        }
    }
    .into()
}

#[proc_macro]
pub fn implement_lifting_scheme(input: TokenStream) -> TokenStream {
    let scheme = parse_macro_input!(input with LiftingScheme::<f64>::parse);
    let LiftingScheme::<f64> { name, steps } = scheme;
    //println!("parsed steps: {:?}", steps);
    //let scheme = parse_macro_input!(input as LiftingScheme<f64>);
    //expand_forward_scheme(&scheme)
    let forward_func = generate_forward_op(&steps);
    let inverse_func = generate_inverse_op(&steps);

    quote! {

    impl crate::lwt::LiftingTransform for #name {
            #forward_func
            #inverse_func
    }
    }
    .into()
}

struct OrthogonalDWT<T> {
    name: Ident,
    g: Vec<T>,
}

impl Parse for OrthogonalDWT<f64> {
    fn parse(input: ParseStream) -> Result<Self> {
        let name = input.parse()?;
        input.parse::<Token![,]>()?;

        let coeff_content;
        syn::bracketed!(coeff_content in input);
        let g: Vec<f64> = coeff_content
            .parse_terminated(syn::LitFloat::parse, Token![,])?
            .into_iter()
            .map(|lit| lit.base10_parse().unwrap())
            .collect();
        Ok(Self { name, g })
    }
}

#[proc_macro]
pub fn implement_dwt_orthogonal(input: TokenStream) -> TokenStream {
    let OrthogonalDWT { name, g } = parse_macro_input!(input as OrthogonalDWT<f64>);
    let gi = g.clone().into_iter().rev().collect::<Vec<_>>();
    let h = gi
        .iter()
        .enumerate()
        .map(|(i, &v)| match i % 2 {
            0 => -v,
            1 => v,
            _ => unreachable!(),
        })
        .collect::<Vec<_>>();
    let hi = h.clone().into_iter().rev().collect::<Vec<_>>();

    quote! {
    impl crate::dwt::DiscreteTransform<f64, {#name::WIDTH}> for #name {
            const G: [f64; #name::WIDTH] = [#(#g), *];
            const H: [f64; #name::WIDTH] = [#(#h), *];
            const GI: [f64; #name::WIDTH] = [#(#gi), *];
            const HI: [f64; #name::WIDTH] = [#(#hi), *];
    }
    }
    .into()
}

struct BiorthogonalDWT<T> {
    name: Ident,
    g: Vec<T>,
    h: Vec<T>,
}

impl Parse for BiorthogonalDWT<f64> {
    fn parse(input: ParseStream) -> Result<Self> {
        let name = input.parse()?;
        input.parse::<Token![,]>()?;

        let coeff_content;
        syn::bracketed!(coeff_content in input);
        let g: Vec<f64> = coeff_content
            .parse_terminated(syn::LitFloat::parse, Token![,])?
            .into_iter()
            .map(|lit| lit.base10_parse().unwrap())
            .collect();

        input.parse::<Token![,]>()?;

        let coeff_content;
        syn::bracketed!(coeff_content in input);
        let h: Vec<f64> = coeff_content
            .parse_terminated(syn::LitFloat::parse, Token![,])?
            .into_iter()
            .map(|lit| lit.base10_parse().unwrap())
            .collect();
        Ok(Self { name, g, h })
    }
}

#[proc_macro]
pub fn implement_dwt_biorthogonal(input: TokenStream) -> TokenStream {
    let BiorthogonalDWT { name, g, h } = parse_macro_input!(input as BiorthogonalDWT<f64>);
    let hi = g
        .iter()
        .enumerate()
        .map(|(i, &v)| match i % 2 {
            0 => v,
            1 => -v,
            _ => unreachable!(),
        })
        .collect::<Vec<_>>();

    let gi = h
        .iter()
        .enumerate()
        .map(|(i, &v)| match i % 2 {
            0 => -v,
            1 => v,
            _ => unreachable!(),
        })
        .collect::<Vec<_>>();

    quote! {
    impl crate::dwt::DiscreteTransform<f64, {#name::WIDTH}> for #name {
            const G: [f64; #name::WIDTH] = [#(#g), *];
            const H: [f64; #name::WIDTH] = [#(#h), *];
            const GI: [f64; #name::WIDTH] = [#(#gi), *];
            const HI: [f64; #name::WIDTH] = [#(#hi), *];
    }
    }
    .into()
}
