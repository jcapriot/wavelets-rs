extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{
    DeriveInput, Ident, Result, Token, Type,
    parse::{Parse, ParseStream},
    parse_macro_input,
};

use syn::LitFloat;
use syn::spanned::Spanned;

#[proc_macro_derive(LiftedTransform)]
pub fn derive_lifted_transform(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;
    let generics = &input.generics;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    // Find the `steps` field
    let data = match input.data {
        syn::Data::Struct(ref s) => s,
        _ => panic!("#[derive(LiftedTransform)] only works on structs"),
    };

    let first_generic = input
        .generics
        .params
        .first()
        .expect("struct must have at least one generic parameter");

    let generic_ident = match first_generic {
        syn::GenericParam::Type(type_param) => &type_param.ident,
        _ => panic!("first generic must be a type parameter"),
    };

    let steps_field = data
        .fields
        .iter()
        .find(|f| f.ident.as_ref().map(|id| id == "steps").unwrap_or(false))
        .expect("expected a field named `steps`");

    // Extract the type of the `steps` field
    let step_type = &steps_field.ty;

    // Count tuple elements (we expect the steps field to be a tuple)
    let tuple_elems = match step_type {
        Type::Tuple(t) => &t.elems,
        Type::Path(_) => panic!("`steps` must be a tuple type"),
        _ => panic!("unsupported steps type"),
    };

    let n = tuple_elems.len();
    let indices: Vec<_> = (0..n).collect();

    // Forward calls in order
    let forward_calls = indices.iter().map(|i| {
        let idx = syn::Index::from(*i);
        quote! { steps.#idx.forward(s, d, bc); }
    });

    // Inverse calls in reverse order
    let inverse_calls = indices.iter().rev().map(|i| {
        let idx = syn::Index::from(*i);
        quote! { steps.#idx.inverse(s, d, bc); }
    });

    let expanded = quote! {

        impl #impl_generics crate::lwt::LiftedTransform<#generic_ident> for #name #ty_generics
        #where_clause
        {

            type StepListType = #step_type;

            fn get_steps(&self) -> &Self::StepListType {
                &self.steps
            }

            fn forward<SD, BC>(&self, s: &mut [SD], d: &mut [SD], bc: &BC)
            where
                SD: ::num_traits::Num
                    + ::num_traits::NumAssignOps
                    + Copy
                    + ::std::ops::Mul<#generic_ident, Output=SD>
                    + ::std::ops::MulAssign<#generic_ident>,
                BC: crate::lwt::BoundaryExtension
            {

                use crate::lwt::steps::LiftedStep;
                let steps = self.get_steps();
                #(#forward_calls)*
            }

            fn inverse<SD, BC>(&self, s: &mut [SD], d: &mut [SD], bc: &BC)
            where
                SD: ::num_traits::Num
                    + ::num_traits::NumAssignOps
                    + Copy
                    + ::std::ops::Mul<#generic_ident, Output=SD>
                    + ::std::ops::MulAssign<#generic_ident>,
                BC: crate::lwt::BoundaryExtension
            {
                use crate::lwt::steps::LiftedStep;
                let steps = self.get_steps();
                #(#inverse_calls)*
            }
        }
    };

    TokenStream::from(expanded)
}

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
                                c.#j.clone() * BC::get_bc(#r, i + #i_off)
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
                                c.#j.clone() * BC::get_bc(#r, i + #i_off)
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
        fn forward<T, BC>(s: &mut [T], d: &mut [T], _bc: &BC)
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
        fn inverse<T, BC>(s: &mut [T], d: &mut [T], _bc: &BC)
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
