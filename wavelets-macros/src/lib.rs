extern crate proc_macro;

use proc_macro::TokenStream;
use proc_macro2::TokenTree;
use quote::{ToTokens, quote};
use syn::parse::{Parse, ParseStream};
use syn::spanned::Spanned;
use syn::{Ident, LitFloat, Result, Token, parse_macro_input};

const WVLTS: [&'static str; 7] = [
    "Daubechies1",
    "Daubechies2",
    "Daubechies3",
    "Daubechies4",
    "Daubechies5",
    "Daubechies6",
    "Bior3_1",
];

fn wavelet_idents() -> Vec<Ident> {
    WVLTS
        .iter()
        .cloned()
        .map(|v| Ident::new(v, v.span()))
        .collect::<Vec<_>>()
}

#[proc_macro]
pub fn generate_wavelet_enum(input: TokenStream) -> TokenStream {
    let name = parse_macro_input!(input as syn::Ident);
    let wvlts = wavelet_idents();
    quote! {
        #[derive(Clone, Copy, Debug)]
        pub enum #name{
            #(#wvlts), *
        }
    }
    .into()
}

struct WaveletMatchArms {
    enum_name: Ident,
    var_name: Ident,
    template: proc_macro2::TokenStream,
}

impl Parse for WaveletMatchArms {
    fn parse(input: ParseStream) -> Result<Self> {
        let enum_name: Ident = input.parse()?;
        input.parse::<Token![,]>()?;
        let var_name: Ident = input.parse()?;
        input.parse::<Token![,]>()?;
        let content;
        syn::braced!(content in input);
        let template: proc_macro2::TokenStream = content.parse()?;

        Ok(Self {
            enum_name,
            var_name,
            template,
        })
    }
}

fn substitute(template: &proc_macro2::TokenStream, ident: &Ident) -> proc_macro2::TokenStream {
    let mut out = proc_macro2::TokenStream::new();

    for tt in template.into_token_stream() {
        match tt {
            TokenTree::Punct(p) if p.as_char() == '#' => {
                // expect #wvlt
                // skip and replace
            }
            TokenTree::Ident(i) if i == "wvlt" => {
                out.extend(quote!(#ident));
            }
            other => out.extend(std::iter::once(other)),
        }
    }

    out
}

#[proc_macro]
pub fn generate_wavelet_match_arms(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as WaveletMatchArms);
    let WaveletMatchArms {
        enum_name,
        var_name,
        template,
    } = input;

    let mut match_arms = proc_macro2::TokenStream::new();

    for v in WVLTS {
        let ident = Ident::new(v, v.span());
        let replaced = substitute(&template, &ident);
        match_arms = quote! {
            #match_arms
            #enum_name::#ident => #replaced
        };
    }

    let out = quote! {
        match #var_name {
            #match_arms
        }
    };

    out.into()
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

impl Parse for LiftingStep<LitFloat> {
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
                let coefs: Vec<LitFloat> = coeff_content
                    .parse_terminated(syn::LitFloat::parse, Token![,])?
                    .into_iter()
                    .collect();

                if ident == "UpdateD" {
                    Ok(Self::UpdateD { offset, coefs })
                } else {
                    Ok(Self::UpdateS { offset, coefs })
                }
            }
            "Scale" => {
                let scale: LitFloat = content.parse()?;
                Ok(Self::Scale { scale })
            }
            _ => Err(input.error("unknown lifting step")),
        }
    }
}

impl Parse for LiftingScheme<LitFloat> {
    fn parse(input: ParseStream) -> Result<Self> {
        let name = input.parse()?;
        input.parse::<Token![,]>()?;

        let steps: Vec<LiftingStep<LitFloat>> = input
            .parse_terminated(LiftingStep::<LitFloat>::parse, Token![,])?
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
    step: &LiftingStep<LitFloat>,
    direction: LiftingDirection,
) -> proc_macro2::TokenStream {
    match step {
        LiftingStep::UpdateD {
            offset,
            coefs: lit_coefs,
        }
        | LiftingStep::UpdateS {
            offset,
            coefs: lit_coefs,
        } => {
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

            let coefs = lit_coefs
                .iter()
                .map(|v| v.base10_parse().unwrap())
                .collect::<Vec<f64>>();

            let n_coefs = coefs.len();
            let n_front = std::cmp::max(0, -offset) as usize;
            let max_offset = n_coefs as isize - 1 + offset;

            let terms = coefs.iter().map(|c| {
                quote! {
                    T::ScalarType::from(#c)
                }
            });

            let mut loop_body = quote! {
                let c = (#(#terms), * ,);
                let mut #l_iter = (0..#l.len() as isize).zip(#l.iter_mut());
            };

            if *offset < 0 {
                let accumulators = lit_coefs.iter().enumerate().filter_map(|(j, v)| {
                    if let Ok(v) = v.base10_parse::<f64>()
                        && v == 0.0
                    {
                        None
                    } else {
                        let i_off = offset + j as isize;
                        let j = syn::Index::from(j);
                        Some(quote! {
                            bc.get_bc(#r, i + #i_off).and_then(|v| Some(v * c.#j.clone()))
                        })
                    }
                });
                loop_body = quote! {
                    #loop_body
                    for (i, #l_i) in #l_iter.by_ref().take(#n_front){
                        let vs = [#(#accumulators), *];
                        if let Some(v) = vs.into_iter().filter_map(|v| v).reduce(|acc, v| acc + v){
                            *#l_i #update_op v;
                        }
                    }
                };
            }

            let r_iter = if *offset > 0 {
                let ind = syn::Index::from(*offset as usize);
                quote! {#r[#ind..]}
            } else {
                quote! {#r}
            };

            let is_back_loop = match is_s {
                true => max_offset > -1,
                false => max_offset > 0,
            };
            // ensure the iterator is consumed if there is no back loop
            let main_loop_l_iter = {
                if is_back_loop {
                    quote! {#l_iter.by_ref()}
                } else {
                    quote! {#l_iter}
                }
            };

            let accumulators = coefs.iter().enumerate().filter_map(|(j, v)| {
                if *v == 0.0 {
                    None
                } else {
                    let j = syn::Index::from(j);
                    Some(quote! {
                        #r_i[#j].clone() * c.#j.clone()
                    })
                }
            });

            loop_body = quote! {
                #loop_body
                #r_iter.windows(#n_coefs)
                    .zip(#main_loop_l_iter)
                    .for_each(|(#r_i, (_, #l_i))|{
                        *#l_i #update_op #(#accumulators)+*;
                    });
            };
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
                                bc.get_bc(#r, i + #i_off).and_then(|v| Some(v * c.#j.clone()))
                            }
                        });
                loop_body = quote! {
                    #loop_body
                    for (i, #l_i) in #l_iter{
                        let vs = [#(#accumulators), *];
                        if let Some(v) = vs.into_iter().filter_map(|v| v).reduce(|acc, v| acc + v){
                            *#l_i #update_op v;
                        }
                        //*#l_i #update_op #(#accumulators)+*;
                    }
                };
            }
            loop_body
        }
        LiftingStep::Scale { scale } => {
            let scale_step = quote! {let scaling = T::ScalarType::from(#scale);};

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

fn expand_lifting_step_chunk(
    step: &LiftingStep<LitFloat>,
    direction: LiftingDirection,
) -> proc_macro2::TokenStream {
    match step {
        LiftingStep::UpdateD {
            offset,
            coefs: lit_coefs,
        }
        | LiftingStep::UpdateS {
            offset,
            coefs: lit_coefs,
        } => {
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
            let nr = format!("n{}", r);
            let nr = syn::Ident::new(&nr, nr.span());

            let update_op = match direction {
                LiftingDirection::Forward => quote! {+=},
                LiftingDirection::Inverse => quote! {-=},
            };

            let coefs = lit_coefs
                .iter()
                .map(|v| v.base10_parse().unwrap())
                .collect::<Vec<f64>>();

            let n_coefs = coefs.len();
            let n_front = std::cmp::max(0, -offset) as usize;
            let max_offset = n_coefs as isize - 1 + offset;

            let terms = coefs.iter().map(|c| {
                quote! {
                    T::ScalarType::from(#c)
                }
            });

            let mut loop_body = quote! {
                let c = (#(#terms), * ,);
                let mut #l_iter = (0..#l.len() as isize).zip(#l.chunks_exact_mut(chunk_size));
            };

            if *offset < 0 {
                let bc_accumulators = coefs.iter().enumerate().filter_map(|(j, v)| {
                    if *v == 0.0 {
                        None
                    } else {
                        let i_off = offset + j as isize;
                        let j = syn::Index::from(j);
                        Some(quote! {
                            let vs = bc.get_parts::<T>(#nr, i + #i_off);
                            for (v, io) in vs{
                                if let Some(v) = v{
                                    let c = c.#j.clone() * v;
                                    #l_i.iter_mut()
                                        .zip(&#r[io * chunk_size..(io + 1) * chunk_size])
                                        .for_each(|(#l_i, #r_i)|{
                                            *#l_i #update_op #r_i.clone() * c.clone();
                                        });
                                }else{
                                    #l_i.iter_mut()
                                        .zip(&#r[io * chunk_size..(io + 1) * chunk_size])
                                        .for_each(|(#l_i, #r_i)|{
                                            *#l_i #update_op #r_i.clone() * c.#j.clone();
                                        });
                                }
                            }
                        })
                    }
                });

                loop_body = quote! {
                    #loop_body

                    for (i, #l_i) in #l_iter.by_ref().take(#n_front){
                        #(#bc_accumulators) *
                    }
                };
            }

            let is_back_loop = match is_s {
                true => max_offset > -1,
                false => max_offset > 0,
            };
            // ensure the iterator is consumed if there is no back loop
            let main_loop_l_iter = {
                if is_back_loop {
                    quote! {#l_iter.by_ref()}
                } else {
                    quote! {#l_iter}
                }
            };

            let r_slices = coefs
                .iter()
                .enumerate()
                .filter_map(|(j, v)| {
                    if *v == 0.0 {
                        None
                    } else {
                        let slc_start = if *offset > 0 {
                            syn::Index::from(j + *offset as usize)
                        } else {
                            syn::Index::from(j)
                        };
                        Some(quote! {
                            #r[#slc_start * chunk_size..].chunks_exact(chunk_size)
                        })
                    }
                })
                .collect::<Vec<_>>();

            let rs = coefs
                .iter()
                .enumerate()
                .filter_map(|(j, v)| {
                    if *v == 0.0 {
                        None
                    } else {
                        let r_j = format!("{}_{}", r, j);
                        Some(syn::Ident::new(&r_j, r_j.span()))
                    }
                })
                .collect::<Vec<_>>();

            let r_chunk_iter = quote! {
                izip!(#(#r_slices), *)
            };
            let r_params = if rs.len() == 1 {
                let r = &rs[0];
                quote! { #r}
            } else {
                quote! {
                    (#(#rs), *)
                }
            };
            let r_zip = if rs.len() == 1 {
                quote! { (#r_params)}
            } else {
                quote! {
                    #r_params
                }
            };

            let cs = coefs.iter().enumerate().filter_map(|(j, v)| {
                if *v == 0.0 {
                    None
                } else {
                    let j = syn::Index::from(j);
                    Some(quote! {
                        c.#j
                    })
                }
            });

            let accumulators = rs.iter().zip(cs).map(|(r, c)| {
                quote! {
                    #r.clone() * #c.clone()
                }
            });

            loop_body = quote! {
                #loop_body
                #r_chunk_iter
                    .zip(#main_loop_l_iter)
                    .for_each(|(#r_params, (_, #l_i))|{
                        #l_i.iter_mut()
                            .zip(izip!#r_zip)
                            .for_each(|(#l_i, #r_params)|{
                                *#l_i #update_op #(#accumulators)+*;
                            });
                    });
            };
            if is_back_loop {
                let bc_accumulators = coefs.iter().enumerate().filter_map(|(j, v)| {
                    if *v == 0.0 {
                        None
                    } else {
                        let i_off = offset + j as isize;
                        let j = syn::Index::from(j);
                        Some(quote! {
                            let vs = bc.get_parts::<T>(#nr, i + #i_off);
                            for (v, io) in vs{
                                if let Some(v) = v{
                                    let c = c.#j.clone() * v;
                                    #l_i.iter_mut()
                                        .zip(&#r[io * chunk_size..(io + 1) * chunk_size])
                                        .for_each(|(#l_i, #r_i)|{
                                            *#l_i #update_op #r_i.clone() * c.clone();
                                        });
                                }else{
                                    #l_i.iter_mut()
                                        .zip(&#r[io * chunk_size..(io + 1) * chunk_size])
                                        .for_each(|(#l_i, #r_i)|{
                                            *#l_i #update_op #r_i.clone() * c.#j.clone();
                                        });
                                }
                            }
                        })
                    }
                });

                loop_body = quote! {
                    #loop_body
                    for (i, #l_i) in #l_iter{
                        #(#bc_accumulators) *
                    }
                };
            }
            loop_body
        }
        LiftingStep::Scale { scale } => {
            let scale_step = quote! {let scaling = T::ScalarType::from(#scale);};

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

fn expand_adjoint_lifting_step(
    step: &LiftingStep<LitFloat>,
    direction: LiftingDirection,
) -> proc_macro2::TokenStream {
    match step {
        LiftingStep::UpdateD { offset, coefs } | LiftingStep::UpdateS { offset, coefs } => {
            let (l, r, is_s) = match step {
                LiftingStep::UpdateS { .. } => (quote! {d}, quote! {s}, true),
                LiftingStep::UpdateD { .. } => (quote! {s}, quote! {d}, false),
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

            let coefs = coefs
                .iter()
                .map(|v| v.base10_parse().unwrap())
                .collect::<Vec<f64>>();

            let n_coefs = coefs.len();
            let n_front = std::cmp::max(0, -offset) as usize;
            let max_offset = n_coefs as isize - 1 + offset;
            let n_back = std::cmp::max(0, max_offset) as usize;

            let offset_r = -max_offset;

            let terms = coefs.iter().rev().map(|c| {
                quote! {
                    T::ScalarType::from(#c)
                }
            });

            let mut loop_body = quote! {
                let c = [#(#terms), *];
            };

            if n_front > 0 {
                loop_body = quote! {
                    #loop_body
                    for i in 0..#n_front as isize{
                        let i_left = i + #offset;
                        let i_right = i_left + #offset_r;
                        bc.adjoint_op(|v, x| *v #update_op x, #l, #r, #offset_r, &c, i_left);
                    }
                };
            }
            loop_body = quote! {
                #loop_body
                let mut #l_iter = (0..#l.len() as isize).zip(#l.iter_mut());
            };
            if n_back > 0 {
                let accumulators = coefs.iter().enumerate().filter_map(|(j, v)| {
                    if *v == 0.0 {
                        None
                    } else {
                        let i_off = offset_r + j as isize;
                        let j = syn::Index::from(j);
                        Some(quote! {#r.get((i + #i_off) as usize).cloned().and_then(|v| Some(v * c[#j].clone()))})
                    }
                });
                loop_body = quote! {
                    #loop_body
                    for (i, #l_i) in #l_iter.by_ref().take(#n_back){
                        let vals = [#(#accumulators),*];
                        if let Some(v) = vals.into_iter().filter_map(|v| v).reduce(|acc, v| acc + v){
                            *#l_i #update_op v;
                        }
                    }
                }
            }

            let r_iter = if offset_r > 0 {
                let ind = syn::Index::from(offset_r as usize);
                quote! {#r[#ind..]}
            } else {
                quote! {#r}
            };

            let is_back_loop = match is_s {
                true => n_front > 0,
                false => true,
            };
            // ensure the iterator is consumed if there is no back loop
            let main_loop_l_iter = {
                if is_back_loop {
                    quote! {#l_iter.by_ref()}
                } else {
                    quote! {#l_iter}
                }
            };
            let accumulators = coefs.iter().enumerate().filter_map(|(j, v)| {
                if *v == 0.0 {
                    None
                } else {
                    let j = syn::Index::from(j);
                    Some(quote! {
                        #r_i[#j].clone() * c[#j].clone()
                    })
                }
            });

            loop_body = quote! {
                #loop_body
                #r_iter.windows(#n_coefs)
                    .zip(#main_loop_l_iter)
                    .for_each(|(#r_i, (_, #l_i))|{
                        *#l_i #update_op #(#accumulators)+*;
                    });
            };
            if is_back_loop {
                let accumulators = coefs
                    .iter()
                    .enumerate()
                    .filter_map(|(j, v)| {
                        if *v == 0.0 {
                            None
                        }else{
                            let i_off = offset_r + j as isize;
                            let j = syn::Index::from(j);
                            Some(quote!{#r.get((i + #i_off) as usize).cloned().and_then(|v| Some(v * c[#j].clone()))})
                        }
                    });
                loop_body = quote! {
                    #loop_body
                    for (i, #l_i) in #l_iter{
                        let vals = [#(#accumulators),*];
                        if let Some(v) = vals.into_iter().filter_map(|v| v).reduce(|acc, v| acc + v){
                            *#l_i #update_op v;
                        }
                    }
                };
            }
            if is_s || n_back > 0 {
                loop_body = quote! {
                    #loop_body
                    let n_l = #l.len() as isize;
                    let n_r = #r.len() as isize;
                    for i_left in n_l..(n_r + #n_back as isize){
                        bc.adjoint_op(|v, x| *v #update_op x, #l, #r, #offset_r, &c, i_left);
                    }
                }
            }

            loop_body
        }
        LiftingStep::Scale { scale } => {
            let scale_step = quote! {let scaling = T::ScalarType::from(#scale);};

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

fn generate_forward_op(steps: &[LiftingStep<LitFloat>]) -> proc_macro2::TokenStream {
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
            T: crate::Transformable,
            T::ScalarType: From<f64>,
            BC: crate::boundarys::BoundaryExtension
        {
            #func_body
        }
    }
    .into()
}

fn generate_forward_chunk_op(steps: &[LiftingStep<LitFloat>]) -> proc_macro2::TokenStream {
    let mut func_body = quote! {
        assert_eq!(s.len() % chunk_size, 0, "smooth coefficient slice length must be a multiple of chunk_size.");
        assert_eq!(d.len() % chunk_size, 0, "detail coefficient slice length must be a multiple of chunk_size.");
        let ns = s.len() / chunk_size;
        let nd = d.len() / chunk_size;
        assert!(ns == nd || nd + 1 == ns, "detail and smooth coefficient arrays must have compatible lengths, got {nd} d-chunks and {ns} s-chunks.");
    };
    for (_i, step) in steps.iter().enumerate() {
        let step_ts = expand_lifting_step_chunk(step, LiftingDirection::Forward);
        func_body.extend(step_ts);
    }

    let temp = quote! {
        fn forward_chunk<T, BC>(s: &mut [T], d: &mut [T], chunk_size: usize, bc: &BC)
        where
            T: crate::Transformable,
            T::ScalarType: From<f64>,
            BC: crate::boundarys::BoundaryExtension
        {
            use ::itertools::izip;
            #func_body
        }
    };
    temp.into()
}

fn generate_inverse_op(steps: &[LiftingStep<LitFloat>]) -> proc_macro2::TokenStream {
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
            T: crate::Transformable,
            T::ScalarType: From<f64>,
            BC: crate::lwt::BoundaryExtension
        {
            #func_body
        }
    }
    .into()
}

fn generate_adjoint_inverse_op(steps: &[LiftingStep<LitFloat>]) -> proc_macro2::TokenStream {
    let mut func_body = quote! {
        assert!(d.len() == s.len() || d.len() + 1 == s.len(), "detail and scaling coefficient arrays must have compatible lengths");
    };
    for (_i, step) in steps.iter().enumerate() {
        let step_ts = expand_adjoint_lifting_step(step, LiftingDirection::Inverse);
        func_body.extend(step_ts);
    }

    quote! {
        fn adjoint_inverse<T, BC>(s: &mut [T], d: &mut [T], bc: &BC)
        where
            T: crate::Transformable,
            T::ScalarType: From<f64>,
            BC: crate::lwt::LiftedAdjointBoundary
        {
            #func_body
        }
    }
    .into()
}

fn generate_adjoint_forward_op(steps: &[LiftingStep<LitFloat>]) -> proc_macro2::TokenStream {
    let mut func_body = quote! {
        assert!(d.len() == s.len() || d.len() + 1 == s.len(), "detail and scaling coefficient arrays must have compatible lengths");
    };
    for (_i, step) in steps.iter().enumerate().rev() {
        let step_ts = expand_adjoint_lifting_step(step, LiftingDirection::Forward);
        func_body.extend(step_ts);
    }

    let temp = quote! {
        fn adjoint_forward<T, BC>(s: &mut [T], d: &mut [T], bc: &BC)
        where
            T: crate::Transformable,
            T::ScalarType: From<f64>,
            BC: crate::lwt::LiftedAdjointBoundary
        {
            #func_body
        }
    };

    temp.into()
}

#[proc_macro]
pub fn implement_lifting_scheme(input: TokenStream) -> TokenStream {
    let scheme = parse_macro_input!(input with LiftingScheme::<LitFloat>::parse);
    let LiftingScheme::<LitFloat> { name, steps } = scheme;
    //println!("parsed steps: {:?}", steps);
    //let scheme = parse_macro_input!(input as LiftingScheme<f64>);
    //expand_forward_scheme(&scheme)
    let forward_func = generate_forward_op(&steps);
    let inverse_func = generate_inverse_op(&steps);
    let adj_fwd_func = generate_adjoint_forward_op(&steps);
    let adj_inv_func = generate_adjoint_inverse_op(&steps);

    let forward_chunk_func = generate_forward_chunk_op(&steps);

    quote! {

    impl crate::lwt::LiftingTransform for #name {
            #forward_func
            #inverse_func
            #adj_fwd_func
            #adj_inv_func
            #forward_chunk_func
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
