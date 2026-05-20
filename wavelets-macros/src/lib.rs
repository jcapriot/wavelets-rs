extern crate proc_macro;

//use proc_macro::TokenStream as PMTS;
use proc_macro2::{TokenStream, TokenTree};
use quote::{ToTokens, quote};
use syn::ext::IdentExt;
use syn::parse::{Parse, ParseStream};
use syn::spanned::Spanned;
use syn::{Ident, LitFloat, Result, Token, parse_macro_input};

const WVLTS: [&'static str; 33] = [
    "Daubechies1",
    "Daubechies2",
    "Daubechies3",
    "Daubechies4",
    "Daubechies5",
    "Daubechies6",
    "Daubechies7",
    "Daubechies8",
    "Daubechies9",
    "Daubechies10",
    "Symlet4",
    "Symlet5",
    "Symlet6",
    "Coiflet2",
    "Coiflet3",
    "Bior1_3",
    "Bior1_5",
    "Bior2_2",
    "Bior2_4",
    "Bior2_6",
    "Bior2_8",
    "Bior3_1",
    "Bior3_3",
    "Bior3_5",
    "Bior3_7",
    "Bior3_9",
    "Bior4_2",
    "Bior4_4",
    "Bior4_6",
    "Bior5_5",
    "Bior6_8",
    "CDF5_3",
    "CDF9_7",
];

fn wavelet_idents() -> Vec<Ident> {
    WVLTS
        .iter()
        .cloned()
        .map(|v| Ident::new(v, v.span()))
        .collect::<Vec<_>>()
}

struct WaveletEnum {
    enum_name: Ident,
    derives: Vec<Ident>,
    extras: Option<TokenStream>,
}

impl Parse for WaveletEnum {
    fn parse(input: ParseStream) -> Result<Self> {
        let enum_name: Ident = input.parse()?;
        input.parse::<Token![,]>()?;

        let derives_content;
        syn::parenthesized!(derives_content in input);
        let derives: Vec<Ident> = derives_content
            .parse_terminated(Ident::parse, Token![,])?
            .into_iter()
            .collect();
        let extras = if let Ok(_) = input.parse::<Token![,]>() {
            let extra_content;
            syn::braced!(extra_content in input);
            let extras: TokenStream = extra_content.parse()?;
            Some(extras)
        } else {
            None
        };

        Ok(Self {
            enum_name,
            derives,
            extras,
        })
    }
}

#[proc_macro]
pub fn generate_wavelet_enum(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let WaveletEnum {
        enum_name,
        derives,
        extras,
    } = parse_macro_input!(input as WaveletEnum);
    let wvlts = wavelet_idents();
    let docs: Vec<String> = WVLTS
        .iter()
        .map(|v| format!("The {} wavelet.", v))
        .collect();
    quote! {
        #extras
        /// Wavelet family selector.
        ///
        /// Pass one of these variants to [`dwt::driver::WaveletTransform::new`] or
        /// [`lwt::driver::WaveletTransform::new`] to select the filter coefficients.
        #[derive(#(#derives),*)]
        pub enum #enum_name {
            #(#[doc = #docs] #wvlts),*
        }
    }
    .into()
}

struct WaveletMatchArms {
    enum_name: Ident,
    var_name: Ident,
    template: TokenStream,
}

impl Parse for WaveletMatchArms {
    fn parse(input: ParseStream) -> Result<Self> {
        let enum_name: Ident = input.call(Ident::parse_any)?;
        input.parse::<Token![,]>()?;
        let var_name: Ident = input.call(Ident::parse_any)?;
        input.parse::<Token![,]>()?;
        let content;
        syn::braced!(content in input);
        let template: TokenStream = content.parse()?;

        Ok(Self {
            enum_name,
            var_name,
            template,
        })
    }
}

fn substitute(template: TokenStream, ident: &Ident) -> TokenStream {
    let mut out = TokenStream::new();

    for tt in template.into_token_stream() {
        match tt {
            TokenTree::Punct(p) if p.as_char() == '#' => {
                // expect #wvlt
                // skip and replace
            }
            TokenTree::Ident(i) if i == "wvlt" => {
                out.extend(quote!(#ident));
            }
            TokenTree::Group(g) => {
                let inner = substitute(g.stream(), ident);

                let mut new_group = proc_macro2::Group::new(g.delimiter(), inner);
                new_group.set_span(g.span()); // preserve span!

                out.extend(quote! {#new_group});
            }
            other => out.extend(std::iter::once(other)),
        }
    }

    out
}

#[proc_macro]
pub fn generate_wavelet_match_arms(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as WaveletMatchArms);
    let WaveletMatchArms {
        enum_name,
        var_name,
        template,
    } = input;

    let mut match_arms = TokenStream::new();

    for v in WVLTS {
        let ident = Ident::new(v, v.span());
        let replaced = substitute(template.clone(), &ident);
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

fn expand_lifting_step_simd(
    step: &LiftingStep<LitFloat>,
    direction: LiftingDirection,
) -> TokenStream {
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
            let l_i_concat = format!("{}_i", l);
            let l_i = syn::Ident::new(&l_i_concat, l.span());

            let (update_op, add_op, sub_op) = match direction {
                LiftingDirection::Forward => (quote! {+=}, quote!(+=), quote!(-=)),
                LiftingDirection::Inverse => (quote! {-=}, quote!(-=), quote!(+=)),
            };

            let (simd_mul_add_op, simd_add_op, simd_sub_op) = match direction {
                LiftingDirection::Forward => (
                    quote! {T::simd_mul_add},
                    quote!(T::simd_add),
                    quote!(T::simd_sub),
                ),
                LiftingDirection::Inverse => (
                    quote! {T::simd_negate_mul_add},
                    quote!(T::simd_sub),
                    quote!(T::simd_add),
                ),
            };

            let coefs = lit_coefs
                .iter()
                .map(|v| v.base10_parse().unwrap())
                .collect::<Vec<f64>>();

            let n_coefs = coefs.len();
            let n_front = std::cmp::max(0, -offset) as usize;
            let max_offset = n_coefs as isize + offset;

            let terms = coefs.iter().map(|c| {
                quote! {
                    T::scalar_type_from_f64(#c)
                }
            });

            let no_cs = coefs.iter().all(|c| *c == 0.0 || *c == 1.0 || *c == -1.0);

            let mut loop_body = if no_cs {
                quote! {}
            } else {
                quote! {
                    let c = (#(#terms), * ,);
                }
            };

            if *offset < 0 {
                loop_body.extend(quote! {
                    let n1 = std::cmp::min(#n_front, #l.len());
                });

                let accumulators = coefs.iter().enumerate().filter_map(|(j, &v)| {
                    let i_off = offset + j as isize;
                    let j = syn::Index::from(j);
                    if v == 0.0 {
                        None
                    } else if v == 1.0 {
                        Some(quote! {
                            if let Some(r_i) = bc.get_bc(#r, i + #i_off){
                                *#l_i #add_op r_i;
                            }
                        })
                    } else if v == -1.0 {
                        Some(quote! {
                            if let Some(r_i) = bc.get_bc(#r, i + #i_off){
                                *#l_i #sub_op r_i;
                            }
                        })
                    } else {
                        Some(quote! {
                            if let Some(r_i) = bc.get_bc(#r, i + #i_off){
                                *#l_i #update_op r_i * c.#j;
                            }
                        })
                    }
                });
                loop_body.extend(quote! {
                    (0..n1 as isize)
                        .zip(&mut #l[..n1])
                        .for_each(|(i, #l_i)| {
                            #(#accumulators)*
                        });
                });
            } else {
                loop_body.extend(quote! {
                    let n1 = 0;
                });
            }

            //let l_start = n_front;
            let r_start = std::cmp::max(0, *offset) as usize;

            let maybe_back_loop = match is_s {
                true => max_offset - 1 > -1, // if it is an s update, and the max offset is -1 or less, there will never be a back loop
                false => max_offset - 1 > 0, // if it is a d update and the max offset is 0 or less, there will never be a back loop
            };

            // main loop:
            if maybe_back_loop {
                loop_body.extend(quote! {
                    let ir_end = std::cmp::min(nd, #l.len().checked_add_signed(#max_offset).unwrap_or(0));
                });
            } else {
                loop_body.extend(quote! {
                    let ir_end = #l.len().checked_add_signed(#max_offset).unwrap_or(0);
                });
            }

            loop_body.extend(quote! {
                let nr = (ir_end).checked_sub(#n_coefs + #r_start).unwrap_or(0);
            });

            let cv_terms = (0..n_coefs).map(|i| syn::Index::from(i)).map(|i| {
                quote! {T::simd_splat(simd, c.#i)}
            });

            let mut main_loop_body = if no_cs {
                quote! {}
            } else {
                quote! {
                let cv = (#(#cv_terms), *, );
                }
            };

            main_loop_body.extend(quote! {

                let (l_h, l) = T::as_mut_simd(simd, &mut #l[n1..nr + n1]);
                let (l_h4, l_h) = l_h.as_chunks_mut::<4>();
            });

            let r_tokens = (0..n_coefs)
                .map(|i| {
                    let r_string = format!("r{i}");
                    syn::Ident::new(&r_string, r.span())
                })
                .collect::<Vec<_>>();

            let rh_tokens = (0..n_coefs)
                .map(|i| {
                    let r_string = format!("r{i}_h");
                    syn::Ident::new(&r_string, r.span())
                })
                .collect::<Vec<_>>();

            let rh4_tokens = (0..n_coefs)
                .map(|i| {
                    let r_string = format!("r{i}_h4");
                    syn::Ident::new(&r_string, r.span())
                })
                .collect::<Vec<_>>();

            r_tokens
                .iter()
                .zip(&rh_tokens)
                .zip(&rh4_tokens)
                .enumerate()
                .for_each(|(i, ((r_id, r_h), r_h4))| {
                    let ir = syn::Index::from(r_start + i);
                    main_loop_body.extend(quote! {
                        let (#r_h, #r_id) = T::as_simd(simd, &#r[#ir..nr + #ir]);
                        let (#r_h4, #r_h) = #r_h.as_chunks::<4>();

                        debug_assert_eq!(#r_h4.len(), l_h4.len());
                        debug_assert_eq!(#r_h.len(), l_h.len());
                        debug_assert_eq!(#r_id.len(), l.len());
                    });
                });

            let unrolled_accumulators =
                coefs
                    .iter()
                    .zip(r_tokens.iter())
                    .enumerate()
                    .filter_map(|(j, (&v, r))| {
                        let j = syn::Index::from(j);
                        if v == 0.0 {
                            None
                        } else if v == 1.0 {
                            Some(quote! {
                                *l0 = #simd_add_op(simd, *l0, #r[0]);
                                *l1 = #simd_add_op(simd, *l1, #r[1]);
                                *l2 = #simd_add_op(simd, *l2, #r[2]);
                                *l3 = #simd_add_op(simd, *l3, #r[3]);
                            })
                        } else if v == -1.0 {
                            Some(quote! {
                                *l0 = #simd_sub_op(simd, *l0, #r[0]);
                                *l1 = #simd_sub_op(simd, *l1, #r[1]);
                                *l2 = #simd_sub_op(simd, *l2, #r[2]);
                                *l3 = #simd_sub_op(simd, *l3, #r[3]);
                            })
                        } else {
                            Some(quote! {
                                *l0 = #simd_mul_add_op(simd, #r[0], cv.#j, *l0);
                                *l1 = #simd_mul_add_op(simd, #r[1], cv.#j, *l1);
                                *l2 = #simd_mul_add_op(simd, #r[2], cv.#j, *l2);
                                *l3 = #simd_mul_add_op(simd, #r[3], cv.#j, *l3);
                            })
                        }
                    });

            let simd_accumulators =
                coefs
                    .iter()
                    .zip(r_tokens.iter())
                    .enumerate()
                    .filter_map(|(j, (&v, r))| {
                        let j = syn::Index::from(j);
                        if v == 0.0 {
                            None
                        } else if v == 1.0 {
                            Some(quote! {
                                *l = #simd_add_op(simd, *l, *#r);
                            })
                        } else if v == -1.0 {
                            Some(quote! {
                                *l = #simd_sub_op(simd, *l, *#r);
                            })
                        } else {
                            Some(quote! {
                                *l = #simd_mul_add_op(simd, *#r, cv.#j, *l);
                            })
                        }
                    });

            let accumulators =
                coefs
                    .iter()
                    .zip(r_tokens.iter())
                    .enumerate()
                    .filter_map(|(j, (&v, r))| {
                        let j = syn::Index::from(j);
                        if v == 0.0 {
                            None
                        } else if v == 1.0 {
                            Some(quote! {
                                *l #add_op #r.clone();
                            })
                        } else if v == -1.0 {
                            Some(quote! {
                                *l #sub_op #r.clone();
                            })
                        } else {
                            Some(quote! {
                                *l #add_op #r.clone() * c.#j;
                            })
                        }
                    });

            if rh_tokens.len() == 1 {
                main_loop_body.extend(quote! {
                    l_h4.iter_mut()
                        .zip(izip!(#(#rh4_tokens), *))
                        .for_each(|([l0, l1, l2, l3], r0)|{
                            #(#unrolled_accumulators)*
                        });

                    l_h.iter_mut()
                        .zip(izip!(#(#rh_tokens), *))
                        .for_each(|(l, r0)|{
                            #(#simd_accumulators)*
                        });

                    l.iter_mut()
                        .zip(izip!(#(#r_tokens), *))
                        .for_each(|(l, r0)|{
                            #(#accumulators)*
                        });
                });
            } else {
                main_loop_body.extend(quote! {
                    l_h4.iter_mut()
                        .zip(izip!(#(#rh4_tokens), *))
                        .for_each(|([l0, l1, l2, l3], (#(#r_tokens), *))|{
                            #(#unrolled_accumulators)*
                        });

                    l_h.iter_mut()
                        .zip(izip!(#(#rh_tokens), *))
                        .for_each(|(l, (#(#r_tokens), *))|{
                            #(#simd_accumulators)*
                        });

                    l.iter_mut()
                        .zip(izip!(#(#r_tokens), *))
                        .for_each(|(l, (#(#r_tokens), *))|{
                            #(#accumulators)*
                        });
                });
            }

            loop_body.extend(quote! {
                if nr > 0 {
                    #main_loop_body
                }
            });

            if maybe_back_loop {
                let accumulators = coefs.iter().enumerate().filter_map(|(j, &v)| {
                    let i_off = offset + j as isize;
                    let j = syn::Index::from(j);
                    if v == 0.0 {
                        None
                    } else if v == 1.0 {
                        Some(quote! {
                            if let Some(r_i) = bc.get_bc(#r, i + #i_off){
                                *#l_i #add_op r_i;
                            }
                        })
                    } else if v == -1.0 {
                        Some(quote! {
                            if let Some(r_i) = bc.get_bc(#r, i + #i_off){
                                *#l_i #sub_op r_i;
                            }
                        })
                    } else {
                        Some(quote! {
                            if let Some(r_i) = bc.get_bc(#r, i + #i_off){
                                *#l_i #update_op r_i * c.#j;
                            }
                        })
                    }
                });
                loop_body.extend(quote! {
                    let n2 = std::cmp::min(n1 + nr, #l.len());
                    (n2 as isize..#l.len() as isize)
                        .zip(&mut #l[n2..])
                        .for_each(|(i, #l_i)| {
                            #(#accumulators)*
                        });
                });
            }

            loop_body
        }

        LiftingStep::Scale { scale } => {
            let mut loop_body = match direction {
                LiftingDirection::Forward => {
                    quote! {
                        let s_scale = T::scalar_type_from_f64(#scale);
                        let d_scale = T::scalar_type_from_f64(1.0/#scale);
                    }
                }
                LiftingDirection::Inverse => {
                    quote! {
                        let s_scale = T::scalar_type_from_f64(1.0/#scale);
                        let d_scale = T::scalar_type_from_f64(#scale);
                    }
                }
            };

            loop_body.extend(quote! {

                let scaling_vec = T::simd_splat(simd, s_scale);

                let (s_h, s_t) = T::as_mut_simd(simd, s);
                let (s_h4, s_h) = s_h.as_chunks_mut::<4>();

                s_h4.iter_mut().for_each(|[s0, s1, s2, s3]| {
                    *s0 = T::simd_mul(simd, *s0, scaling_vec);
                    *s1 = T::simd_mul(simd, *s1, scaling_vec);
                    *s2 = T::simd_mul(simd, *s2, scaling_vec);
                    *s3 = T::simd_mul(simd, *s3, scaling_vec);
                });
                s_h.iter_mut()
                    .for_each(|s| *s = T::simd_mul(simd, *s, scaling_vec));
                s_t.iter_mut().for_each(|s| *s *= s_scale.clone());

                let scaling_vec = T::simd_splat(simd, d_scale);

                let (d_h, d_t) = T::as_mut_simd(simd, d);
                let (d_h4, d_h) = d_h.as_chunks_mut::<4>();
                d_h4.iter_mut().for_each(|[d0, d1, d2, d3]| {
                    *d0 = T::simd_mul(simd, *d0, scaling_vec);
                    *d1 = T::simd_mul(simd, *d1, scaling_vec);
                    *d2 = T::simd_mul(simd, *d2, scaling_vec);
                    *d3 = T::simd_mul(simd, *d3, scaling_vec);
                });
                d_h.iter_mut()
                    .for_each(|d| *d = T::simd_mul(simd, *d, scaling_vec));
                d_t.iter_mut().for_each(|d| *d *= d_scale.clone());
            });

            loop_body
        }
    }
}

fn expand_lifting_step_chunk(
    step: &LiftingStep<LitFloat>,
    direction: LiftingDirection,
) -> TokenStream {
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
                    T::scalar_type_from_f64(#c)
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
            let scale_step = quote! {let scaling = T::scalar_type_from_f64(#scale);};

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

fn expand_adjoint_lifting_step_simd(
    step: &LiftingStep<LitFloat>,
    direction: LiftingDirection,
) -> TokenStream {
    match step {
        LiftingStep::UpdateD { offset, coefs } | LiftingStep::UpdateS { offset, coefs } => {
            let (l, r, is_s) = match step {
                LiftingStep::UpdateS { .. } => (quote! {d}, quote! {s}, true),
                LiftingStep::UpdateD { .. } => (quote! {s}, quote! {d}, false),
                _ => unreachable!(),
            };
            let l_i_concat = format!("{}_i", l);
            let l_i = syn::Ident::new(&l_i_concat, l.span());

            let (update_op, add_op, sub_op) = match direction {
                LiftingDirection::Forward => (quote! {+=}, quote!(+=), quote!(-=)),
                LiftingDirection::Inverse => (quote! {-=}, quote!(-=), quote!(+=)),
            };

            let (simd_mul_add_op, simd_add_op, simd_sub_op) = match direction {
                LiftingDirection::Forward => (
                    quote! {T::simd_mul_add},
                    quote!(T::simd_add),
                    quote!(T::simd_sub),
                ),
                LiftingDirection::Inverse => (
                    quote! {T::simd_negate_mul_add},
                    quote!(T::simd_sub),
                    quote!(T::simd_add),
                ),
            };

            let coefs = coefs
                .iter()
                .rev()
                .map(|v| v.base10_parse().unwrap())
                .collect::<Vec<f64>>();

            let n_coefs = coefs.len();
            let n_front = std::cmp::max(0, -offset) as usize;
            let max_offset = n_coefs as isize + offset;
            let n_back = std::cmp::max(0, max_offset - 1) as usize;

            let offset_r = -(max_offset - 1);
            let n_front_r = std::cmp::max(0, -offset_r) as usize;
            let max_offset_r = n_coefs as isize + offset_r;

            let terms = coefs.iter().map(|c| {
                quote! {
                    T::scalar_type_from_f64(#c)
                }
            });

            let no_cs = coefs.iter().all(|c| *c == 0.0 || *c == 1.0 || *c == -1.0);

            let mut loop_body = if no_cs {
                quote! {}
            } else {
                quote! {
                    let c = (#(#terms), * ,);
                }
            };
            if n_front > 0 {
                let accumulators = coefs.iter().enumerate().filter_map(|(j, &v)| {
                    let i_off = offset_r + j as isize;
                    let j = syn::Index::from(j);
                    if v == 0.0 {
                        None
                    } else if v == 1.0 {
                        Some(quote! {
                            if let Some(r_i) = #r.get((i_left + #i_off) as usize).cloned(){
                                r_sum = match r_sum{
                                    Some(v) => Some(v + r_i),
                                    None => Some(r_i)
                                };
                            }
                        })
                    } else if v == -1.0 {
                        Some(quote! {
                            if let Some(r_i) = #r.get((i_left + #i_off) as usize).cloned(){
                                r_sum = match r_sum{
                                    Some(v) => Some(v - r_i),
                                    None => Some(-r_i)
                                };
                            }
                        })
                    } else {
                        Some(quote! {
                            if let Some(r_i) = #r.get((i_left + #i_off) as usize).cloned(){
                                let r_i = r_i * c.#j;
                                r_sum = match r_sum{
                                    Some(v) => Some(v + r_i),
                                    None => Some(r_i)
                                };
                            }
                        })
                    }
                });

                loop_body = quote! {
                    #loop_body
                    for i in 0..#n_front as isize{
                        let i_left = i + #offset;

                        let parts = bc.get_parts::<T>(#l.len(), i_left);

                        for (scale, io) in parts {
                            if let Some(#l_i) = #l.get_mut(io) {

                                let mut r_sum = None;
                                #(#accumulators)*

                                match (r_sum, scale) {
                                    (Some(r), Some(v)) => *#l_i #update_op r * v,
                                    (Some(r), None) => *#l_i #update_op r,
                                    _ => {}
                                };
                            }
                        }
                    }
                };
            }

            if offset_r < 0 {
                loop_body.extend(quote! {
                    let n1 = std::cmp::min(#n_front_r, #l.len());
                });

                let accumulators = coefs.iter().enumerate().filter_map(|(j, &v)| {
                    let i_off = offset_r + j as isize;
                    let j = syn::Index::from(j);
                    if v == 0.0 {
                        None
                    } else if v == 1.0 {
                        Some(quote! {
                            if let Some(r_i) = #r.get((i + #i_off) as usize).cloned(){
                                *#l_i #add_op r_i;
                            }
                        })
                    } else if v == -1.0 {
                        Some(quote! {
                            if let Some(r_i) = #r.get((i + #i_off) as usize).cloned(){
                                *#l_i #sub_op r_i;
                            }
                        })
                    } else {
                        Some(quote! {
                            if let Some(r_i) = #r.get((i + #i_off) as usize).cloned(){
                                *#l_i #update_op r_i * c.#j;
                            }
                        })
                    }
                });

                loop_body.extend(quote! {
                    (0..n1 as isize)
                        .zip(&mut #l[..n1])
                        .for_each(|(i, #l_i)| {
                            #(#accumulators)*
                        });
                });
            } else {
                loop_body.extend(quote! {
                    let n1 = 0;
                });
            }

            //let l_start = n_front;
            let r_start = std::cmp::max(0, offset_r) as usize;

            let maybe_back_loop = match is_s {
                true => max_offset_r - 1 > 0, // if it is an s update adjoint (d update), and the max offset is -1 or less, there will never be a back loop
                false => max_offset_r - 1 > -1, // if it is a d update adjoint (s update), and the max offset is 0 or less, there will never be a back loop
            };

            // main loop:
            if maybe_back_loop {
                loop_body.extend(quote! {
                    let ir_end = std::cmp::min(nd, #l.len().checked_add_signed(#max_offset_r).unwrap_or(0));
                });
            } else {
                loop_body.extend(quote! {
                    let ir_end = #l.len().checked_add_signed(#max_offset_r).unwrap_or(0);
                });
            }

            loop_body.extend(quote! {
                let nr = (ir_end).checked_sub(#n_coefs + #r_start).unwrap_or(0);
            });

            let cv_terms = (0..n_coefs).map(|i| syn::Index::from(i)).map(|i| {
                quote! {T::simd_splat(simd, c.#i)}
            });

            let mut main_loop_body = if no_cs {
                quote! {}
            } else {
                quote! {
                let cv = (#(#cv_terms), *, );
                }
            };

            main_loop_body.extend(quote! {

                let (l_h, l) = T::as_mut_simd(simd, &mut #l[n1..nr + n1]);
                let (l_h4, l_h) = l_h.as_chunks_mut::<4>();
            });

            let r_tokens = (0..n_coefs)
                .map(|i| {
                    let r_string = format!("r{i}");
                    syn::Ident::new(&r_string, r.span())
                })
                .collect::<Vec<_>>();

            let rh_tokens = (0..n_coefs)
                .map(|i| {
                    let r_string = format!("r{i}_h");
                    syn::Ident::new(&r_string, r.span())
                })
                .collect::<Vec<_>>();

            let rh4_tokens = (0..n_coefs)
                .map(|i| {
                    let r_string = format!("r{i}_h4");
                    syn::Ident::new(&r_string, r.span())
                })
                .collect::<Vec<_>>();

            r_tokens
                .iter()
                .zip(&rh_tokens)
                .zip(&rh4_tokens)
                .enumerate()
                .for_each(|(i, ((r_id, r_h), r_h4))| {
                    let ir = syn::Index::from(r_start + i);
                    main_loop_body.extend(quote! {
                        let (#r_h, #r_id) = T::as_simd(simd, &#r[#ir..nr + #ir]);
                        let (#r_h4, #r_h) = #r_h.as_chunks::<4>();

                        debug_assert_eq!(#r_h4.len(), l_h4.len());
                        debug_assert_eq!(#r_h.len(), l_h.len());
                        debug_assert_eq!(#r_id.len(), l.len());
                    });
                });

            let unrolled_accumulators =
                coefs
                    .iter()
                    .zip(r_tokens.iter())
                    .enumerate()
                    .filter_map(|(j, (&v, r))| {
                        let j = syn::Index::from(j);
                        if v == 0.0 {
                            None
                        } else if v == 1.0 {
                            Some(quote! {
                                *l0 = #simd_add_op(simd, *l0, #r[0]);
                                *l1 = #simd_add_op(simd, *l1, #r[1]);
                                *l2 = #simd_add_op(simd, *l2, #r[2]);
                                *l3 = #simd_add_op(simd, *l3, #r[3]);
                            })
                        } else if v == -1.0 {
                            Some(quote! {
                                *l0 = #simd_sub_op(simd, *l0, #r[0]);
                                *l1 = #simd_sub_op(simd, *l1, #r[1]);
                                *l2 = #simd_sub_op(simd, *l2, #r[2]);
                                *l3 = #simd_sub_op(simd, *l3, #r[3]);
                            })
                        } else {
                            Some(quote! {
                                *l0 = #simd_mul_add_op(simd, #r[0], cv.#j, *l0);
                                *l1 = #simd_mul_add_op(simd, #r[1], cv.#j, *l1);
                                *l2 = #simd_mul_add_op(simd, #r[2], cv.#j, *l2);
                                *l3 = #simd_mul_add_op(simd, #r[3], cv.#j, *l3);
                            })
                        }
                    });

            let simd_accumulators =
                coefs
                    .iter()
                    .zip(r_tokens.iter())
                    .enumerate()
                    .filter_map(|(j, (&v, r))| {
                        let j = syn::Index::from(j);
                        if v == 0.0 {
                            None
                        } else if v == 1.0 {
                            Some(quote! {
                                *l = #simd_add_op(simd, *l, *#r);
                            })
                        } else if v == -1.0 {
                            Some(quote! {
                                *l = #simd_sub_op(simd, *l, *#r);
                            })
                        } else {
                            Some(quote! {
                                *l = #simd_mul_add_op(simd, *#r, cv.#j, *l);
                            })
                        }
                    });

            let accumulators =
                coefs
                    .iter()
                    .zip(r_tokens.iter())
                    .enumerate()
                    .filter_map(|(j, (&v, r))| {
                        let j = syn::Index::from(j);
                        if v == 0.0 {
                            None
                        } else if v == 1.0 {
                            Some(quote! {
                                *l #add_op #r.clone();
                            })
                        } else if v == -1.0 {
                            Some(quote! {
                                *l #sub_op #r.clone();
                            })
                        } else {
                            Some(quote! {
                                *l #add_op #r.clone() * c.#j;
                            })
                        }
                    });

            if rh_tokens.len() == 1 {
                main_loop_body.extend(quote! {
                    l_h4.iter_mut()
                        .zip(izip!(#(#rh4_tokens), *))
                        .for_each(|([l0, l1, l2, l3], r0)|{
                            #(#unrolled_accumulators)*
                        });

                    l_h.iter_mut()
                        .zip(izip!(#(#rh_tokens), *))
                        .for_each(|(l, r0)|{
                            #(#simd_accumulators)*
                        });

                    l.iter_mut()
                        .zip(izip!(#(#r_tokens), *))
                        .for_each(|(l, r0)|{
                            #(#accumulators)*
                        });
                });
            } else {
                main_loop_body.extend(quote! {
                    l_h4.iter_mut()
                        .zip(izip!(#(#rh4_tokens), *))
                        .for_each(|([l0, l1, l2, l3], (#(#r_tokens), *))|{
                            #(#unrolled_accumulators)*
                        });

                    l_h.iter_mut()
                        .zip(izip!(#(#rh_tokens), *))
                        .for_each(|(l, (#(#r_tokens), *))|{
                            #(#simd_accumulators)*
                        });

                    l.iter_mut()
                        .zip(izip!(#(#r_tokens), *))
                        .for_each(|(l, (#(#r_tokens), *))|{
                            #(#accumulators)*
                        });
                });
            }

            loop_body.extend(quote! {
                if nr > 0 {
                    #main_loop_body
                }
            });

            if maybe_back_loop {
                let accumulators = coefs.iter().enumerate().filter_map(|(j, &v)| {
                    let i_off = offset_r + j as isize;
                    let j = syn::Index::from(j);
                    if v == 0.0 {
                        None
                    } else if v == 1.0 {
                        Some(quote! {
                            if let Some(r_i) = #r.get((i + #i_off) as usize).cloned(){
                                *#l_i #add_op r_i;
                            }
                        })
                    } else if v == -1.0 {
                        Some(quote! {
                            if let Some(r_i) = #r.get((i + #i_off) as usize).cloned(){
                                *#l_i #sub_op r_i;
                            }
                        })
                    } else {
                        Some(quote! {
                            if let Some(r_i) = #r.get((i + #i_off) as usize).cloned(){
                                *#l_i #update_op r_i * c.#j;
                            }
                        })
                    }
                });
                loop_body.extend(quote! {
                    let n2 = std::cmp::min(n1 + nr, #l.len());
                    (n2 as isize..#l.len() as isize)
                        .zip(&mut #l[n2..])
                        .for_each(|(i, #l_i)| {
                            #(#accumulators)*
                        });
                });
            }

            // if there was a potential back loop in the normal transform operation.
            let maybe_back_loop_norm = match is_s {
                true => max_offset - 1 > -1, // if it is an s update, and the max offset is -1 or less, there will never be a back loop
                false => max_offset - 1 > 0, // if it is a d update and the max offset is 0 or less, there will never be a back loop
            };

            if maybe_back_loop_norm {
                let accumulators = coefs.iter().enumerate().filter_map(|(j, &v)| {
                    let i_off = offset_r + j as isize;
                    let j = syn::Index::from(j);
                    if v == 0.0 {
                        None
                    } else if v == 1.0 {
                        Some(quote! {
                            if let Some(r_i) = #r.get((i_left + #i_off) as usize).cloned(){
                                r_sum = match r_sum{
                                    Some(v) => Some(v + r_i),
                                    None => Some(r_i)
                                };
                            }
                        })
                    } else if v == -1.0 {
                        Some(quote! {
                            if let Some(r_i) = #r.get((i_left + #i_off) as usize).cloned(){
                                r_sum = match r_sum{
                                    Some(v) => Some(v - r_i),
                                    None => Some(-r_i)
                                };
                            }
                        })
                    } else {
                        Some(quote! {
                            if let Some(r_i) = #r.get((i_left + #i_off) as usize).cloned(){
                                let r_i = r_i * c.#j;
                                r_sum = match r_sum{
                                    Some(v) => Some(v + r_i),
                                    None => Some(r_i)
                                };
                            }
                        })
                    }
                });

                loop_body.extend(quote! {

                    let n_l = #l.len() as isize;
                    let n_r = #r.len() as isize;
                    for i_left in n_l as isize..(n_r + #n_back as isize){

                        let parts = bc.get_parts::<T>(#l.len(), i_left);

                        for (scale, io) in parts {
                            if let Some(#l_i) = #l.get_mut(io) {

                                let mut r_sum = None;
                                #(#accumulators)*

                                match (r_sum, scale) {
                                    (Some(r), Some(v)) => *#l_i #update_op r * v,
                                    (Some(r), None) => *#l_i #update_op r,
                                    _ => {}
                                };
                            }
                        }
                    }
                });

                // loop_body.extend(quote! {
                //     #loop_body
                //     let n_l = #l.len() as isize;
                //     let n_r = #r.len() as isize;
                //     for i_left in n_l..(n_r + #n_back as isize){
                //         bc.adjoint_op(|v, x| *v #update_op x, #l, #r, #offset_r, &c, i_left);
                //     }
                // });
            }

            loop_body
        }
        LiftingStep::Scale { scale } => {
            let mut loop_body = match direction {
                LiftingDirection::Forward => {
                    quote! {
                        let s_scale = T::scalar_type_from_f64(#scale);
                        let d_scale = T::scalar_type_from_f64(1.0/#scale);
                    }
                }
                LiftingDirection::Inverse => {
                    quote! {
                        let s_scale = T::scalar_type_from_f64(1.0/#scale);
                        let d_scale = T::scalar_type_from_f64(#scale);
                    }
                }
            };

            loop_body.extend(quote! {

                let scaling_vec = T::simd_splat(simd, s_scale);

                let (s_h, s_t) = T::as_mut_simd(simd, s);
                let (s_h4, s_h) = s_h.as_chunks_mut::<4>();

                s_h4.iter_mut().for_each(|[s0, s1, s2, s3]| {
                    *s0 = T::simd_mul(simd, *s0, scaling_vec);
                    *s1 = T::simd_mul(simd, *s1, scaling_vec);
                    *s2 = T::simd_mul(simd, *s2, scaling_vec);
                    *s3 = T::simd_mul(simd, *s3, scaling_vec);
                });
                s_h.iter_mut()
                    .for_each(|s| *s = T::simd_mul(simd, *s, scaling_vec));
                s_t.iter_mut().for_each(|s| *s *= s_scale.clone());

                let scaling_vec = T::simd_splat(simd, d_scale);

                let (d_h, d_t) = T::as_mut_simd(simd, d);
                let (d_h4, d_h) = d_h.as_chunks_mut::<4>();
                d_h4.iter_mut().for_each(|[d0, d1, d2, d3]| {
                    *d0 = T::simd_mul(simd, *d0, scaling_vec);
                    *d1 = T::simd_mul(simd, *d1, scaling_vec);
                    *d2 = T::simd_mul(simd, *d2, scaling_vec);
                    *d3 = T::simd_mul(simd, *d3, scaling_vec);
                });
                d_h.iter_mut()
                    .for_each(|d| *d = T::simd_mul(simd, *d, scaling_vec));
                d_t.iter_mut().for_each(|d| *d *= d_scale.clone());
            });

            loop_body
        }
    }
}

fn generate_forward_chunk_op(steps: &[LiftingStep<LitFloat>]) -> TokenStream {
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

    quote! {
        fn forward_chunk<T, BC>(s: &mut [T], d: &mut [T], chunk_size: usize, bc: &BC)
        where
            T: crate::Transformable,
            BC: crate::boundarys::BoundaryExtension
        {
            use ::itertools::izip;
            #func_body
        }
    }
}

fn generate_forward_op(steps: &[LiftingStep<LitFloat>]) -> TokenStream {
    let mut func_body = quote! {
        assert!(d.len() == s.len() || d.len() + 1 == s.len(), "detail and scaling coefficient arrays must have compatible lengths");
    };
    for (_i, step) in steps.iter().enumerate() {
        let step_ts = expand_lifting_step_simd(step, LiftingDirection::Forward);
        func_body.extend(step_ts);
    }

    quote! {
        fn forward<T, BC>(s: &mut [T], d: &mut [T], bc: &BC)
        where
            T: crate::simd::SimdTransformable,
            BC: crate::boundarys::BoundaryExtension
        {
            use ::itertools::izip;
            use crate::simd::Dispatch;

            struct Impl<'a, 'b, 'c, T, BC>(&'a mut [T], &'b mut [T], &'c BC);
            impl<'a, 'b, 'c, T, BC> crate::simd::WithSimd for Impl<'a, 'b, 'c, T, BC>
            where
                T: crate::simd::SimdTransformable,
                BC: crate::boundarys::BoundaryExtension
            {
                type Output = ();
                #[inline(always)]
                fn with_simd<S: crate::simd::Simd>(self, simd: S) -> Self::Output {
                    let s = self.0;
                    let d = self.1;
                    let bc = self.2;

                    let _ns = s.len();
                    let nd = d.len();

                    #func_body
                }
            }

            crate::simd::ARCH.dispatch_wvlt(Impl(s, d, bc));
        }
    }
}

fn generate_inverse_op(steps: &[LiftingStep<LitFloat>]) -> TokenStream {
    let mut func_body = quote! {
        assert!(d.len() == s.len() || d.len() + 1 == s.len(), "detail and scaling coefficient arrays must have compatible lengths");
    };
    for (_i, step) in steps.iter().enumerate().rev() {
        let step_ts = expand_lifting_step_simd(step, LiftingDirection::Inverse);
        func_body.extend(step_ts);
    }

    quote! {
        fn inverse<T, BC>(s: &mut [T], d: &mut [T], bc: &BC)
        where
            T: crate::simd::SimdTransformable,
            BC: crate::boundarys::BoundaryExtension
        {
            use ::itertools::izip;
            use crate::simd::Dispatch;

            struct Impl<'a, 'b, 'c, T, BC>(&'a mut [T], &'b mut [T], &'c BC);
            impl<'a, 'b, 'c, T, BC> crate::simd::WithSimd for Impl<'a, 'b, 'c, T, BC>
            where
                T: crate::simd::SimdTransformable,
                BC: crate::boundarys::BoundaryExtension
            {
                type Output = ();
                #[inline(always)]
                fn with_simd<S: crate::simd::Simd>(self, simd: S) -> Self::Output {
                    let s = self.0;
                    let d = self.1;
                    let bc = self.2;

                    let _ns = s.len();
                    let nd = d.len();

                    #func_body
                }
            }

            crate::simd::ARCH.dispatch_wvlt(Impl(s, d, bc));
        }
    }
}

fn generate_adjoint_inverse_op(steps: &[LiftingStep<LitFloat>]) -> TokenStream {
    let mut func_body = quote! {
        assert!(d.len() == s.len() || d.len() + 1 == s.len(), "detail and scaling coefficient arrays must have compatible lengths");
    };
    for (_i, step) in steps.iter().enumerate() {
        let step_ts = expand_adjoint_lifting_step_simd(step, LiftingDirection::Inverse);
        func_body.extend(step_ts);
    }

    quote! {
        fn adjoint_inverse<T, BC>(s: &mut [T], d: &mut [T], bc: &BC)
        where
            T: crate::simd::SimdTransformable,
            BC: crate::boundarys::BoundaryExtension
        {
            use ::itertools::izip;
            use crate::simd::Dispatch;

            struct Impl<'a, 'b, 'c, T, BC>(&'a mut [T], &'b mut [T], &'c BC);
            impl<'a, 'b, 'c, T, BC> crate::simd::WithSimd for Impl<'a, 'b, 'c, T, BC>
            where
                T: crate::simd::SimdTransformable,
                BC: crate::boundarys::BoundaryExtension
            {
                type Output = ();
                #[inline(always)]
                fn with_simd<S: crate::simd::Simd>(self, simd: S) -> Self::Output {
                    let s = self.0;
                    let d = self.1;
                    let bc = self.2;

                    let _ns = s.len();
                    let nd = d.len();

                    #func_body
                }
            }

            crate::simd::ARCH.dispatch_wvlt(Impl(s, d, bc));
        }
    }
}

fn generate_adjoint_forward_op(steps: &[LiftingStep<LitFloat>]) -> TokenStream {
    let mut func_body = quote! {
        assert!(d.len() == s.len() || d.len() + 1 == s.len(), "detail and scaling coefficient arrays must have compatible lengths");
    };
    for (_i, step) in steps.iter().enumerate().rev() {
        let step_ts = expand_adjoint_lifting_step_simd(step, LiftingDirection::Forward);
        func_body.extend(step_ts);
    }

    quote! {
        fn adjoint_forward<T, BC>(s: &mut [T], d: &mut [T], bc: &BC)
        where
            T: crate::simd::SimdTransformable,
            BC: crate::boundarys::BoundaryExtension
        {
            use ::itertools::izip;
            use crate::simd::Dispatch;

            struct Impl<'a, 'b, 'c, T, BC>(&'a mut [T], &'b mut [T], &'c BC);
            impl<'a, 'b, 'c, T, BC> crate::simd::WithSimd for Impl<'a, 'b, 'c, T, BC>
            where
                T: crate::simd::SimdTransformable,
                BC: crate::boundarys::BoundaryExtension
            {
                type Output = ();
                #[inline(always)]
                fn with_simd<S: crate::simd::Simd>(self, simd: S) -> Self::Output {
                    let s = self.0;
                    let d = self.1;
                    let bc = self.2;

                    let _ns = s.len();
                    let nd = d.len();

                    #func_body
                }
            }

            crate::simd::ARCH.dispatch_wvlt(Impl(s, d, bc));
        }
    }
}

#[proc_macro]
pub fn implement_lifting_scheme(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let scheme = parse_macro_input!(input with LiftingScheme::<LitFloat>::parse);
    let LiftingScheme::<LitFloat> { name, steps } = scheme;

    let forward_func = generate_forward_op(&steps);
    let inverse_func = generate_inverse_op(&steps);
    let adj_fwd_func = generate_adjoint_forward_op(&steps);
    let adj_inv_func = generate_adjoint_inverse_op(&steps);

    let forward_chunk_func = generate_forward_chunk_op(&steps);

    let temp = quote! {
        impl crate::lwt::LiftingTransform for #name {

                #forward_func

                #inverse_func

                #adj_fwd_func

                #adj_inv_func

                #forward_chunk_func
        }
    };
    temp.into()
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
pub fn implement_dwt_orthogonal(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
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
    let half_n: usize = g.len() / 2;

    quote! {
    impl crate::dwt::DiscreteTransform<{#name::WIDTH}, #half_n> for #name {
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
pub fn implement_dwt_biorthogonal(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
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
    let half_n: usize = g.len() / 2;

    quote! {
    impl crate::dwt::DiscreteTransform<{#name::WIDTH}, #half_n> for #name {
            const G: [f64; #name::WIDTH] = [#(#g), *];
            const H: [f64; #name::WIDTH] = [#(#h), *];
            const GI: [f64; #name::WIDTH] = [#(#gi), *];
            const HI: [f64; #name::WIDTH] = [#(#hi), *];
    }
    }
    .into()
}
