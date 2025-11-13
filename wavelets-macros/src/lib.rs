extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Type};

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

    let first_generic = input.generics.params.first()
        .expect("struct must have at least one generic parameter");

    let generic_ident = match first_generic {
        syn::GenericParam::Type(type_param) => &type_param.ident,
        _ => panic!("first generic must be a type parameter"),
    };

    let steps_field = data.fields.iter()
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
