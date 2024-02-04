use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DataEnum, DataStruct, DeriveInput, Expr, Field};
use heck::{self, ToLowerCamelCase};

#[derive(Debug, Default)]
struct MagicAttrArgs {
    // The #[magic(description = "...")]
    description: Option<Expr>,
    // The #[magic(validator = "...")]
    validators: Vec<Expr>,
}

impl MagicAttrArgs {
    fn merge(&mut self, other: Self) {
        if other.description.is_some() {
            self.description = other.description;
        }
        self.validators.extend(other.validators);
    }
}

impl syn::parse::Parse for MagicAttrArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut description = None;
        let mut validators = vec![];
        while !input.is_empty() {
            let name: syn::Ident = input.parse()?;
            match name.to_string().as_str() {
                "description" => {
                    input.parse::<syn::Token![=]>()?;
                    let value: Expr = input.parse()?;
                    description = Some(value);
                },
                "validator" => {
                    input.parse::<syn::Token![=]>()?;
                    let value: syn::Expr = input.parse()?;
                    validators.push(value);
                },
                _ => return Err(syn::Error::new(name.span(), "Unknown attribute")),
            }
            if input.is_empty() {
                break;
            }
            input.parse::<syn::Token![,]>()?;
        }
        Ok(Self { description, validators })
    }
}


fn attributes<'a>(attrs: impl Iterator<Item = &'a syn::Attribute>) -> MagicAttrArgs {
    let mut result = MagicAttrArgs::default();
    for attr in attrs {
        if attr.path().is_ident("magic") {
            let attr_args: MagicAttrArgs = attr.parse_args().unwrap();
            result.merge(attr_args);
        }
    }
    result
}

fn field_attributes<'a>(fields: impl Iterator<Item = &'a Field>) -> Vec<MagicAttrArgs> {
    let mut results = vec![];
    for field in fields {
        results.push(attributes(field.attrs.iter()));
    }
    results
}


/// Derive the `MagicInstantiate` trait for a struct or enum.
/// Descriptions and validators can be added to fields using the `#[magic(description = ...)]` and `#[magic(validator = ...)]` attributes.
#[proc_macro_derive(MagicInstantiate, attributes(magic))]
pub fn derive_magic_instantiate(input: TokenStream) -> TokenStream {
    let DeriveInput { ident, data, generics, attrs, .. } = parse_macro_input!(input as DeriveInput);

    let attrs = attributes(attrs.iter());
    let definition_description = attrs.description.into_iter().collect::<Vec<_>>();
    let definition_validators = attrs
        .validators
        .iter()
        .map(|v| quote! { openai_magic_instantiate::Validator::<Self>::validate(&#v, &result)?; })
        .collect::<Vec<_>>();
    let definition_validator_instructions = attrs
        .validators
        .iter()
        .map(|v| quote! { openai_magic_instantiate::Validator::<Self>::instructions(&#v) })
        .collect::<Vec<_>>();

    let mut generics = generics.clone();
    for generic in generics.params.iter_mut() {
        if let syn::GenericParam::Type(type_param) = generic {
            type_param.bounds.push(syn::parse_quote!(MagicInstantiate));
        }
    }
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    match data {
        Data::Struct(DataStruct { fields, .. }) => {
            match &fields {
                syn::Fields::Unit => {
                    quote! {
                        impl #impl_generics MagicInstantiate for #ident #ty_generics #where_clause {
                            fn name() -> String {
                                stringify!(#ident).to_string()
                            }

                            fn reference() -> String {
                                ()::reference()
                            }

                            fn definition() -> String {
                                ()::definition()
                            }

                            fn add_dependencies(builder: &mut openai_magic_instantiate::TypeScriptAccumulator) -> String {
                                ()::add_dependencies(builder)
                            }
                    
                            fn validate(value: &openai_magic_instantiate::export::JsonValue) -> Result<Self, String> {
                                ()::validate(value)?;
                                Ok(Self)
                            }
                    
                            fn default_if_omitted() -> Option<Self> {
                                Some(Self)
                            }
                        }
                    }
                },
                syn::Fields::Unnamed(fields) => {
                    let field_types = fields.unnamed.iter().map(|f| &f.ty).collect::<Vec<_>>();
                    let field_indices = (0..field_types.len()).collect::<Vec<_>>();
                    let field_count = field_types.len();
                    let type_definition = if field_count == 1 {
                        quote! {
                            result.push_str(&format!("type {} = {};", stringify!(#ident), references[0]));
                        }
                    } else {
                        quote! {
                            result.push_str(&format!("type {} = [{}];", stringify!(#ident), references.join(", ")));
                        }
                    };
                    let validate_definition = if field_count == 1 {
                        let field_type = &field_types[0];
                        quote! {
                            let value = <#field_type>::validate(value)?;
                            let result = Self(value);
                        }
                    } else {
                        quote! {
                            let openai_magic_instantiate::export::JsonValue::Array(value) = value else { return Err(format!("Expected array tuple, got {}", openai_magic_instantiate::JsonValueExt::type_str(value))) };
                            if value.len() != #field_count {
                                return Err(format!("Expected {} elements but got {}", #field_count, value.len()));
                            }
                            let result = Self(#(#field_types::validate(&value[#field_indices])?),*);
                        }
                    };
                    quote! {
                        impl #impl_generics MagicInstantiate for #ident #ty_generics #where_clause {
                            fn name() -> String {
                                stringify!(#ident).to_string()
                            }

                            fn reference() -> String {
                                Self::name()
                            }

                            fn definition() -> String {
                                let mut result = String::new();
                                #(
                                    for line in #definition_description.lines() {
                                        result.push_str(&format!("// {}\n", line));
                                    }
                                )*
                                #(
                                    for line in #definition_validator_instructions.lines() {
                                        result.push_str(&format!("// {}\n", line));
                                    }
                                )*
                                let references = vec![#(<#field_types>::reference()),*];
                                #type_definition
                            }

                            fn add_dependencies(builder: &mut openai_magic_instantiate::TypeScriptAccumulator) {
                                #(
                                    builder.add::<#field_types>();
                                )*
                            }
                    
                            fn validate(value: &openai_magic_instantiate::export::JsonValue) -> Result<Self, String> {
                                #validate_definition
                                #(
                                    #definition_validators
                                )*
                                Ok(result)
                            }
                    
                            fn default_if_omitted() -> Option<Self> {
                                Some(#ident(#(<#field_types>::default_if_omitted()?),*))
                            }
                        }
                    }
                },
                syn::Fields::Named(fields) => {
                    let attributes = field_attributes(fields.named.iter());
                    let field_idents = fields.named.iter().map(|f| f.ident.as_ref().unwrap()).collect::<Vec<_>>();
                    let field_types = fields.named.iter().map(|f| &f.ty).collect::<Vec<_>>();
                    let field_names_camel = field_idents.iter().map(|f| f.to_string().to_lower_camel_case()).collect::<Vec<_>>();
                    let field_is_optionals = field_types.iter().map(|f| {
                        quote! {
                            if <#f>::default_if_omitted().is_some() { "?" } else { "" }
                        }
                    }).collect::<Vec<_>>();

                    let descriptions = attributes.iter().map(|a| {
                        if let Some(description) = &a.description {
                            quote! {
                                result.push_str(&format!("    // {}\n", #description));
                            }
                        } else {
                            quote! {}
                        }
                    }).collect::<Vec<_>>();

                    let validation_comments = attributes.iter().zip(&field_types).map(|(a, field_type)| {
                        let validators = &a.validators;
                        quote! {
                            #(
                                for line in openai_magic_instantiate::Validator::<#field_type>::instructions(&#validators).lines() {
                                    result.push_str(&format!("    // {}\n", line));
                                }
                            )*
                        }
                    }).collect::<Vec<_>>();

                    let field_validators = (0..field_types.len()).map(|i| {
                        let field_type = &field_types[i];
                        let validators = &attributes[i].validators;
                        quote! {
                            #(
                                openai_magic_instantiate::Validator::<#field_type>::validate(&#validators, &value)?;
                            )*
                        }
                    }).collect::<Vec<_>>();

                    quote! {
                        impl #impl_generics MagicInstantiate for #ident #ty_generics #where_clause {
                            fn name() -> String {
                                stringify!(#ident).to_string()
                            }

                            fn reference() -> String {
                                Self::name()
                            }

                            fn definition() -> String {
                                let mut result = String::new();
                                #(
                                    for line in #definition_description.lines() {
                                        result.push_str(&format!("// {}\n", line));
                                    }
                                )*
                                #(
                                    for line in #definition_validator_instructions.lines() {
                                        result.push_str(&format!("// {}\n", line));
                                    }
                                )*
                                result.push_str(&format!("type {} = {{\n", Self::name()));
                                #(
                                    #descriptions
                                    #validation_comments
                                    result.push_str(&format!("    {}{}: {};\n", #field_names_camel, #field_is_optionals, <#field_types>::reference()));
                                )*
                                result.push_str("};");
                                result
                            }

                            fn add_dependencies(builder: &mut openai_magic_instantiate::TypeScriptAccumulator) {
                                #(
                                    builder.add::<#field_types>();
                                )*
                            }
                    
                            fn validate(value: &openai_magic_instantiate::export::JsonValue) -> Result<Self, String> {
                                let openai_magic_instantiate::export::JsonValue::Object(value) = value else { return Err(format!("Expected object with fields {:?}, got {}", [#(#field_names_camel),*], openai_magic_instantiate::JsonValueExt::type_str(value))) };
                                let result = Self {
                                    #(
                                        #field_idents: {
                                            let value = match value.get(#field_names_camel) {
                                                None => match <#field_types>::default_if_omitted() {
                                                    Some(value) => value,
                                                    None => return Err(format!("Expected field {}, but it wasn't present", #field_names_camel)),
                                                },
                                                Some(value) => match <#field_types>::validate(value) {
                                                    Ok(value) => value,
                                                    Err(error) => return Err(format!("Validation error for field {}:\n{}", #field_names_camel, error)),
                                                }
                                            };
                                            #field_validators
                                            value
                                        },
                                    )*
                                };
                                #(
                                    #definition_validators
                                )*
                                Ok(result)
                            }
                    
                            fn default_if_omitted() -> Option<Self> {
                                Some(#ident {
                                    #(
                                        #field_idents: <#field_types>::default_if_omitted()?,
                                    )*
                                })
                            }
                        }
                    } 
                },
            }
        }
        Data::Enum(DataEnum { variants, .. }) => {
            let mut variant_definitions = vec![];
            let mut variant_struct_names = vec![];
            let mut variant_struct_kinds = vec![];
            let mut variant_struct_to_variants = vec![];

            for variant in variants {
                let variant_attributes = variant
                    .attrs
                    .iter()
                    .filter(|a| a.path().is_ident("magic"))
                    .collect::<Vec<_>>();

                let variant_ident = variant.ident;
                let variant_struct_name = syn::Ident::new(&format!("{}{}", ident, variant_ident), proc_macro2::Span::call_site());
                variant_struct_names.push(variant_struct_name.clone());

                let variant_struct_kind = syn::Ident::new(&format!("{}{}", variant_struct_name, variant_ident), proc_macro2::Span::call_site());
                variant_struct_kinds.push(variant_ident.clone());

                let mut variant_fields = vec![
                    quote! {
                        kind: #variant_struct_kind,
                    }
                ];

                match variant.fields {
                    syn::Fields::Unit => {
                        variant_struct_to_variants.push(quote! {
                            Ok(Self::#variant_ident)
                        });
                    },
                    syn::Fields::Unnamed(fields) => {
                        let field_types = fields.unnamed.iter().map(|f| &f.ty).collect::<Vec<_>>();
                        variant_fields.push(quote! {
                            value: (#(#field_types,)*),
                        });
                        let field_idents = (0..field_types.len()).map(|i| syn::Ident::new(&format!("field{}", i), proc_macro2::Span::call_site())).collect::<Vec<_>>();
                        variant_struct_to_variants.push(quote! {
                            let (#(#field_idents,)*) = value.value;
                            Ok(Self::#variant_ident(#(#field_idents),*))
                        });
                    },
                    syn::Fields::Named(fields) => {
                        for field in &fields.named {
                            let field_attributes = &field.attrs;
                            let field_name = field.ident.as_ref().unwrap();
                            let field_type = &field.ty;

                            variant_fields.push(quote! {
                                #(#field_attributes)*
                                #field_name: #field_type,
                            });
                        }
                        let field_idents = fields.named.iter().map(|f| f.ident.as_ref().unwrap()).collect::<Vec<_>>();
                        variant_struct_to_variants.push(quote! {
                            Ok(Self::#variant_ident {
                                #(#field_idents: value.#field_idents,)*
                            })
                        });
                    }
                }

                variant_definitions.push(quote! {

                    struct #variant_struct_kind;

                    impl MagicInstantiate for #variant_struct_kind {
                        fn name() -> String {
                            stringify!(#variant_ident).to_string()
                        }
                        fn reference() -> String {
                            format!("\"{}\"", stringify!(#variant_ident))
                        }
                        fn add_dependencies(builder: &mut openai_magic_instantiate::TypeScriptAccumulator) {}
                        fn definition() -> String { "".to_string() }

                        fn validate(value: &openai_magic_instantiate::export::JsonValue) -> Result<Self, String> {
                            let expected = stringify!(#variant_ident);
                            if value.as_str() == Some(expected.as_ref()) {
                                Ok(Self)
                            } else {
                                Err(format!("Expected \"{expected}\""))
                            }
                        }
                        fn default_if_omitted() -> Option<Self> { None }
                    }

                    #[derive(MagicInstantiate)]
                    #(#variant_attributes)*
                    struct #variant_struct_name {
                        #(#variant_fields)*
                    }
                });
            }

            quote! {
                #(#variant_definitions)*

                impl #impl_generics MagicInstantiate for #ident #ty_generics #where_clause {
                    fn name() -> String {
                        stringify!(#ident).to_string()
                    }

                    fn reference() -> String {
                        Self::name()
                    }

                    fn definition() -> String {
                        let mut result = String::new();
                        #(
                            for line in #definition_description.lines() {
                                result.push_str(&format!("// {}\n", line));
                            }
                        )*
                        #(
                            for line in #definition_validator_instructions.lines() {
                                result.push_str(&format!("// {}\n", line));
                            }
                        )*
                        result.push_str(&format!("type {} =\n", stringify!(#ident)));
                        #(
                            result.push_str(&format!("    | {}\n", <#variant_struct_names>::reference()));
                        )*
                        result.push_str(";");
                        result
                    }

                    fn add_dependencies(builder: &mut openai_magic_instantiate::TypeScriptAccumulator) {
                        #(
                            builder.add::<#variant_struct_names>();
                        )*
                    }
            
                    fn validate(value: &openai_magic_instantiate::export::JsonValue) -> Result<Self, String> {
                        let kind = value.get("kind").ok_or("Expected field 'kind'")?;
                        let kind = kind.as_str().ok_or_else(|| format!("Expected 'kind' to be a string, got {}", openai_magic_instantiate::JsonValueExt::type_str(value)))?;
                        let result = match kind {
                            #(
                                stringify!(#variant_struct_kinds) => {
                                    let value = <#variant_struct_names>::validate(value)?;
                                    #variant_struct_to_variants
                                },
                            )*
                            _ => Err(format!("Unknown variant {}", kind)),
                        }?;
                        #(
                            #definition_validators
                        )*
                        Ok(result)
                    }
            
                    fn default_if_omitted() -> Option<Self> {
                        None
                    }
                }
            }
        },
        Data::Union(_) => todo!(),
    }.into()
}

#[proc_macro]
pub fn implement_integers(_input: TokenStream) -> TokenStream {
    let type_tokens = vec![
        quote! { u8 },
        quote! { u16 },
        quote! { u32 },
        quote! { u64 },
        quote! { usize },
        quote! { i8 },
        quote! { i16 },
        quote! { i32 },
        quote! { i64 },
        quote! { isize },
    ];

    let names = vec![
        "U8",
        "U16",
        "U32",
        "U64",
        "USize",
        "I8",
        "I16",
        "I32",
        "I64",
        "ISize",
    ];

    quote! {
        #(
            impl MagicInstantiate for #type_tokens {
                fn name() -> String {
                    #names.to_string()
                }

                fn reference() -> String {
                    #names.to_string()
                }

                fn definition() -> String {
                    let min = Self::MIN;
                    let max = Self::MAX;
                    let name = #names;
                    format!("
// Integer in [{min}, {max}]
type {name} = number;
                    ").trim().to_string()
                }

                fn add_dependencies(builder: &mut TypeScriptAccumulator) {}

                fn validate(value: &JsonValue) -> Result<Self, String> {
                    match value {
                        JsonValue::Number(number) => {
                            match number.as_i64() {
                                Some(number) => {
                                    if number >= (Self::MIN as i64) && number < (Self::MAX as i64) {
                                        Ok(number as _)
                                    } else {
                                        Err(format!("Expected integer in [{}, {}], got {}", Self::MIN, Self::MAX, number))
                                    }
                                }
                                None => Err(format!("Expected integer in [{}, {}], got {}", Self::MIN, Self::MAX, number)),
                            }
                        }
                        _ => Err(format!("Expected integer, got {}", value.type_str())),
                    }
                }

                fn default_if_omitted() -> Option<Self> {
                    None
                }
            }
        )*
    }.into()
}

#[proc_macro]
pub fn implement_tuples(_input: TokenStream) -> TokenStream {
    let mut results = vec![];

    for i in 2..16usize {

        let generic_names = (1..=i).map(|i| syn::Ident::new(&format!("T{}", i), proc_macro2::Span::call_site())).collect::<Vec<_>>();
        let indexes = (0..i).collect::<Vec<_>>();

        results.push(quote! {

            impl<#(#generic_names: MagicInstantiate),*> MagicInstantiate for (#(#generic_names,)*) {
                fn name() -> String {
                    let names = vec![#(<#generic_names>::name()),*];
                    format!("Tuple{}", names.join(""))
                }

                fn reference() -> String {
                    let references = vec![#(<#generic_names>::reference()),*];
                    format!("[{}]", references.join(", "))
                }

                fn definition () -> String { "".to_string() }

                fn add_dependencies(builder: &mut TypeScriptAccumulator) {
                    #(
                    builder.add::<#generic_names>();
                    )*
                }

                fn validate(value: &JsonValue) -> Result<Self, String> {
                    let JsonValue::Array(value) = value else { return Err(format!("Expected array tuple, got {}", value.type_str())) };
                    if value.len() != #i {
                        return Err(format!("Expected {} elements but got {}", #i, value.len()));
                    }
                    Ok((#(<#generic_names>::validate(&value[#indexes])?,)*))
                }

                fn default_if_omitted() -> Option<Self> {
                    None
                }
            }
        });
    }

    quote! {
        #( #results )*
    }.into()
}