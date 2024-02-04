//! This crate provides a way to instantiate well-typed and validated values using the OpenAI API.
//! 
//! The main trait is [`MagicInstantiate`] which can be derived automatically using the `#[derive(MagicInstantiate)]` macro.
//! 
//! For a type that implements [`MagicInstantiate`], you can call [`instantiate`](MagicInstantiate::instantiate) to get a value of that type:
//! 
//! ```
//! use openai_magic_instantiate::*;
//! 
//! #[derive(Debug, MagicInstantiate)]
//! // You can add descriptions to the struct and its fields that will appear
//! // as comments in the generated TypeScript
//! #[magic(description = "A simple container for a name and date of birth")]
//! // You can also add validators to the struct and its fields
//! // The validator's instructions will also appear as comments in the
//! // generated TypeScript, the validation will be called during instantiation
//! struct Person {
//!     #[magic(description = "<given name> <surname> (no middle)")]
//!     name: String,
//! 
//!     // Multiple validators can be used.
//!     // There are types provided for common assertions like Min and Max
//!     #[magic(validator = YearValidator)]
//!     #[magic(validator = Min(1900))]
//!     #[magic(validator = Max(2100))]
//!     year_of_birth: u32,
//! }
//! 
//! struct YearValidator;
//! 
//! impl Validator<u32> for YearValidator {
//!     fn instructions(&self) -> String {
//!         "Value must be a 4-digit year, do not use 2-digit abbreviations".into()
//!     }
//! 
//!     fn validate(&self, value: &u32) -> Result<(), String> {
//!         if *value < 1000 || *value > 9999 { 
//!             Err(format!("{} is not a 4-digit year", value))
//!         } else {
//!             Ok(())
//!         }
//!     }
//! }
//! 
//! async fn example() {
//!     let eisenhower = Person::instantiate("President of the United States in 1954").await.unwrap();
//!     assert_eq!(eisenhower.name, "Dwight Eisenhower");
//! }
//! ```
//! 
//! Internally, the library will compile these type definitions and instructions into a snippet of TypeScript
//! code. Then, it will prompt the model to generate a JSON value of the specified type. The JSON value is then validated
//! and converted to the Rust type.
//! 
//! For the example above, the following prompt is generated:
//! 
//! ```text
//! // Integer in [0, 4294967295]
//! type U32 = number;
//! 
//! // A simple container for a name and date of birth
//! type Person = {
//!     // <given name> <surname> (no middle)
//!     name: string;
//!     // Value must be a 4-digit year, do not use 2-digit abbreviations
//!     // Value must be greater than or equal to 1900
//!     // Value must be less than or equal to 2100
//!     yearOfBirth: U32;
//! };
//! 
//! User request:
//! President of the United States in 1954
//! 
//! Give the result as a JSON value of type Person.
//! Use minified JSON on a single line.
//! Use the exact type specified.
//! ```
//! 
use std::any::{Any, TypeId};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::future::Future;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex, OnceLock};

use async_openai::config::OpenAIConfig;
use async_openai::error::OpenAIError;
use async_openai::types::{ChatCompletionRequestAssistantMessage, ChatCompletionRequestMessage, ChatCompletionRequestUserMessage, ChatCompletionResponseFormat, ChatCompletionResponseFormatType, CreateChatCompletionRequestArgs, Role};

use async_openai::Client as OpenAIClient;
use openai_magic_instantiate_derive::*;
use serde_json::Value as JsonValue;

extern crate self as openai_magic_instantiate;

#[doc(hidden)]
pub mod export;

pub use openai_magic_instantiate_derive::MagicInstantiate;

/// Accumulates TypeScript definitions needed to define a type.
/// You probably do not need to use interact with this type unless you are implementing [`MagicInstantiate`] manually.
///
/// A type might have fields that are other types that need their
/// own definitions. This struct accumulates those definitions.
/// 
/// The [`add`](TypeScriptAccumulator::add) method is used to add a type to the accumulator.
#[derive(Debug, Default)]
pub struct TypeScriptAccumulator {
    definitions: String,
    definition_by_name: HashMap<String, String>,
    visited: HashSet<TypeId>,
}

impl TypeScriptAccumulator {
    /// Add a type to the accumulator.
    /// 
    /// This is a no-op if the type has already been added.
    /// 
    /// The accumulator will internally call [`add_dependencies`](MagicInstantiate::add_dependencies) to add any
    /// dependencies of the type. 
    pub fn add<T: MagicInstantiate>(&mut self) {
        if !self.visited.insert(TypeId::of::<T>()) {
            return;
        }

        let definition = T::definition();

        if let Some(old) = self.definition_by_name.insert(T::name(), definition.clone()) {
            assert_eq!(old, definition, "Type {} has conflicting definitions", T::name());
            return;
        }

        T::add_dependencies(self);

        if !definition.is_empty() {
            self.definitions.push_str(&format!("\n\n{definition}"));
        }

    }
}

/// A `Validator` is a type that can validate a value of type `T`.
pub trait Validator<T> {
    /// A human-readable string that describes the validation to be performed.
    /// 
    /// Example: "Value must be a positive integer"
    fn instructions(&self) -> String;

    /// Validate a value of type `T`.
    /// 
    /// If possible, the error should indicate how the caller might fix the problem.
    /// 
    /// Example: if the value must be a prime number, the error message might indicate
    /// the nearest prime number to the value that was given.
    fn validate(&self, value: &T) -> Result<(), String>;
}

#[doc(hidden)]
pub trait JsonValueExt {
    fn type_str(&self) -> &'static str;
}

#[doc(hidden)]
impl JsonValueExt for JsonValue {
    fn type_str(&self) -> &'static str {
        match self {
            JsonValue::Null => "null",
            JsonValue::Bool(_) => "boolean",
            JsonValue::Number(_) => "number",
            JsonValue::String(_) => "string",
            JsonValue::Array(_) => "array",
            JsonValue::Object(_) => "object",
        }
    }
}

/// `()`` maps to the `null` constant in TypeScript.
impl MagicInstantiate for () {
    fn name() -> String {
        "Null".to_string()
    }

    fn reference() -> String {
        "null".to_string()
    }

    fn add_dependencies(_builder: &mut TypeScriptAccumulator) {}
    fn definition() -> String { "".into() }

    fn validate(value: &JsonValue) -> Result<Self, String> {
        match value {
            JsonValue::Null => Ok(()),
            _ => Err(format!("Expected null, got {}", value.type_str())),
        }
    }

    fn default_if_omitted() -> Option<Self> {
        Some(())
    }
}

/// `Option<T>` maps to `T | null` in TypeScript.
impl<T: MagicInstantiate> MagicInstantiate for Option<T> {
    fn name() -> String {
        format!("Optional{}", T::name())
    }

    fn reference() -> String {
        format!("{} | null", T::reference())
    }

    fn definition() -> String { "".into() }

    fn add_dependencies(builder: &mut TypeScriptAccumulator) {
        builder.add::<T>();
    }

    fn validate(value: &JsonValue) -> Result<Self, String> {
        match value {
            JsonValue::Null => Ok(None),
            _ => {
                let value = T::validate(value)?;
                Ok(Some(value))
            }
        }
    }

    fn default_if_omitted() -> Option<Self> {
        Some(None)
    }
}

/// `Vec<T>` maps to `Array<T>` in TypeScript.
impl<T: MagicInstantiate> MagicInstantiate for Vec<T> {
    fn name() -> String {
        format!("{}Array", T::name())
    }

    fn reference() -> String {
        format!("Array<{}>", T::reference())
    }

    fn definition() -> String { "".into() }

    fn add_dependencies(builder: &mut TypeScriptAccumulator) {
        builder.add::<T>();
    }

    fn validate(value: &JsonValue) -> Result<Self, String> {
        match value {
            JsonValue::Array(values) => {
                let mut result = Vec::new();
                for value in values {
                    let value = T::validate(value)?;
                    result.push(value);
                }
                Ok(result)
            }
            _ => Err(format!("Expected array, got {}", value.type_str())),
        }
    }

    fn default_if_omitted() -> Option<Self> {
        None
    }
}

/// `Box<T>` maps to `T` in TypeScript.
impl<T: MagicInstantiate> MagicInstantiate for Box<T> {
    fn name() -> String {
        T::name()
    }

    fn reference() -> String {
        T::reference()
    }

    fn definition() -> String {
        T::definition()
    }

    fn add_dependencies(builder: &mut TypeScriptAccumulator) {
        builder.add::<T>();
    }

    fn validate(value: &JsonValue) -> Result<Self, String> {
        let value = T::validate(value)?;
        Ok(Box::new(value))
    }

    fn default_if_omitted() -> Option<Self> {
        Some(Box::new(T::default_if_omitted()?))
    }
}

/// `String` maps to `string` in TypeScript.
impl MagicInstantiate for String {
    fn name() -> String {
        "String".to_string()
    }

    fn reference() -> String {
        "string".to_string()
    }

    fn definition() -> String { "".into() }

    fn add_dependencies(_builder: &mut TypeScriptAccumulator) {}

    fn validate(value: &JsonValue) -> Result<Self, String> {
        match value {
            JsonValue::String(string) => Ok(string.clone()),
            _ => Err(format!("Expected string, got {}", value.type_str())),
        }
    }

    fn default_if_omitted() -> Option<Self> {
        None
    }
}

/// `bool` maps to `boolean` in TypeScript.
impl MagicInstantiate for bool {
    fn name() -> String {
        "Boolean".to_string()
    }

    fn reference() -> String {
        "boolean".to_string()
    }

    fn definition() -> String { "".into() }

    fn add_dependencies(_builder: &mut TypeScriptAccumulator) {}

    fn validate(value: &JsonValue) -> Result<Self, String> {
        match value {
            JsonValue::Bool(boolean) => Ok(*boolean),
            _ => Err(format!("Expected boolean, got {}", value.type_str())),
        }
    }

    fn default_if_omitted() -> Option<Self> {
        None
    }
}

macro_rules! impl_for_float {
    ($($t:ty)*) => {
        $(
            impl MagicInstantiate for $t {
                fn name() -> String {
                    stringify!($t).to_ascii_uppercase()
                }

                fn reference() -> String {
                    "number".to_string()
                }

                fn definition() -> String { "".into() }

                fn add_dependencies(_builder: &mut TypeScriptAccumulator) {} 

                fn validate(value: &JsonValue) -> Result<Self, String> {
                    match value {
                        JsonValue::Number(number) => {
                            number.as_f64().ok_or_else(|| "Expected number".to_string()).map(|n| n as $t)
                        }
                        _ => Err(format!("Expected number, got {}", value.type_str())),
                    }
                }

                fn default_if_omitted() -> Option<Self> {
                    None
                }
            }
        )*
    };
}

impl_for_float!(f32 f64);

implement_integers!();
implement_tuples!();

// 1-tuple is handled separately
impl<T: MagicInstantiate> MagicInstantiate for (T,) {
    fn name() -> String {
        T::name()
    }

    fn reference() -> String {
        T::reference()
    }

    fn definition() -> String {
        T::definition()
    }

    fn add_dependencies(builder: &mut TypeScriptAccumulator) {
        builder.add::<T>();
    }

    fn validate(value: &JsonValue) -> Result<Self, String> {
        T::validate(value).map(|t| (t,))
    }

    fn default_if_omitted() -> Option<Self> {
        T::default_if_omitted().map(|t| (t,))
    }
}

#[derive(Debug)]
pub enum InstantiateError {
    /// The model was unable to instantiate a validated value. The messages contain the chat history for debugging purposes.
    MaxCorrectionsExceeded {
        messages: Vec<ChatCompletionRequestMessage>,
    },
    /// An error occurred while communicating with the OpenAI API.
    OpenAIError{
        error: OpenAIError,
    },
}

/// A wrapper around all the arguments needed to instantiate a value.
#[derive(Debug, Clone)]
pub struct MagicInstantiateParameters {
    /// The name of the models to use, in order.
    /// 
    /// This is a `Vec` because the user might want to start with a weaker model and upgrade to a stronger one after a few validation failures.
    /// If you just want to use a single model for all requests, just provide `vec![model; count]`.
    pub models: Vec<String>,
    /// Corresponds to the OpenAI `temperature` parameter.
    pub temperature: f32,
    /// The OpenAI client to use.
    pub client: Arc<OpenAIClient<OpenAIConfig>>,
}

/// By default, the parameters are set to use the `gpt-3.5-turbo-1106` model for 3 validation 
/// attempts, with a temperature of 0.0, and a fresh OpenAI client with default settings.
impl Default for MagicInstantiateParameters {
    fn default() -> Self {
        MagicInstantiateParameters {
            models: vec!["gpt-3.5-turbo-1106".to_string(); 3],
            temperature: 0.0,
            client: Arc::new(OpenAIClient::new()),
        }
    }
}

static GLOBAL_PARAMETERS: OnceLock<Mutex<MagicInstantiateParameters>> = OnceLock::new();

/// If you want to set some global parameters for all instantiations, you can use this function to mutate them.
pub fn mutate_global_parameters(f: impl FnOnce(&mut MagicInstantiateParameters)) {
    let lock = GLOBAL_PARAMETERS.get_or_init(|| Default::default());
    let mut lock = lock.lock().unwrap();
    f(&mut *lock);
}

/// Get a copy of the current global parameters used for [`instantiate`](MagicInstantiate::instantiate).
pub fn get_global_parameters() -> MagicInstantiateParameters {
    let lock = GLOBAL_PARAMETERS.get_or_init(|| Default::default());
    let lock = lock.lock().unwrap();
    lock.clone()
}

/// The `MagicInstantiate` trait is the main trait of this library, but you should not need to implement it manually.
/// Prefer using the `#[derive(MagicInstantiate)]` macro instead.
/// 
/// If you must implement it manually, here is an example:
/// 
/// ```
/// use openai_magic_instantiate::*;
/// use openai_magic_instantiate::export::JsonValue;
/// 
/// struct Person {
///     name: String,
///     year_of_birth: u32,
/// }
/// 
/// impl MagicInstantiate for Person {
///     fn definition() -> String {
///         let string_ref = <String>::reference();
///         let u32_ref = <u32>::reference();
///         // Must use {{ to escape the curly braces
///         format!("
/// type Person = {{
///     // <given name> <surname> (no middle)
///     name: {string_ref};
///     // Value must be a 4-digit year, do not use 2-digit abbreviations
///     yearOfBirth: {u32_ref};
/// }};
///         ").trim().to_string()
///     }
/// 
///     fn name() -> String {
///         "Person".to_string()
///     }
/// 
///     fn reference() -> String {
///         // Since we defined a type, `reference` is the same as `name`
///         // This will be the case for all structs and enums
///         Self::name()
///     }
/// 
///     fn add_dependencies(builder: &mut TypeScriptAccumulator) {
///         // We need to add all the types that are fields of this type
///         builder.add::<String>();
///         builder.add::<u32>();
///     }
/// 
///     fn validate(value: &JsonValue) -> Result<Self, String> {
///         let JsonValue::Object(value) = value else {
///             return Err("Expected object with fields [\"name\", \"yearOfBirth\"]".to_string());
///         };
///         let result = Self {
///             name: {
///                 let value = value.get("name").ok_or("Expected field name, but it wasn't present")?;
///                 <String>::validate(value)?
///             },
///             year_of_birth: {
///                 let value = value.get("yearOfBirth").ok_or("Expected field yearOfBirth, but it wasn't present")?;
///                 let value = <u32>::validate(value)?;
///                 if value < 1000 || value > 9999 {
///                     return Err(format!("{} is not a 4-digit year", value));
///                 }
///                 value
///             },
///         };
///         Ok(result)
///     }
/// 
///     fn default_if_omitted() -> Option<Self> {
///         None
///     }
/// }
/// ```
///       
pub trait MagicInstantiate: Any + Sized {
    /// The TypeScript definition of the type if it needs one.
    /// 
    /// For example:
    /// 
    /// ```typescript
    /// type Person = {
    ///     name: string;
    ///     // Value must be a 4-digit year, do not use 2-digit abbreviations
    ///     age: number;
    /// };
    /// ```
    /// 
    /// Note that the definition should include helpful comments.
    /// 
    /// Note that the definition might be empty. For example, `f64`'s implementation of this method is empty, because
    /// the [`reference`](MagicInstantiate::reference) is just the TypeScript primitive `number`.
    fn definition() -> String;

    /// How the type should be referred to in other types.
    /// 
    /// If the type returns a [`definition`](MagicInstantiate::definition), this should be the same as the type name in the definition.
    /// 
    /// If the type does not return a definition, this should be some type that is valid in vanilla TypeScript.
    /// For example, `Array<[string, number]>`, `"myEnumKind"`, and `string | null` are all valid references.
    fn reference() -> String;

    /// Name of the type if it needed to be referred to in another type name.
    /// 
    /// Even types that don't need a definition should have a name.
    /// For example, the Rust type `Option<String>` doesn't need a definition and just uses
    /// a [reference](MagicInstantiate::reference) of `string | null`, but this method returns 
    /// the *name* `OptionalString`.
    /// 
    /// This is needed because another type's name might depend on this type's name. For example, `MyCollection<Option<String>>`
    /// might be named `MyCollectionOfOptionalString`.
    fn name() -> String;

    /// If this type is non-primitive, it contains some other types as fields.
    /// This method should call `builder.add::<T>()` for each field type `T`.
    fn add_dependencies(builder: &mut TypeScriptAccumulator);

    /// Validate a JSON value and convert it to the type, return a helpful error message if the value is invalid.
    fn validate(value: &JsonValue) -> Result<Self, String>;

    /// If the value is omitted, what the default is.
    /// Most objects should return `None` because they are not optional.
    /// `Option<T>` returns `Some(None)`, that is, omitting a field with type `Option<T>`` is the same as including the field with value `None`
    /// Another case you might want to return `Some` is for a collection type where omission defaults to the empty collection.
    fn default_if_omitted() -> Option<Self>;

    /// Convert this type into a prompt for the LLM.
    /// 
    /// The default implementation is pretty good and you probably don't need to override it.
    /// 
    /// This method is exposed mostly for debugging purposes.
    fn prompt_for(instructions: &str) -> String {
        let mut builder = TypeScriptAccumulator::default();
        builder.add::<Self>();
        let name = Self::reference();
        let definitions = builder.definitions;

        format!("\
{}

User request:
{}

Give the result as a JSON value of type {}.
Use the exact type specified.",
            definitions, instructions, name
        )
    }

    /// Instantiate a value using the global parameters.
    /// 
    /// Use [`mutate_global_parameters`] to set the global parameters.
    /// 
    /// See the [`instantiate_parameterized`](MagicInstantiate::instantiate_parameterized) method for more information.
    fn instantiate(instructions: impl AsRef<str>) -> impl Future<Output = Result<Self, InstantiateError>> {
        async move {
            let parameters = get_global_parameters();
            Self::instantiate_parameterized(parameters, instructions.as_ref()).await
        }
    }

    /// Instantiate a value using custom parameters.
    /// 
    /// `instructions` is a string that describes how the user wants the value to be instantiated.
    fn instantiate_parameterized(parameters: MagicInstantiateParameters, instructions: &str) -> impl Future<Output = Result<Self, InstantiateError>> {
        async move {
            let prompt = Self::prompt_for(instructions);

            eprintln!("Prompt:\n\n{}", prompt);

            let mut messages: Vec<ChatCompletionRequestMessage> = vec![];
            messages.push(ChatCompletionRequestUserMessage {
                role: Role::User,
                content: prompt.clone().into(),
                ..Default::default()
            }.into());

            for model in parameters.models {
                let request = CreateChatCompletionRequestArgs::default()
                    .model(model)
                    .messages(messages.clone())
                    .temperature(parameters.temperature)
                    .response_format(ChatCompletionResponseFormat { r#type: ChatCompletionResponseFormatType::JsonObject })
                    .build()
                    .unwrap();

                let response = parameters
                    .client
                    .chat()
                    .create(request)
                    .await
                    .map_err(|error| InstantiateError::OpenAIError { error })?;

                let choice = response
                    .choices
                    .into_iter()
                    .next()
                    .unwrap();

                let content = choice.message.content.unwrap();

                eprintln!("Response: {}", content);

                messages.push(ChatCompletionRequestAssistantMessage {
                    content: Some(content.clone()),
                    role: Role::Assistant,
                    ..Default::default()
                }.into());

                match serde_json::from_str::<JsonValue>(&content) {
                    Err(e) => {

                        let line = content.lines().nth(e.line() - 1).unwrap();

                        messages.push(ChatCompletionRequestUserMessage {
                            role: Role::User,
                            content: format!("JSON syntax error. Make sure to use valid JSON literal syntax. Do not use any javascript functions or arithmetic in the JSON definition.\n\nFailure line:\n{line}\n\nCorrect the error in the next attempt.").into(),
                            ..Default::default()
                        }.into());
                    }
                    Ok(value) => {

                        let validated = Self::validate(&value);

                        match validated {
                            Err(e) => {
                                eprintln!("Validation error: {}", e);
                                messages.push(ChatCompletionRequestUserMessage {
                                    role: Role::User,
                                    content: format!("JSON object validation error:\n{e}\n\nCorrect the error in the next attempt.").into(),
                                    ..Default::default()
                                }.into());
                            },
                            Ok(output) => {
                                return Ok(output);
                            }
                        }
                    }
                }
            }

            Err(InstantiateError::MaxCorrectionsExceeded { messages })
        }
    }
}

#[macro_export]
macro_rules! magic {
    ($fmt_str:literal $(, $args:expr)*) => {
        {
            use $crate::MagicInstantiate;
            $crate::MagicInstantiate::instantiate(format!($fmt_str, $($args)*)).await
        }
    };
}

/// A [`Validator`] that creates a minimum valid value constraint (i.e. a floor).
pub struct Min<T>(pub T);

impl<T: Ord + fmt::Display> Validator<T> for Min<T> {
    fn instructions(&self) -> String {
        format!("Value must be greater than or equal to {}", self.0)
    }

    fn validate(&self, value: &T) -> Result<(), String> {
        if *value < self.0 {
            Err(format!("{} is less than {}", value, self.0))
        } else {
            Ok(())
        }
    }
}

/// A [`Validator`] that creates a maximum valid value constraint (i.e. a ceiling).
pub struct Max<T>(pub T);

impl<T: Ord + fmt::Display> Validator<T> for Max<T> {
    fn instructions(&self) -> String {
        format!("Value must be less than or equal to {}", self.0)
    }

    fn validate(&self, value: &T) -> Result<(), String> {
        if *value > self.0 {
            Err(format!("{} is greater than {}", value, self.0))
        } else {
            Ok(())
        }
    }
}

/// A [`Validator`] that creates a minimum length constraint for strings and lists.
/// 
/// Use with [`IndexedArray`] for best results.
pub struct MinLength(pub usize);

impl Validator<String> for MinLength {
    fn instructions(&self) -> String {
        format!("String must have at least {} characters", self.0)
    }

    fn validate(&self, value: &String) -> Result<(), String> {
        let len = value.len();
        if len < self.0 {
            Err(format!("String has only {} characters, expected at least {}", len, self.0))
        } else {
            Ok(())
        }
    }
}

impl<T> Validator<Vec<T>> for MinLength {
    fn instructions(&self) -> String {
        format!("Array must have at least {} elements", self.0)
    }

    fn validate(&self, value: &Vec<T>) -> Result<(), String> {
        let len = value.len();
        if len < self.0 {
            Err(format!("Array has only {} elements, expected at least {}", len, self.0))
        } else {
            Ok(())
        }
    }
}

/// A [`Validator`] that creates a maximum length constraint for strings and lists.
/// 
/// Use with [`IndexedArray`] for best results.
pub struct MaxLength(pub usize);

impl Validator<String> for MaxLength {
    fn instructions(&self) -> String {
        format!("String must have at most {} characters", self.0)
    }

    fn validate(&self, value: &String) -> Result<(), String> {
        let len = value.len();
        if len > self.0 {
            Err(format!("String has {} characters, expected at most {}", len, self.0))
        } else {
            Ok(())
        }
    }
}

impl<T> Validator<Vec<T>> for MaxLength {
    fn instructions(&self) -> String {
        format!("Array must have at most {} elements", self.0)
    }

    fn validate(&self, value: &Vec<T>) -> Result<(), String> {
        let len = value.len();
        if len > self.0 {
            Err(format!("Array has {} elements, expected at most {}", len, self.0))
        } else {
            Ok(())
        }
    }
}

/// A [`Validator`] that creates a constraint that the string must be exactly a certain length.
/// 
/// Use with [`IndexedArray`] for best results.
pub struct ExactLength(pub usize);

impl Validator<String> for ExactLength {
    fn instructions(&self) -> String {
        format!("String must have exactly {} characters", self.0)
    }

    fn validate(&self, value: &String) -> Result<(), String> {
        let len = value.len();
        if len != self.0 {
            Err(format!("String has {} characters, expected exactly {}", len, self.0))
        } else {
            Ok(())
        }
    }
}

impl<T> Validator<Vec<T>> for ExactLength {
    fn instructions(&self) -> String {
        format!("Array must have exactly {} elements", self.0)
    }

    fn validate(&self, value: &Vec<T>) -> Result<(), String> {
        let len = value.len();
        if len != self.0 {
            Err(format!("Array has {} elements, expected exactly {}", len, self.0))
        } else {
            Ok(())
        }
    }
}

/// `IndexedArray` is a wrapper around a `Vec` that should be used when you want to enforce a particular length.
/// 
/// If you want a list of a particular length, `IndexedArray` will validate as that length
/// much more often than a plain `Vec`.
/// 
/// This is because `IndexedArray` makes the LLM give each element a number, forcing it to count the elements as it generates them.
#[derive(Debug)]
pub struct IndexedArray<T>(pub Vec<T>);

impl<T> Deref for IndexedArray<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for IndexedArray<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: MagicInstantiate> MagicInstantiate for IndexedArray<T> {
    fn name() -> String {
        format!("IndexedArray{}", T::name())
    }

    fn reference() -> String {
        Self::name()
    }

    fn definition() -> String {
        let u32_ref = u32::reference();
        let t_ref = T::reference();
        format!("// Array of [array index, element] tuples\ntype {} = Array<[{}, {}]>", Self::name(), u32_ref, t_ref)
    }

    fn add_dependencies(builder: &mut TypeScriptAccumulator) {
        builder.add::<u32>();
        builder.add::<T>();
    }

    fn validate(value: &JsonValue) -> Result<Self, String> {
        match value {
            JsonValue::Array(values) => {
                let mut items = Vec::new();
                let mut expected_next_index = 0;
                for tuple in values {
                    let (index, item) = match tuple {
                        JsonValue::Array(tuple) => {
                            if tuple.len() != 2 {
                                return Err(format!("Expected [index, item] tuple, got array of length {}", tuple.len()));
                            }
                            let index = tuple[0].as_u64().ok_or_else(|| "Expected index to be an integer")? as u32;
                            let item = T::validate(&tuple[1]).map_err(|e| format!("Error in item at index {}: {}", index, e))?;
                            (index, item)
                        }
                        _ => return Err(format!("Expected [index, item] array tuple, got {}", tuple.type_str())),
                    };
                    if index != expected_next_index {
                        return Err(format!("Expected {} index {} but got {}", Self::name(), expected_next_index, index));
                    }
                    items.push(item);
                    expected_next_index += 1;
                }
                Ok(IndexedArray(items))
            }
            _ => Err(format!("Expected array, got {}", value.type_str())),
        }
    }

    fn default_if_omitted() -> Option<Self> {
        None
    }
}
