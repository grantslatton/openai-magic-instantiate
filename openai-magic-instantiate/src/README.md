# Magic Instantiate

## Quickstart

```rust
use openai_magic_instantiate::*;

#[derive(MagicInstantiate)]
struct Person {
    // Descriptions can help the LLM understand how to generate the value
    #[magic(description = "The person's name without any titles or honorifics")]
    name: String,
    // Validators can be used to enforce constraints on the generated value
    #[magic(validator = Min(1800))]
    #[magic(validator = Max(2100))]
    year_of_birth: u32,
}

let person = Person::instantiate("The president of the USA in 1954").await?;
```

Descriptions and validators can be applied at the field level, or the struct/enum level.

Some basic validators are provided, but you can also define your own by implementing the `Validator` trait.

What happens here is the derived `MagicInstantiate` trait allows this struct to be represented as a TypeScript type definition.

This type definition plus a few instructions are used as a prompt to the LLM. The output of the LLM is validated and marshalled back into the Rust type. Attempts are made to re-prompt the LLM to fix any validation errors.

With this simple mechanism, you can write entire programs infused with AI.