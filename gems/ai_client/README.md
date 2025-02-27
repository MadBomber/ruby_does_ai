# AiClient

First and foremost a big **THANK YOU** to [Kevin Sylvestre](https://ksylvest.com/) for his gem [OmniAI](https://github.com/ksylvest/omniai) and [Olympia](https://olympia.chat/) for their [open_router gem](https://github.com/OlympiaAI/open_router) upon which this effort depends.

Version 0.4.0 has two changes which may break your existing application.
1. The default configuration no longer has a Logger instance.  You will need to add your own instance to either the class or instance configuration using `AiClient.class_config.logger = YourLogger` and/or `client.config.logger = YourLogger`
2. The `chat` method now keeps a context window.  The window length is defined by the configuration item `context_length`  If you do not want to maintain a context window, set the `context_length` configuration item to either nil or zero.

See the  [change log](CHANGELOG.md) for recent modifications.

You should also checkout the [raix gem](https://github.com/OlympiaAI/raix).  I like the way that Obie's API is setup for callback functions.  [raix-rails](https://github.com/OlympiaAI/raix-rails) is also available.


<!-- Tocer[start]: Auto-generated, don't remove. -->

## Table of Contents

  - [Summary](#summary)
  - [Installation](#installation)
  - [Environment Variables for Provider Access](#environment-variables-for-provider-access)
    - [Changing Envar API Key Names](#changing-envar-api-key-names)
    - [api_key: Parameter](#api_key-parameter)
    - [provider: Parameter](#provider-parameter)
  - [Usage](#usage)
    - [Basic Top-leve Methods](#basic-top-leve-methods)
    - [Utility Methods](#utility-methods)
    - [Configuration](#configuration)
        - [Default Configuration](#default-configuration)
      - [Class Configuration](#class-configuration)
        - [1. Class Configuration Block](#1-class-configuration-block)
        - [2. Set by a Config File](#2-set-by-a-config-file)
        - [3. Supplemented by a Config File](#3-supplemented-by-a-config-file)
      - [Instance Configuration](#instance-configuration)
        - [1. Supplement from a Constructor Block](#1-supplement-from-a-constructor-block)
        - [2. Supplement from a YAML File](#2-supplement-from-a-yaml-file)
        - [3. Load Complete Configuration from a YAML File](#3-load-complete-configuration-from-a-yaml-file)
    - [Top-level Client Methods](#top-level-client-methods)
        - [chat](#chat)
            - [Cpmtext](#cpmtext)
        - [embed](#embed)
        - [speak](#speak)
        - [transcribe](#transcribe)
    - [Options](#options)
        - [Common Options for All Methods](#common-options-for-all-methods)
        - [Chat-specific Options](#chat-specific-options)
        - [Embed-specific Options](#embed-specific-options)
        - [Speak-specific Options](#speak-specific-options)
        - [Transcribe-specific Options](#transcribe-specific-options)
    - [Advanced Prompts](#advanced-prompts)
    - [Callback Functions (aka Tools)](#callback-functions-aka-tools)
        - [Defining a Callback Function](#defining-a-callback-function)
    - [OpenRouter Extensions and AiClient::LLM](#openrouter-extensions-and-aiclientllm)
        - [Instance Methods](#instance-methods)
        - [Class Methods](#class-methods)
    - [AiClient::LLM Data Table](#aiclientllm-data-table)
        - [Key Features](#key-features)
        - [Class Methods](#class-methods-1)
        - [Instance Methods](#instance-methods-1)
        - [Usage Example](#usage-example)
        - [Integration with ActiveHash](#integration-with-activehash)
  - [Best ?? Practices](#best--practices)
  - [OmniAI and OpenRouter](#omniai-and-openrouter)
  - [The `chatbot` Method](#the-chatbot-method)
  - [Contributing](#contributing)
  - [License](#license)

<!-- Tocer[finish]: Auto-generated, don't remove. -->


## Summary

AiClient is a Ruby gem that simplifies integration with AI services by providing:

- Unified interface across multiple LLM providers
- Automatic provider detection from model names
- Context management for chat conversations
- Function calling (tools) support
- Middleware for logging, retries, and customization
- Batch processing for embeddings
- Text-to-speech and transcription
- Interactive chatbot mode

The gem supports leading providers including OpenAI, Anthropic, Google, Mistral, LocalAI, and Ollama. Whether you're building a chatbot, implementing RAG, or need speech services, AiClient handles the complexity so you can focus on your application logic.

## Installation

If you are using a Gemfile and bundler in your project just install the gem by executing:

```bash
bundle add ai_client
```

If bundler is not being used to manage dependencies, install the gem by executing:

```bash
gem install ai_client
```

## Environment Variables for Provider Access

For fee providers require an account and provide an access token to allow the use of their LLM models.  The value of these access tokens is typically saved in system environment variables or some other secure data store.  AiClient has a default set of system environment variable names for these access tokens based upon the pattern of `provider_api_key` which can be over-ridden.

| Symbol | Envar API Key | Client Source |
| --- | --- | --- |
| :anthropic | [ANTHROPIC_API_KEY](https://www.anthropic.com/) | OmniAI |
| :google | [GOOGLE_API_KEY](https://cloud.google.com/gemini) | OmniAI |
| :localai | [LOCALAI_API_KEY](https://localai.io/) | AiClient Extension |
| :mistral | [MISTRAL_API_KEY](https://mistral.ai/) | OmniAI |
| :ollama | [OLLAMA_API_KEY](https://ollama.com/) | AiClient Extension |
| :open_router | [OPEN_ROUTER_API_KEY](https://openrouter.ai/) | AiClient Extension |
| :openai | [OPENAI_API_KEY](https://www.openai.com/) | OmniAI |


### Changing Envar API Key Names

You can also configure the system environment variable names to match your on standards at the class level.

```ruby
AiClient.class_config.envar_api_key_bames = {
  anthropic:    'your_envar_name',
  google:       'your_envar_name',
  mistral:      'your_envar_name',
  open_router:  'your_envar_name',
  openai:       'your_envar_name'
}

AiClient.class_config.save('path/to/file.yml')
```

### api_key: Parameter

In case you are using a different environment variable for your access token than the ones shown above you can use the `api_key:` parameter.

```ruby
client = AiClient.new('provider/model_name', api_key: ENV['OPENROUTER_API_KEY'])
```

This way if you are using `AiClient` inside of a Rails application you can retrieve your access token from a secretes file.


### provider: Parameter

To explicitly designate a provider to use with an AiClient instance use the parameter `provider: :your_provider` with the Symbol for the supported provider you want to use with the model you specify.  The following providers are supported by the OmniAI gem upon which AiClient depends along with a few extensions.


## Usage

### Basic Top-leve Methods

```ruby
require 'ai_client'

# Simple chat with default model
ai = AiClient.new
response = ai.chat('Hello!')

# Interactive chatbot mode
ai.chatbot('Start conversation')

# Batch embeddings
texts = ['text1', 'text2', 'text3']
embeddings = ai.batch_embed(texts)

# Text-to-speech
audio = ai.speak("Convert this to speech")

# Transcription
text = ai.transcribe("path/to/audio.mp3")

# Function calling
WeatherFunction.register
response = ai.chat("What's the weather?", tools: ['weather'])
```


### Utility Methods

```ruby
require 'ai_client'
ai = AiClient.new # use default model and provider
ai.model    #=> 'gpt-4o' is the default
ai.provider #=> :openai  is the default
#
# To change the class defaults:
#
AiClient.default_provider       = :anthropic
AiClient.default_model[:openai] = 'gpt-4o-mini'
#
# To get an Array of models and providers
#
AiClient.models     # from open_router.ai
AiClient.providers  # from open_router.ai
#
# To get details about a specific provider/model pair:
#
AiClient.model_details('openai/gpt-4o-mini')  # from open_router.ai
```

You can specify which model you want to use and `AiClient` will use the provider associated with that model.


```ruby
ai = AiClient.new('gpt-4o-mini') # sets provider to :openai
#
# If you want to use the open_router.ai service instead of
# going directly to OpenAI do it this way:
#
ai = AiClient.new('openai/gpt-4o-mini') # sets provider to :open_router
```

That's it.  What could be simpler?  If your application is using more than one model, no worries, just create multiple `AiClient` instances.

```ruby
c1 = AiClient.new('nomic-embed-text')
c2 = AiClient.new('gpt-4o-mini')
```

You can also use the `provider:` parameter in the event that the model you want to use is available through multiple providers or that AiClient can not automatically associate the model name with a provider.


```ruby
ai = AiClient.new('nomic-embed-text', provider: :ollama)
```

### Configuration

There are three levels of configuration, each inherenting from the level above. The following sections describe those configuration levels.

##### Default Configuration

The file [lib/ai_client/configuration.rb] hard codes the default configuration.  This is used to update the [lib/ai_client/config.yml] file during development.  If you have some changes for this configuration please send me a pull request so we can all benefit from your efforts.

```ruby
{
                 :logger => nil,
                :timeout => nil,
             :return_raw => false,
         :context_length => 5,
              :providers => {},
    :envar_api_key_names => {
          :anthropic => [
            "ANTHROPIC_API_KEY"
        ],
             :google => [
            "GOOGLE_API_KEY"
        ],
            :mistral => [
            "MISTRAL_API_KEY"
        ],
        :open_router => [
            "OPEN_ROUTER_API_KEY",
            "OPENROUTER_API_KEY"
        ],
             :openai => [
            "OPENAI_API_KEY"
        ]
    },
      :provider_patterns => {
          :anthropic => /^claude/i,
             :openai => /^(gpt|chatgpt|o1|davinci|curie|babbage|ada|whisper|tts|dall-e)/i,
             :google => /^(gemini|gemma|palm)/i,
            :mistral => /^(mistral|codestral|mixtral)/i,
            :localai => /^local-/i,
             :ollama => /(llama|nomic)/i,
        :open_router => /\//
    },
       :default_provider => :openai,
          :default_model => {
          :anthropic => "claude-3-5-sonnet-20240620",
             :openai => "gpt-4o",
             :google => "gemini-pro-1.5",
            :mistral => "mistral-large",
            :localai => "llama3.2",
             :ollama => "llama3.2",
        :open_router => "auto"
    }
}
```

#### Class Configuration

The class configuration is derived initially from the default configuration.  It can be changed in three ways.

##### 1. Class Configuration Block

```ruby
AiClient.configuration do |config|
  config.some_item = some_value
  ...
end
```

##### 2. Set by a Config File

```ruby
AiClient.class_config = AiClient::Config.load('path/to/file.yml')
```

##### 3. Supplemented by a Config File

```ruby
AiClient.class_config.merge! AiClient::Config.load('path/to/file.yml')
```

#### Instance Configuration

All instances have a configuration.  Initially that configuration is the same as the class configuration; however, each instance can have its own separate configuration.  For an instance the class configuration can either be supplemented or complete over-ridden.

##### 1. Supplement from a Constructor Block

```ruby
client = AiClient.new('super-ai-overlord-model') do |config|
  config.some_item = some_value
  ...
end
```

##### 2. Supplement from a YAML File

```ruby
client = AiClient.new('baby-model', config: 'path/to/file.yml')
```

##### 3. Load Complete Configuration from a YAML File

```ruby
client = AiClient.new('your-model')
client.config = AiClient::Config.load('path/to/file.yml')
```

### Top-level Client Methods

See the [examples directory](examples/README.md) for some ideas on how to use AiClient.

The following examples are based upon the same client configuration.

```ruby
AI = AiClient.new(...) do ... end
```

##### chat

Typically `chat(...)` is the most used top-level.  Sometimes refered to as completion.  You are giving a prompt to an LLM and expecting the LLM to respond (ie. complete its transformation).  If you consider the prompt to be a question, the response would be the answer.  If the prompt were a task, the response would be the completion of that task.

```ruby
response = AI.chat(...)
```

The simplest form is a string prompt.  The prompt can come from anywher - a litteral, variable, or get if from a database or a file.

```ruby
response = AI.chat("Is there anything simpler than this?")
```

The response will be a simple string or a response object based upon the setting of your `config.return_raw` item.  If `true` then you get the whole shebang.  If `false` you get just the string.

See the [Advanced Prompts] section to learn how to configure a complex prompt message.

####### Cpmtext

**context_length**

The `context_length` configuration item is used to keep the last "context_length" responses within the chat context window.  If you do not want to keep a context window, you should set the value of `config.context_length = 0`  When you do either at the class or instance level, the chat response will be provided without the LLM knowing any prior context.  If you are implementing a chat-bot, you will want it to have a context of the current conversation.

```ruby
AiClient.config.context_length #=> 5
AiClient.config.context_length = 0  # Turns off the context window
```

##### embed

Embeddings (as in 'embed additional information') is how retrial augmented generation (RAG) works - which is a deeper subject for another place.  Basically when using an LLM that supports the vectorization of stuff to create embeddings you can use `embed(stuff)` to return the vector associated with the stuff you gave the model.  This vector (an Array of Floating Points Numbers) is a mathematical representation of the stuff that can be used to compare, mathematically, one piece of stuff to a collection of stuff to find other stuff in that collection that closely resembles the stuff for which you are looking.  Q: What is stuff?  A: You know; its just stuff.

```ruby
AI.embed(...)
response = AI.batch_embed(...)
```

Recommendation: Use PostgreSQL, pg_vector and the neighbor gem.

##### speak

```ruby
response = AI.speak("Isn't it nice to have a computer that will talk to you?")
```

The response will contain audio data that can be played, manipulated or saved to a file.

##### transcribe

```ruby
response = AI.transcribe(...)
```


### Options

The four major methods (chat, embed, speak, and transcribe) support various options that can be passed to the underlying client code. Here's a breakdown of the common options for each method:

##### Common Options for All Methods

- `provider:` - Specifies the AI provider to use (e.g., `:openai`, `:anthropic`, `:google`, `:mistral`, `:ollama`, `:localai`).
- `model:` - Specifies the model to use within the chosen provider.
- `api_key:` - Allows passing a specific API key, overriding the default environment variable.
- `temperature:` - Controls the randomness of the output (typically a float between 0 and 1).
- `max_tokens:` - Limits the length of the generated response.

##### Chat-specific Options

- `messages:` - An array of message objects for multi-turn conversations.
- `functions:` - An array of available functions/tools for the model to use.
- `function_call:` - Specifies how the model should use functions ("auto", "none", or a specific function name).
- `stream:` - Boolean to enable streaming responses.

##### Embed-specific Options

- `input:` - The text or array of texts to embed.
- `dimensions:` - The desired dimensionality of the resulting embeddings (if supported by the model).

##### Speak-specific Options

- `voice:` - Specifies the voice to use for text-to-speech (provider-dependent).
- `speed:` - Adjusts the speaking rate (typically a float, where 1.0 is normal speed).
- `format:` - Specifies the audio format of the output (e.g., "mp3", "wav").

##### Transcribe-specific Options

- `file:` - The audio file to transcribe (can be a file path or audio data).
- `language:` - Specifies the language of the audio (if known).
- `prompt:` - Provides context or specific words to aid in transcription accuracy.

Note: The availability and exact names of these options may vary depending on the specific provider and model being used. Always refer to the documentation of the chosen provider for the most up-to-date and accurate information on supported options.

### Advanced Prompts

In more complex application providing a simple string as your prompt is not sufficient.  AiClient can take advantage of OmniAI's complex message builder.

```ruby
client = AiClient.new 'some_model_name'

completion = client.chat do |prompt|
  prompt.system('You are an expert biologist with an expertise in animals.')
  prompt.user do |message|
    message.text 'What species are in the attached photos?'
    message.url('https://.../cat.jpeg', "image/jpeg")
    message.url('https://.../dog.jpeg', "image/jpeg")
    message.file('./hamster.jpeg', "image/jpeg")
  end
end

completion #=> 'The photos are of a cat, a dog, and a hamster.'
```

Of course if `client.config.return_raw` is true, the completion value will be the complete response object.


### Callback Functions (aka Tools)

With the release of version 0.3.0, the way callback functions (also referred to as tools) are defined in the `ai_client` gem has undergone significant changes. This section outlines the new approach in detail.  These changes are designed to create a clearer and more robust interface for developers when working with callback functions. If you encounter any issues while updating your functions, please consult the official documentation or raise an issue in the repository.

##### Defining a Callback Function

To define a callback function, you need to create a subclass of `AiClient::Function`. In this subclass, both the `call` and `details` methods must be implemented.

**Example**

Here's an example illustrating how to define a callback function using the new convention:

```ruby
class WeatherFunction < AiClient::Function
  # The call class method returns a String to be used by the LLM
  def self.call(location:, unit: 'Celsius')
    "#{rand(20..50)}° #{unit} in #{location}"
  end

  # The details method must return a hash with metadata about the function.
  def self.details
    {
      name:         'weather',
      description:  "Lookup the weather in a location",
      parameters:   AiClient::Tool::Parameters.new(
        properties: {
          location: AiClient::Tool::Property.string(description: 'e.g. Toronto'),
          unit:     AiClient::Tool::Property.string(enum: %w[Celsius Fahrenheit]),
        },
        required: [:location]
      )
    }
  end
end

# Register the WeatherFunction for use.
WeatherFunction.register

# Use the *.details[:name] value to reference the tools available for
# the LLM to use in processing the prompt.
response = AI.chat("what is the weather in London", tools: ['weather'])
```

In this example:
- The `call` method is defined to accept named parameters: `location` and `unit`. The default value for `unit` is set to `'Celsius'`.
- The `details` method provides metadata about the function, ensuring that the parameters section clearly indicates which parameters are required.

See the [examples/tools.rb file](examples/tools.rb) for additional examples.

### OpenRouter Extensions and AiClient::LLM

The `open_router.ai` API provides a service that allows you to download a JSON file containing detailed information about all of the providers and their available models. `AiClient` has saved an old copy of this information in the [`models.yml`](lib/ai_client/models.yml) file. If you want to update this file with the latest information from `open_router.ai`, you must have a valid API key.

You can still use the included `models.yml` file with the `AiClient::LLM` class. The following sections describe the convenient instance and class methods that are available. See the section on `AiClient::LLM` for complete details.

##### Instance Methods

- **`model_details`**: Retrieves details for the current model. Returns a hash containing the model's attributes or `nil` if not found.

```ruby
client = AiClient.new('gpt-3.5-turbo')
details = client.model_details
details #=>
{
                    :id => "openai/gpt-3.5-turbo",
                  :name => "OpenAI: GPT-3.5 Turbo",
               :created => 1685232000,
           :description => "GPT-3.5 Turbo is OpenAI's fastest model. It can understand and generate natural language or code, and is optimized for chat and traditional completion tasks.\n\nTraining data up to Sep 2021.",
        :context_length => 16385,
          :architecture => {
             "modality" => "text->text",
            "tokenizer" => "GPT",
        "instruct_type" => nil
    },
               :pricing => {
            "prompt" => "0.0000005",
        "completion" => "0.0000015",
             "image" => "0",
           "request" => "0"
    },
          :top_provider => {
               "context_length" => 16385,
        "max_completion_tokens" => 4096,
                 "is_moderated" => true
    },
    :per_request_limits => {
            "prompt_tokens" => "40395633",
        "completion_tokens" => "13465211"
    }
}
```

- **`models`**: Retrieves model names for the current provider. Returns an array of strings with the names of the models.

```ruby
models = client.models
```

##### Class Methods

- **`providers`**: Retrieves all available providers. Returns an array of unique symbols representing provider names.

```ruby
available_providers = AiClient.providers
```

- **`models(substring = nil)`**: Retrieves model IDs, optionally filtered by a substring.

```ruby
available_models = AiClient.models('turbo')
```

- **`model_details(model_id)`**: Retrieves details for a specific model using its ID. Accepts a string representing the model ID. Returns a hash containing the model's attributes or `nil` if not found.

```ruby
model_info = AiClient.model_details('openai/gpt-3.5-turbo')
```

- **`reset_llm_data`**: Resets the LLM data with the available ORC models. Returns `void`.

```ruby
AiClient.reset_llm_data
```

### AiClient::LLM Data Table

AiClient makes use of the [active_hash](http://github.com/active-hash/active_hash) to provide an ActiveRecord-like capability within its AiClient::LLM class.  This class provides database-like access to the [models.yml](lib/ai_client/models.yml) YAML file.  Using the AiClient::LLM class you have access to details on the models maintained and available through the [open_router.ai](https://open_router.ai) service.  With an API key for OpenRouter you cab update your local copy of the `models.yml` file.


##### Key Features

- **Model and Provider Extraction**:
  - The class provides methods to extract the model name and provider from the LLM's ID.
  - The `model` method returns the model ID derived from the ID.
  - The `provider` method extracts the provider name as a Symbol.

##### Class Methods

- **`reset_llm_data`**:
  - A class-level method that fetches the latest model data from the open_router.ai service and updates the `models.yml` file accordingly.

##### Instance Methods

- **`model`**:
  - Returns the name of the model derived from the LLM's ID.

```ruby
llm_instance = AiClient::LLM.find('openai/gpt-3.5-turbo')
puts llm_instance.model # Output: gpt-3.5-turbo
```

- **`provider`**:
  - Returns the name of the provider associated with the LLM's ID.

```ruby
llm_instance = AiClient::LLM.find('openai/gpt-3.5-turbo')
puts llm_instance.provider # Output: :openai
```

##### Usage Example

The `AiClient::LLM` class is predominantly used to interact with different providers of LLMs. By utilizing the `model` and `provider` methods, users can seamlessly retrieve and utilize models in their applications.

```ruby
llm_instance = AiClient::LLM.find('google/bard')
puts "Model: #{llm_instance.model}, Provider: #{llm_instance.provider}"
```

##### Integration with ActiveHash

The `AiClient::LLM` class inherits from `ActiveHash::Base`, which provides an easy way to define a set of data and allows for lookups and easy manipulation of the data structure. The use of ActiveHash makes it easier to manage the LLM data effectively without needing a full database.


## Best ?? Practices

If you are going to be using one model for multiple purposes in different parts of your application you can assign the instance of `AiClient` to a constant so that the same client can be used everywhere.

```ruby
AI = AiClient.new 'gpt-4o'
...
AI.chat "do something with this #{stuff}"
...
AI.speak "warning  Will Robinson! #{bad_things_happened}"
...
```

Using the constant for the instance allows you to reference the same client instance inside any method through out your application.  Of course it does not apply to only one instance.  You could assign multiple instances for different models/providers.  For example you could have `AI` for your primary client and `AIbackup` for a fallback client in case you have a problem on the primary; or, maybe `Vectorizer` as a client name tied to a model specializing in embedding vectorization.


## OmniAI and OpenRouter

Both OmniAI and OpenRouter have similar goals - to provide a common interface to multiple providers and LLMs.  OmniAI is a Ruby gem that supports specific providers directly using a common-ish API.  You incur costs directly from those providers for which you have individual API keys (aka access tokens.) OpenRouter, on the other hand, is a web service that also establishes a common API for many providers and models; however, OpenRouter adds a small fee on top of the fee charged by those providers.  You trade off cost for flexibility.  With OpenRouter you only need one API key (OPEN_ROUTER_API_KEY) to access all of its supported services.

The advantage of AiClient is that you have the added flexibility to choose on a client by client bases where you want your model to be processed.  You get free local processing through Ollama and LocalAI.  You get less costly direct access to some providers via OmniAI.  You get slightly more costly wide-spread access via OpenRouter

## The `chatbot` Method

Just for grins I added a `chatbot` instance method.  Try this out in IRB:

```ruby
require 'ai_client'
ai = AiClient.new
ai.chatbot
```

## Contributing

I can sure use your help.  This industry is moving faster than I can keep up with.  If you have a bug fix or new feature idea then have at it.  Send me a pull request so we all can benefit from your efforts.

If you only have time to report a bug, that's fine.  Just create an issue in this repo.

## License

The gem is available as open source under the terms of the [MIT License](https://opensource.org/licenses/MIT).
